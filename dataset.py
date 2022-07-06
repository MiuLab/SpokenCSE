import torch
import numpy as np
import os
from torch.utils.data import Dataset
import jiwer


def cal_wer(ground_truth, hypothesis):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.SentencesToListOfWords(word_delimiter=" ")
    ])

    wer = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    return wer


def get_examples(tokenizer, text, max_length=64, is_phone=False):
    text_batch_encoding = tokenizer(
        text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length
    )
    text_examples = []
    for i in range(len(text)):
        example = {
            "input_ids": torch.tensor(text_batch_encoding["input_ids"][i]),
            "token_type_ids": torch.tensor(
                [1 if is_phone else 0 for _ in range(len(text_batch_encoding["input_ids"][i]))]
            ),
            "attention_mask": torch.tensor(text_batch_encoding["attention_mask"][i]),
        }
        text_examples.append(example)
    return text_examples


def get_mixed_token_type_ids(batch_encoding, sep_id=2):
    token_type_ids = []
    N = len(batch_encoding['input_ids'])
    for i in range(N):
        input_ids = batch_encoding['input_ids'][i]
        pad_index = input_ids.index(sep_id) + 1
        token_type_id = [0 for _ in range(pad_index)] + [1 for _ in range(pad_index, len(input_ids))]
        assert len(token_type_id) == len(input_ids)
        token_type_ids.append(token_type_id)
    return token_type_ids


def get_combined_examples(tokenizer, text, phoneme_text, max_length=64):
    batch_encoding = tokenizer(
        text, phoneme_text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length
    )
    token_type_ids = get_mixed_token_type_ids(batch_encoding, sep_id=tokenizer.sep_token_id)
    combined_examples = []
    for i in range(len(text)):
        example = {
            "input_ids": torch.tensor(batch_encoding["input_ids"][i]),
            "token_type_ids": torch.tensor(token_type_ids[i]),
            "attention_mask": torch.tensor(batch_encoding["attention_mask"][i]),
        }
        combined_examples.append(example)
    return combined_examples


def get_contrastive_examples(tokenizer, golden_text, text, phoneme_text=None, golden_phoneme_text=None, max_length=64):
    print('contrative dataset')
    print('preparing text examples ...')
    hypo_text_examples = get_examples(tokenizer, text, max_length=max_length)
    golden_text_examples = get_examples(tokenizer, golden_text, max_length=max_length)
    text_examples = []
    for i in range(len(text)):
        example = {}
        for k, v in hypo_text_examples[i].items():
            example[k] = v
        for k, v in golden_text_examples[i].items():
            example["golden_{}".format(k)] = v
        text_examples.append(example)

    if phoneme_text is None or golden_phoneme_text is None:
        return text_examples, [], []

    print('preparing phone examples ...')
    hypo_phone_examples = get_examples(tokenizer, phoneme_text, is_phone=True, max_length=max_length)
    golden_phone_examples = get_examples(tokenizer, golden_phoneme_text, is_phone=True, max_length=max_length)
    phone_examples = []
    for i in range(len(phoneme_text)):
        example = {}
        for k, v in hypo_phone_examples[i].items():
            example[k] = v
        for k, v in golden_phone_examples[i].items():
            example["golden_{}".format(k)] = v
        phone_examples.append(example)

    print('preparing combined examples ...')
    hypo_combined_examples = get_combined_examples(tokenizer, text, phoneme_text, max_length=max_length)
    golden_combined_examples = get_combined_examples(tokenizer, golden_text, golden_phoneme_text, max_length=max_length)
    combined_examples = []
    for i in range(len(text)):
        example = {}
        for k, v in hypo_combined_examples[i].items():
            example[k] = v
        for k, v in golden_combined_examples[i].items():
            example["golden_{}".format(k)] = v
        combined_examples.append(example)
    return text_examples, phone_examples, combined_examples


class ContrastiveDataset(Dataset):
    def __init__(self, tokenizer, dataset, target,
                 max_length=64, use_phoneme=False, phoneme_only=False):
        if not use_phoneme and not phoneme_only:
            max_length = 32

        systems = ['google', 'wav2vec2']
        if target not in systems:
            raise ValueError('provide proper target: {}'.format(systems))
        self.text = [s[target]['sentence'] for s in dataset if target in s.keys()]
        self.phoneme_text = [s[target]['phonemes'].replace('.', ' ').upper() for s in dataset if target in s.keys()]
        self.golden_text = [s['golden'] for s in dataset if target in s.keys()]
        self.golden_phoneme_text = [
            s['golden_phonemes'].replace('.', ' ').upper() for s in dataset if target in s.keys()
        ]
        assert len(self.phoneme_text) == len(self.text)
        assert len(self.golden_text) == len(self.text)
        assert len(self.golden_phoneme_text) == len(self.text)

        self.text_examples, self.phone_examples, self.combined_examples = get_contrastive_examples(
            tokenizer, self.text, self.golden_text, self.phoneme_text, self.golden_phoneme_text, max_length=max_length)

        print('number of data: text {}, phoneme {}, combined {}'.format(
            len(self.text_examples), len(self.phone_examples), len(self.combined_examples)
        ))
        if phoneme_only:
            print('only using phoneme data')
            self.examples = self.phone_examples
        elif use_phoneme:
            print('using all data')
            self.examples = self.text_examples + self.phone_examples + self.combined_examples
        else:
            print('only using text data')
            self.examples = self.text_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class PhonemeBERTContrastiveDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, split='train', max_length=64,
                 use_phoneme=False, phoneme_only=False):

        dataset_dir = os.path.join(dataset_dir, split)

        with open(os.path.join(dataset_dir, "text_original.txt"), 'r') as f:
            self.golden_text = [line.replace('\n', '') for line in f.readlines()]

        text_path = os.path.join(dataset_dir, "en.classification.txt")
        if not os.path.exists(text_path):
            text_path = os.path.join(dataset_dir, "en.parallel.txt")
        with open(text_path, 'r') as f:
            self.text = [line.replace('\n', '') for line in f.readlines()]
        assert len(self.golden_text) == len(self.text)

        if use_phoneme:
            golden_phone_path = os.path.join(dataset_dir, "phoneme_original.txt")
            if not os.path.exists(golden_phone_path):
                raise NotImplementedError('Can not use golden phoneme of golden text')
            with open(golden_phone_path, 'r') as f:
                self.golden_phoneme_text = [line.replace('\n', '').replace('.', ' ').upper() for line in f.readlines()]

            phone_path = os.path.join(dataset_dir, "ph.parallel.txt")
            with open(phone_path, 'r') as f:
                self.phoneme_text = [line.replace('\n', '').replace('.', ' ').upper() for line in f.readlines()]
            assert len(self.phoneme_text) == len(self.text)
            assert len(self.golden_phoneme_text) == len(self.text)
        else:
            self.phoneme_text = None
            self.golden_phoneme_text = None

        self.text_examples, self.phone_examples, self.combined_examples = get_contrastive_examples(
            tokenizer, self.text, self.golden_text, self.phoneme_text, self.golden_phoneme_text, max_length=max_length)

        print('number of data: text {}, phoneme {}, combined {}'.format(
            len(self.text_examples), len(self.phone_examples), len(self.combined_examples)
        ))
        if phoneme_only:
            print('only using phoneme data')
            self.examples = self.phone_examples
        elif use_phoneme:
            print('using all data')
            self.examples = self.text_examples + self.phone_examples + self.combined_examples
        else:
            print('only using text data')
            self.examples = self.text_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_np(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    return x_log, x_softmax


def separate_phonemebert_test_set(test_set):
    wers = []
    for i in range(len(test_set.text)):
        golden, text = test_set.golden[i], test_set.text[i]
        wers.append(cal_wer(golden, text))
    wer_qs = np.percentile(wers, (25, 50, 75), interpolation='midpoint')
    print("total, wer mean, std, q1, q2, q3")
    print(len(wers), np.mean(wers), np.std(wers), wer_qs)
    groups = [[] for i in range(len(wer_qs)+1)]
    for i, wer in enumerate(wers):
        group_id = 0
        while wer > wer_qs[group_id]:
            group_id += 1
            if group_id > len(wer_qs)-1:
                break
        groups[group_id].append(i)

    test_datasets = []
    for indexes in groups:
        test_datasets.append(PhonemeBERTTestDataset(test_set.examples, indexes))
    print([len(d) for d in test_datasets])

    return test_datasets


class PhonemeBERTTestDataset(Dataset):
    def __init__(self, examples, indexes):
        self.examples = [examples[k] for k in indexes]

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


class PhonemeBERTDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, use_golden=False,
                 use_phoneme=False, max_length=64, id2label=None, use_label=True,
                 pretraining=False):

        with open(os.path.join(dataset_dir, "text_original.txt"), 'r') as f:
            self.golden = [line.replace('\n', '') for line in f.readlines()]
        if use_golden:
            self.text = self.golden
        else:
            text_path = os.path.join(dataset_dir, "en.classification.txt")
            if pretraining:
                text_path = os.path.join(dataset_dir, "en.parallel.txt")
            with open(text_path, 'r') as f:
                self.text = [line.replace('\n', '') for line in f.readlines()]

        if use_phoneme:
            if use_golden:
                phone_path = os.path.join(dataset_dir, "phoneme_original.txt")
                if not os.path.exists(phone_path):
                    raise NotImplementedError('Can not use golden phoneme of golden text')
            else:
                phone_path = os.path.join(dataset_dir, "ph.classification.txt")
            if pretraining:
                phone_path = os.path.join(dataset_dir, "ph.parallel.txt")
            with open(phone_path, 'r') as f:
                self.phoneme_text = [line.replace('\n', '').replace('.', ' ').upper() for line in f.readlines()]
            self.examples = get_combined_examples(tokenizer, self.text, self.phoneme_text, max_length=max_length)
            if pretraining:
                text_examples = get_examples(tokenizer, self.text, max_length=max_length)
                phone_examples = get_examples(tokenizer, self.phoneme_text, is_phone=True, max_length=max_length)
                self.examples += text_examples + phone_examples
        else:
            self.examples = get_examples(tokenizer, self.text, max_length=max_length)

        if use_label:
            label_path = os.path.join(dataset_dir, "labels.classification.txt")
            label_strs = []
            self.label_set = set()
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
                for line in label_lines:
                    label = line.replace('\n', '')
                    if label not in self.label_set:
                        self.label_set.add(label)
                    label_strs.append(label)
            if id2label:
                self.id2label = id2label
            else:
                self.id2label = list(self.label_set)
            self.label2id = {v: i for i, v in enumerate(self.id2label)}
            self.label = [self.label2id[label] for label in label_strs]
            for i in range(len(self.examples)):
                self.examples[i]['label'] = self.label[i]
                self.examples[i]["pseudo_label"] = np.eye(len(self.id2label))[self.label[i]]

        print('dataset dir: {}'.format(dataset_dir))
        print('dataset size: {}'.format(len(self.examples)))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def update_pseudo_label(self, pred, p, verbose=False):

        logits, labels, _ = pred
        loss, probs = cross_entropy_np(logits, labels)
        assert len(probs) == len(self)
        assert len(loss) == len(self)
        for i in range(len(self)):
            self.examples[i]["pseudo_label"] = probs[i]


class SlurpDataset(Dataset):
    def __init__(self, tokenizer, meta, dataset, target,
                 use_phoneme=False, max_length=64, use_label=True, low_resource=-1):
        # feainput_ids: ['attention_mask', 'input_ids', 'label', 'text'],
        if not use_phoneme:
            max_length = 32
        self.meta = meta
        self.scenario_label_list = meta['scenario']
        self.action_label_list = meta['action']
        self.scenario_label = []
        self.action_label = []

        self.golden = []
        self.text = []
        if "golden" in target:
            self.text += [s['golden'] for s in dataset]
            self.golden += [s['golden'] for s in dataset]
            self.scenario_label += [s["scenario"] for s in dataset]
            self.action_label += [s["action"] for s in dataset]
        systems = ['google', 'wav2vec2']
        self.wer = {}
        for system in systems:
            if system in target:
                self.golden += [s['golden'] for s in dataset if system in s.keys()]
                self.text += [s[system]['sentence'] for s in dataset if system in s.keys()]
                self.wer[system] = [s[system]['wer'] for s in dataset if system in s.keys()]
                self.scenario_label += [s["scenario"] for s in dataset if system in s.keys()]
                self.action_label += [s["action"] for s in dataset if system in s.keys()]

        self.phoneme_text = []
        self.golden_phoneme = []
        if use_phoneme:
            if "golden" in target:
                self.golden_phoneme += [s['golden_phonemes'].replace('.', ' ').upper() for s in dataset]
                self.phoneme_text += [s['golden_phonemes'].replace('.', ' ').upper() for s in dataset]

            for system in systems:
                if system in target:
                    self.golden_phoneme += [
                        s['golden_phonemes'].replace('.', ' ').upper() for s in dataset if system in s.keys()
                    ]
                    self.phoneme_text += [
                        s[system]['phonemes'].replace('.', ' ').upper() for s in dataset if system in s.keys()
                    ]
            examples = get_combined_examples(tokenizer, self.text, self.phoneme_text, max_length=max_length)

        else:
            examples = get_examples(tokenizer, self.text, max_length=max_length)

        assert len(examples) == len(self.action_label)
        assert len(examples) == len(self.text)
        assert len(self.scenario_label) == len(self.action_label)
        if use_phoneme:
            assert len(self.text) == len(self.phoneme_text)
        print('single len: {}, total len: {}'.format(len(dataset), len(self.text)))

        if use_label:
            self.examples = []
            for i, example in enumerate(examples):
                example["scenario_label"] = self.scenario_label[i]
                example["action_label"] = self.action_label[i]
                example["pseudo_scenario_label"] = np.eye(len(self.scenario_label_list))[self.scenario_label[i]]
                example["pseudo_action_label"] = np.eye(len(self.action_label_list))[self.action_label[i]]
                self.examples.append(example)
        else:
            self.examples = examples
        if low_resource > 0:
            self.examples = []
            idx = low_resource
            while len(self.examples) < low_resource:
                self.examples.append(examples[idx])
                idx = (idx + low_resource * low_resource) % len(examples)
            assert len(self.examples) == low_resource

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def update_pseudo_label(self, pred, p, verbose=False):

        logits, labels, _ = pred
        scenario_logits, action_logits = logits
        scenario_labels, action_labels = labels
        scenario_loss, scenario_probs = cross_entropy_np(scenario_logits, scenario_labels)
        action_loss, action_probs = cross_entropy_np(action_logits, action_labels)
        assert len(scenario_logits) == len(self)
        assert len(scenario_loss) == len(self)

        for i in range(len(self)):
            self.examples[i]["pseudo_scenario_label"] = scenario_probs[i]
            self.examples[i]["pseudo_action_label"] = action_probs[i]


class SlurpCorrectionDataset(Dataset):
    def __init__(self, tokenizer, dataset, target,
                 use_phoneme=False, max_length=64):
        # feainput_ids: ['attention_mask', 'input_ids', 'label', 'text'],
        if not use_phoneme:
            max_length = 32
        self.text = []
        self.golden = []
        systems = ['google', 'wav2vec2']
        self.wer = {}
        for system in systems:
            if system in target:
                self.golden += [s['golden'] for s in dataset if system in s.keys()]
                self.text += [s[system]['sentence'] for s in dataset if system in s.keys()]
                self.wer[system] = [s[system]['wer'] for s in dataset if system in s.keys()]

        self.phoneme_text = []
        self.golden_phoneme = []
        if use_phoneme:
            for system in systems:
                if system in target:
                    self.golden_phoneme += [
                        s['golden_phonemes'].replace('.', ' ').upper() for s in dataset if system in s.keys()
                    ]
                    self.phoneme_text += [
                        s[system]['phonemes'].replace('.', ' ').upper() for s in dataset if system in s.keys()
                    ]
            examples = get_combined_examples(tokenizer, self.text, self.phoneme_text, max_length=max_length)
            golden_examples = get_combined_examples(tokenizer, self.golden, self.golden_phoneme, max_length=max_length)

        else:
            examples = get_examples(tokenizer, self.text, max_length=max_length)
            golden_examples = get_examples(tokenizer, self.golden, max_length=max_length)

        assert len(examples) == len(self.text)
        assert len(golden_examples) == len(self.text)
        if use_phoneme:
            assert len(self.text) == len(self.phoneme_text)
        print('single len: {}, total len: {}'.format(len(dataset), len(self.text)))

        self.examples = []
        for i in range(len(examples)):
            example = {}
            for k in ["input_ids", "attention_mask"]:
                example[k] = examples[i][k]
                example["labels"] = golden_examples[i]["input_ids"]
            self.examples.append(example)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    pass
