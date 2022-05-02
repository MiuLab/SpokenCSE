import json
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import argparse

from dataset import ContrastiveDataset, PhonemeBERTContrastiveDataset
from collator import DataCollatorWithPaddingMLM
from loss_functions import ContrastiveLoss


class TwoPassNet(nn.Module):
    def __init__(self, args):
        super(TwoPassNet, self).__init__()
        self.args = args
        self.bert = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        self.bert.roberta.config.type_vocab_size = 2
        self.bert.roberta.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.roberta.config.hidden_size)
        self.bert.roberta.embeddings.token_type_embeddings.weight.data.normal_(
            mean=0.0, std=self.bert.roberta.config.initializer_range
        )

        self.cl_criterion = ContrastiveLoss(temperature=args.temperature)
        self.mlm_criterion = nn.CrossEntropyLoss()
        self.do_mlm = args.input_mask_ratio > 0

    def separate_inputs(self, **inputs):
        hypo_inputs = {}
        golden_inputs = {}
        masked_hypo_inputs = {}
        masked_golden_inputs = None
        for k, v in inputs.items():
            if "label" in k:
                continue
            elif "input_ids" in k:
                if k == "masked_input_ids":
                    masked_hypo_inputs[k.replace('masked_', '')] = v
                    # hypo_inputs[k.replace('masked_', '')] = v
                elif k == "masked_golden_input_ids":
                    masked_golden_inputs = v
                elif k == "golden_input_ids":
                    golden_inputs[k.replace('golden_', '')] = v
                else:
                    hypo_inputs[k] = v
            elif "golden" in k:
                golden_inputs[k.replace('golden_', '')] = v
            else:
                hypo_inputs[k] = v
                masked_hypo_inputs[k] = v

        if self.args.mask_golden:
            if masked_golden_inputs is None:
                raise ValueError('please use mlm and probability > 0 for mask_golden')
            golden_inputs["input_ids"] = masked_golden_inputs

        if "input_ids" not in masked_hypo_inputs and self.do_mlm:
            print('no masked input')
            exit()
        if "input_ids" not in hypo_inputs:
            print('no clean input')
            exit()
        return hypo_inputs, golden_inputs, masked_hypo_inputs

    def forward(self, **inputs):
        hypo_inputs, golden_inputs, masked_hypo_inputs = self.separate_inputs(**inputs)

        # do contrastive
        hypo_output = self.bert(**hypo_inputs, output_hidden_states=True)
        hypo_last_hidden = hypo_output.hidden_states[0][:, 0]
        # hypo_last_hidden = torch.mean(hypo_output.last_hidden_state, dim=1)
        if self.args.self_only:
            golden_output = self.bert(**hypo_inputs, output_hidden_states=True)
        else:
            golden_output = self.bert(**golden_inputs, output_hidden_states=True)
        golden_last_hidden = golden_output.hidden_states[0][:, 0]
        # golden_last_hidden = torch.mean(golden_output.last_hidden_state, dim=1)
        loss = self.cl_criterion(hypo_last_hidden, golden_last_hidden)

        # do mlm
        masked_hypo_output = self.bert(**masked_hypo_inputs, labels=inputs['masked_labels'])
        loss += self.args.Lambda * masked_hypo_output.loss
        return loss, hypo_last_hidden, golden_last_hidden, masked_hypo_output


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, hypo_last_hidden, golden_last_hidden, masked_hypo_output = model(**inputs)
        outputs = {'hypo': hypo_last_hidden, 'golden': golden_last_hidden}
        return loss if not return_outputs else (loss, outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default='datasets/slurp/slurp_with_oracle_test.json', type=str, help="dataset.json path"
    )
    parser.add_argument(
        "--model_name_or_path", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--tokenizer_name", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--target", default='google', type=str, help="google or wav2vec2"
    )
    parser.add_argument(
        "--output_dir", default='runs/contrastive_pretrain', type=str, help="dir to save model"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "--max_steps", default=10000, type=int, help="total number of update steps"
    )
    parser.add_argument(
        "--save_steps", default=1000, type=int, help="eval, log & save every [this] steps"
    )
    parser.add_argument(
        "--train_bsize", default=32, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_bsize", default=32, type=int, help="evaluation batch size"
    )
    parser.add_argument(
        "--patience", default=3, type=int, help="early stopping patience"
    )
    parser.add_argument(
        "--use_phoneme", action='store_true', help="use phoneme + text sequence"
    )
    parser.add_argument(
        "--phoneme_only", action='store_true', help="use phoneme sequence"
    )
    parser.add_argument(
        "--mask_golden", action='store_true', help="contrastive with masked golden"
    )
    parser.add_argument(
        "--self_only", action='store_true', help="contrastive with self"
    )
    parser.add_argument(
        "--use_phonemebert", action='store_true', help="use phonemebert dataset"
    )
    parser.add_argument(
        "--Lambda", default=1, type=float, help="mlm loss ratio vs contrastive"
    )
    parser.add_argument(
        "--input_mask_ratio", default=0.15, type=float, help="mlm ratio when training"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="model hidden dropout"
    )
    parser.add_argument(
        "--temperature", default=0.2, type=float, help="temperature for contrastive similarity"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.input_mask_ratio > 0

    # Dataset
    print('reading dataset')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_phonemebert:
        print('using phonemebert dataset')
        train_dataset = PhonemeBERTContrastiveDataset(
            tokenizer, args.dataset, split='train',
            use_phoneme=args.use_phoneme,
            phoneme_only=args.phoneme_only,
        )
        eval_dataset = PhonemeBERTContrastiveDataset(
            tokenizer, args.dataset, split='valid',
            use_phoneme=args.use_phoneme,
            phoneme_only=args.phoneme_only,
        )

    else:
        with open(args.dataset, 'r') as f:
            datasets = json.load(f)

        train_dataset = ContrastiveDataset(
            tokenizer, datasets['train'],
            args.target,
            use_phoneme=args.use_phoneme,
            phoneme_only=args.phoneme_only,
        )
        eval_dataset = ContrastiveDataset(
            tokenizer, datasets['devel'],
            args.target,
            use_phoneme=args.use_phoneme,
            phoneme_only=args.phoneme_only,
        )
    data_collator = DataCollatorWithPaddingMLM(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.input_mask_ratio
    )

    steps = args.save_steps
    model = TwoPassNet(args)

    # Train model
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=steps,
        logging_strategy="steps",
        logging_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.eval_bsize,
        # weight_decay=0.01,               # strength of weight decay
        seed=args.seed,
        label_names=["golden_input_ids"]
    )
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.model.bert.save_pretrained(args.output_dir.replace('runs', 'models'))
