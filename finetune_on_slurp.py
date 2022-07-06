import json
import os
import torch
import torch.nn as nn
from datasets import load_metric
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
import argparse

from dataset import SlurpDataset
from loss_functions import SupervisedContrastiveLoss, KLWithSoftLabelLoss


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scenario_logits, action_logits = logits
    scenario_labels, action_labels = labels
    scenario_predictions = np.argmax(scenario_logits, axis=-1)
    scenario_metric = metric.compute(
        predictions=scenario_predictions,
        references=scenario_labels
    )
    action_predictions = np.argmax(action_logits, axis=-1)
    action_metric = metric.compute(
        predictions=action_predictions,
        references=action_labels
    )

    joint_predictions = scenario_predictions * 46 + action_predictions
    joint_labels = scenario_labels * 46 + action_labels
    joint_metric = metric.compute(
        predictions=joint_predictions,
        references=joint_labels
    )
    return {
        "scenario_accuracy": scenario_metric["accuracy"],
        "action_accuracy": action_metric["accuracy"],
        "joint_accuracy": joint_metric["accuracy"],
    }


class TwoHeadNet(torch.nn.Module):
    def __init__(self, meta, args):
        super(TwoHeadNet, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.bert.config.type_vocab_size = 2
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.bert.config.hidden_dropout_prob = args.dropout
        self.scenario_mlp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.scenario_head = nn.Linear(self.bert.config.hidden_size, len(meta["scenario"]))
        self.action_mlp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.action_head = nn.Linear(self.bert.config.hidden_size, len(meta["action"]))

    def forward(self, **inputs):

        scenario_label = inputs.pop('scenario_label')
        action_label = inputs.pop('action_label')
        pseudo_scenario_label = inputs.pop('pseudo_scenario_label')
        pseudo_action_label = inputs.pop('pseudo_action_label')

        bert_output = self.bert(**inputs)

        # last_hidden = torch.mean(bert_output.last_hidden_state, dim=1)
        last_hidden = bert_output.last_hidden_state[:, 0]
        scenario_logits = self.scenario_head(self.scenario_mlp(last_hidden))
        action_logits = self.action_head(self.action_mlp(last_hidden))

        """ Calculate Loss """
        ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        scenario_loss = ce_loss_fn(scenario_logits, scenario_label)
        action_loss = ce_loss_fn(action_logits, action_label)
        loss = scenario_loss + action_loss

        if self.args.use_pseudo:
            # pseudo_loss = pseudo_scenario_loss + pseudo_action_loss
            if self.args.pseudo_weight > 100:
                loss = 0
                kd_loss_fn = KLWithSoftLabelLoss(self.args.pseudo_label_temperature, 1)
            else:
                kd_loss_fn = KLWithSoftLabelLoss(self.args.pseudo_label_temperature, self.args.pseudo_weight)
            pseudo_scenario_loss = kd_loss_fn(scenario_logits, pseudo_scenario_label)
            pseudo_action_loss = kd_loss_fn(action_logits, pseudo_action_label)
            loss += pseudo_scenario_loss + pseudo_action_loss

        if self.args.use_contrastive:
            contrastive_loss_fn = SupervisedContrastiveLoss(temperature=self.args.contrastive_temperature)
            if self.args.use_pseudo:
                loss += self.args.contrastive_weight * contrastive_loss_fn(
                    last_hidden, scenario_label, soft_labels=pseudo_scenario_label)
                loss += self.args.contrastive_weight * contrastive_loss_fn(
                    last_hidden, scenario_label, soft_labels=pseudo_action_label)
            else:
                loss += self.args.contrastive_weight * contrastive_loss_fn(last_hidden, scenario_label)
                loss += self.args.contrastive_weight * contrastive_loss_fn(last_hidden, action_label)

        return loss, scenario_logits, action_logits


class UpdatePseudoLabelCallback(TrainerCallback):
    def __init__(self, trainer, warmup=0) -> None:
        super().__init__()
        self._trainer = trainer
        self.warmup = warmup

    def on_epoch_end(self, args, state, control, **kwargs):
        pred = self._trainer.predict(test_dataset=self._trainer.train_dataset)
        print("\ntrain metric: ", pred[2])

        if state.epoch > self.warmup:
            percent = max(5 * state.epoch, 30)
            self._trainer.train_dataset.update_pseudo_label(pred, 100-percent, verbose=True)


class SlurpTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, scenario_logits, action_logits = model(**inputs)
        outputs = {'scenario': scenario_logits, 'action': action_logits}
        return loss if not return_outputs else (loss, outputs)


def save_prediction(args, pred, test_dataset):
    logits, labels = pred[0], pred[1]
    scenario_logits, action_logits = logits
    scenario_labels, action_labels = labels
    scenario_predictions = np.argmax(scenario_logits, axis=-1)
    action_predictions = np.argmax(action_logits, axis=-1)
    with open(os.path.join(args.output_dir, "output.npy"), 'wb') as f:
        output = [scenario_predictions, action_predictions, scenario_labels, action_labels]
        np.save(f, output)
    with open(os.path.join(args.output_dir, "test_data.json"), "w") as f:
        test_data = {
            "text": test_dataset.text,
            "phoneme_text": test_dataset.phoneme_text,
            "golden": test_dataset.golden,
            "golden_phoneme": test_dataset.golden_phoneme,
        }
        json.dump(test_data, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default='datasets/slurp/slurp_with_oracle_test.json', type=str, help="dataset.json path"
    )
    parser.add_argument(
        "--meta", default='datasets/slurp/golden/meta.json', type=str, help="meta.json path"
    )
    parser.add_argument(
        "--train_target", default='google', type=str, help="golden and/or google and/or wav2vec2"
    )
    parser.add_argument(
        "--eval_target", default='google', type=str, help="golden and/or google and/or wav2vec2"
    )
    parser.add_argument(
        "--model_name_or_path", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--tokenizer_name", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--output_dir", default='runs/finetune', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--log_dir", default='logs/', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--log_name", default='finetune', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "-n", default=1, type=int, help="num to run & average"
    )
    parser.add_argument(
        "--max_epoch", default=10, type=int, help="total number of epoch"
    )
    parser.add_argument(
        "--train_bsize", default=64, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_bsize", default=64, type=int, help="evaluation batch size"
    )
    parser.add_argument(
        "--patience", default=3, type=int, help="early stopping patience"
    )
    parser.add_argument(
        "--use_phoneme", action='store_true', help="use phoneme + text sequence"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="model hidden dropout"
    )
    parser.add_argument(
        "--save_predict", action='store_true', help="save prediction & test text"
    )
    parser.add_argument(
        "--use_contrastive", action='store_true', help="supervised contrastive objective"
    )
    parser.add_argument(
        "--contrastive_temperature", default=0.2, type=float, help="contrastive temperature"
    )
    parser.add_argument(
        "--contrastive_weight", default=0.1, type=float, help="contrastive loss weight vs classification"
    )
    parser.add_argument(
        "--use_pseudo", action='store_true', help="train from pseudo label"
    )
    parser.add_argument(
        "--pseudo_label_temperature", default=5, type=float, help="contrastive temperature"
    )
    parser.add_argument(
        "--pseudo_weight", default=10, type=float, help="contrastive loss weight vs classification"
    )
    parser.add_argument(
        "--low_resource", default=-1, type=int, help="if > 0, subsample training set to this number"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Dataset
    print('reading dataset')
    with open(args.meta, 'r') as f:
        meta = json.load(f)
    with open(args.dataset, 'r') as f:
        datasets = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    all_preds = []
    for n in range(args.n):
        train_dataset = SlurpDataset(
            tokenizer, meta, datasets['train'], target=args.train_target,
            use_phoneme=args.use_phoneme, low_resource=args.low_resource)
        if 'oracle_eval' in datasets:
            print('using oracle eval set')
            eval_set = datasets['oracle_eval']
        else:
            eval_set = datasets['devel']
        eval_dataset = SlurpDataset(
            tokenizer, meta, eval_set, target=args.eval_target,
            use_phoneme=args.use_phoneme)
        print('start training: {}'.format(n))
        model = TwoHeadNet(meta, args)
        # Train model
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_joint_accuracy",
            num_train_epochs=args.max_epoch,
            per_device_train_batch_size=args.train_bsize,
            per_device_eval_batch_size=args.eval_bsize,
            # weight_decay=0.01,               # strength of weight decay
            seed=args.seed + n,
            label_names=["scenario_label", "action_label"]
        )
        trainer = SlurpTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=0.002)]
        )
        if args.use_pseudo:
            trainer.add_callback(UpdatePseudoLabelCallback(trainer))
        trainer.train()

        # test
        test_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer.data_collator = test_data_collator
        if args.eval_target == 'google':
            test_datasets = []
            overall = []
            for i, d in enumerate(datasets['google_test']):
                overall += d
                dataset = SlurpDataset(
                    tokenizer, meta, d, target=args.eval_target,
                    use_phoneme=args.use_phoneme)
                test_datasets.append(dataset)
            dataset = SlurpDataset(
                tokenizer, meta, overall, target=args.eval_target,
                use_phoneme=args.use_phoneme)
            test_datasets.append(dataset)
        elif args.eval_target == 'wav2vec2':
            test_datasets = []
            overall = []
            for i, d in enumerate(datasets['wave2vec_test']):
                overall += d
                dataset = SlurpDataset(
                    tokenizer, meta, d, target=args.eval_target,
                    use_phoneme=args.use_phoneme)
                test_datasets.append(dataset)
            dataset = SlurpDataset(
                tokenizer, meta, overall, target=args.eval_target,
                use_phoneme=args.use_phoneme)
            test_datasets.append(dataset)
        else:
            test_dataset = SlurpDataset(
                tokenizer, meta, datasets['test'], target=args.eval_target,
                use_phoneme=args.use_phoneme)
            test_datasets = [test_dataset]

        if args.n == 1 and args.save_predict:
            # we do this for collecting model prediction on raw data,
            # and subsample the dataset with agreed pseudo label
            pred = trainer.predict(test_dataset=test_dataset)
            save_prediction(args, pred, test_dataset)

        keys = ['test_loss', 'test_scenario_accuracy', 'test_action_accuracy', 'test_joint_accuracy']
        preds = []
        for i, test_dataset in enumerate(test_datasets):
            pred = trainer.predict(test_dataset=test_dataset)
            pred = {k: pred[2][k] for k in keys}
            preds.append(pred)
        all_preds.append(preds)

        trainer.save_model(args.output_dir)
        trainer = None

    predictions = {}
    for preds in all_preds:
        for i, pred in enumerate(preds):
            for k, v in pred.items():
                key = k+'-{}'.format(i)
                predictions[key] = predictions.get(key, []) + [np.round(v, 4)]
                # predictions[k].append(np.round(v, 4))

    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir, "{}.log".format(args.log_name))
    with open(logfile, 'w') as f:
        f.write("{:>30}\t{:>8}\t{:>8}\t{}\n".format('metric', 'mean', 'std', 'values'))
        print("{:>30}\t{:>8}\t{:>8}\t{}".format('metric', 'mean', 'std', 'values'))
        for k, v in predictions.items():
            mean = np.round(np.mean(v), 4)
            std = np.round(np.std(v), 4)
            print("{:>30}\t{:>8}\t{:>8}\t{}".format(k, mean, std, v))
            f.write("{:>30}\t{:>8}\t{:>8}\t{}\n".format(k, mean, std, v))
