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

from dataset import PhonemeBERTDataset, separate_phonemebert_test_set
from collator import DataCollatorWithPaddingMLM
from loss_functions import SupervisedContrastiveLoss, KLWithSoftLabelLoss


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(
        predictions=predictions,
        references=labels
    )
    return {
        "accuracy": metrics["accuracy"],
    }


class Net(torch.nn.Module):
    def __init__(self, args, num_labels):
        super(Net, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.bert.config.type_vocab_size = 2
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.bert.config.hidden_dropout_prob = args.dropout
        self.mlp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.clf_head = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **inputs):

        label = inputs.pop('labels')
        pseudo_label = inputs.pop('pseudo_label')
        bert_output = self.bert(**inputs)

        # last_hidden = torch.mean(bert_output.last_hidden_state, dim=1)
        last_hidden = bert_output.last_hidden_state[:, 0]
        logits = self.clf_head(self.mlp(last_hidden))

        """ Calculate Loss """
        ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss = ce_loss_fn(logits, label)

        if self.args.use_pseudo:
            kd_loss_fn = KLWithSoftLabelLoss(self.args.pseudo_label_temperature, self.args.pseudo_weight)
            pseudo_loss = kd_loss_fn(logits, pseudo_label)
            loss += pseudo_loss
        if self.args.use_contrastive:
            contrastive_loss_fn = SupervisedContrastiveLoss(temperature=self.args.contrastive_temperature)
            if self.args.use_pseudo:
                loss += self.args.contrastive_weight * contrastive_loss_fn(
                    last_hidden, label, soft_labels=pseudo_label)
            else:
                loss += self.args.contrastive_weight * contrastive_loss_fn(last_hidden, label)

        return loss, logits


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, logits = model(**inputs)
        outputs = {'logits': logits}
        return loss if not return_outputs else (loss, outputs)


def save_prediction(args, pred, test_dataset):
    logits, labels = pred[0], pred[1]
    predictions = np.argmax(logits, axis=-1)
    with open(os.path.join(args.output_dir, "output.npy"), 'wb') as f:
        output = [predictions, labels]
        np.save(f, output)
    with open(os.path.join(args.output_dir, "test_data.json"), "w") as f:
        test_data = {
            "text": test_dataset.text,
            "phoneme_text": test_dataset.phoneme_text,
            "golden": test_dataset.golden,
            "id2label": test_dataset.id2label,
            # "golden_phoneme": test_dataset.golden_phoneme,
        }
        json.dump(test_data, f)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default='../Phoneme-BERT/phomeme-bert-data/downstream-datasets/atis',
        type=str, help="dataset directory"
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
        "-n", default=5, type=int, help="num to run & average"
    )
    parser.add_argument(
        "--max_epoch", default=20, type=int, help="total number of epoch"
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
        "--train_golden", action='store_true', help="train on golden transcript"
    )
    parser.add_argument(
        "--eval_golden", action='store_true', help="eval on golden transcript"
    )
    parser.add_argument(
        "--use_phoneme", action='store_true', help="use phoneme + text sequence"
    )
    parser.add_argument(
        "--input_mask_ratio", default=0, type=float, help="mlm ratio when training"
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Dataset
    print('reading dataset')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_dir = os.path.join(args.dataset_dir, 'train')
    eval_dir = os.path.join(args.dataset_dir, 'valid')
    test_dir = os.path.join(args.dataset_dir, 'test')
    train_dataset = PhonemeBERTDataset(
        tokenizer, train_dir, use_golden=args.train_golden,
        use_phoneme=args.use_phoneme
    )
    eval_dataset = PhonemeBERTDataset(
        tokenizer, eval_dir, use_golden=args.eval_golden,
        use_phoneme=args.use_phoneme,
        id2label=train_dataset.id2label
    )
    test_dataset = PhonemeBERTDataset(
        tokenizer, test_dir, use_golden=args.eval_golden,
        use_phoneme=args.use_phoneme,
        id2label=train_dataset.id2label
    )
    test_datasets = [test_dataset] + separate_phonemebert_test_set(test_dataset)

    data_collator = DataCollatorWithPaddingMLM(
        tokenizer=tokenizer,
        mlm=args.input_mask_ratio > 0,
        mlm_probability=args.input_mask_ratio
    )

    all_preds = []
    for n in range(args.n):
        print('start training: {}'.format(n))
        model = Net(args, len(train_dataset.id2label))
        # Train model
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            num_train_epochs=args.max_epoch,
            per_device_train_batch_size=args.train_bsize,
            per_device_eval_batch_size=args.eval_bsize,
            weight_decay=0.01,               # strength of weight decay
            seed=args.seed + n,
        )
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        if args.use_pseudo:
            trainer.add_callback(UpdatePseudoLabelCallback(trainer))
        trainer.train()
        test_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer.data_collator = test_data_collator

        keys = ['test_accuracy']
        preds = []
        for i, test_dataset in enumerate(test_datasets):
            pred = trainer.predict(test_dataset=test_dataset)
            pred = {k: pred[2][k] for k in keys}
            preds.append(pred)
        all_preds.append(preds)

    predictions = {}
    for preds in all_preds:
        for i, pred in enumerate(preds):
            for k, v in pred.items():
                key = k+'-{}'.format(i)
                predictions[key] = predictions.get(key, []) + [np.round(v, 4)]

    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir, "{}.log".format(args.log_name))
    with open(logfile, 'w') as f:
        f.write("{:>30}\t{:>8}\t{:>8}\t{}\n".format('metric', 'mean', 'std', 'values'))
        print("\n{:>30}\t{:>8}\t{:>8}\t{}".format('metric', 'mean', 'std', 'values'))
        for k, v in predictions.items():
            mean = np.round(np.mean(v), 4)
            std = np.round(np.std(v), 4)
            print("{:>30}\t{:>8}\t{:>8}\t{}".format(k, mean, std, v))
            f.write("{:>30}\t{:>8}\t{:>8}\t{}\n".format(k, mean, std, v))
