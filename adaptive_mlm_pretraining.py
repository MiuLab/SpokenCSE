import json
import os
import torch.nn as nn
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from dataset import SlurpDataset, PhonemeBERTDataset
import argparse


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
        "--tokenizer", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--output_dir", default='runs/tapt', type=str, help="dir to save finetuned model"
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
        "--use_phoneme", action='store_true', help="use phoneme + text sequence"
    )
    parser.add_argument(
        "--use_phonemebert", action='store_true', help="use phonemebert dataset"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    steps = args.save_steps
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Dataset
    print('reading dataset')
    if not args.use_phonemebert:
        with open(args.meta, 'r') as f:
            meta = json.load(f)
        with open(args.dataset, 'r') as f:
            datasets = json.load(f)

        train_dataset = SlurpDataset(
            tokenizer, meta, datasets['train'], target=args.train_target,
            use_phoneme=args.use_phoneme, use_label=False)
        eval_dataset = SlurpDataset(
            tokenizer, meta, datasets['devel'], target=args.eval_target,
            use_phoneme=args.use_phoneme, use_label=False)
    else:
        train_dir = os.path.join(args.dataset, 'train')
        eval_dir = os.path.join(args.dataset, 'valid')
        test_dir = os.path.join(args.dataset, 'test')
        train_dataset = PhonemeBERTDataset(
            tokenizer, train_dir, use_golden='golden' in args.train_target,
            use_phoneme=args.use_phoneme, use_label=False, pretraining=True)
        eval_dataset = PhonemeBERTDataset(
            tokenizer, eval_dir, use_golden='golden' in args.eval_target,
            use_phoneme=args.use_phoneme, use_label=False, pretraining=True)

    # Model
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.config.type_vocab_size = 2
    model.roberta.embeddings.token_type_embeddings = nn.Embedding(2, model.config.hidden_size)
    model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    # Train model
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=steps,
        logging_strategy="steps",
        logging_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.eval_bsize,
        # weight_decay=0.01,               # strength of weight decay
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir.replace('runs', 'models'))
