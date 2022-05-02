# Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding
Source code for Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding.
Paper link: [arxiv link]

## Requirements
```
pip install -r requirements.txt
```
NOTE: 
- you may need additional setting for `phonemizer` if you want to do phoneme-related data preprocessing

## Usage

0. Data Preparation
For SLURP,
The preprocessed dataset is `datasets/slurp/slurp_with_oracle_test.json`.
The preprocessed dataset without filtering and separating test sets is `datasets/slurp/slurp.json`
The data preprocessing includes multiple operations including:
    - Derive ASR hypothesis
    - Generate phoneme sequences by `phonemizer`
    - Preprocess the dataset (1st version)
        - Scripts in `prepare_data` would you understand the process:
        - first run `make_golden_dataset` read only from data provided in SLURP repo
        - and then `make_dataset` would need transcriptions from different systems
    - Fine-tune `roberta-base` models on the 1st version dataset
    - Collect predictions and sub-sample the dataset with agreed pseudo label


For ATIS/TREC6 from [PhonemeBERT](https://github.com/Observeai-Research/Phoneme-BERT),
You can just clone their repo and unzip the dataset.

1. Contrastive Pretraining
```
python contrastive_pretraining.py
```

2. Fine-tuning
```
python finetune_on_slurp.py
```
or on the phonemebert datasets:
```
python finetune_on_phonemebert.py
```

Training and evaluation are both included in these two scripts.
Adjust the arguments as you need.

## Reference
Please cite the following paper:
```
citation
```
