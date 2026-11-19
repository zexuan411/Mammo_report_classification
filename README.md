# Explainable Automated Classification of Portuguese Breast Mammography Reports Using Domain-Specific Pre-Trained Language Models:Evaluation and Analysis

BERT-based classification of mammography reports into BI-RADS labels (`0`–`6`).

## Main scripts

| Script | Description |
|--------|-------------|
| `bertori.py` | Full-report BERT baseline (CE / ordinal / focal / hierarchical) |
| `bert_segment_crossattention.py` | Segmented reports (clinical / findings / comparative) with cross-attention fusion |


## Quick start

```bash
# Full-report baseline
torchrun --nproc_per_node=2 bertori.py \
  --use_ddp --use_amp \
  --loss_type hierarchical --mix_lambda 0.5 --cascade_tau 0.5 \
  --ckpt_dir results/bertori_hier05_hard_zero0

# Segmented cross-attention
torchrun --nproc_per_node=2 bert_segment_crossattention.py \
  --use_ddp --use_amp \
  --loss_type hierarchical --mix_lambda 0.5 --cascade_tau 0.5 \
  --ckpt_dir results/segmented_crossattn_811_hier05_hard_zero0
```

## Data

Place CSVs under `dataset/` (or pass `--train_path` / `--test_path`):

- `train_new.csv` / `train_new_en.csv` — full reports  
- `train_split.csv` — pre-segmented sections for the cross-attention model  

## Results

Fold and OOF summaries live under `results/`

