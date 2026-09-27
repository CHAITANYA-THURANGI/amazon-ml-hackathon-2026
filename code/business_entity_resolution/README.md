# Business Entity Resolution — Reproduction Guide

## Prerequisites

- Python 3.9+
- ~8 GB RAM recommended (dataset is large)

## Setup

```bash
# From the project root
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r code/business_entity_resolution/requirements.txt
```

## Dataset

Ensure the dataset is placed at `dataset/` relative to the project root:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run End-to-End Pipeline

```bash
# From the project root
python code/business_entity_resolution/src/pipeline.py
```

This will:
1. Load and preprocess all source data
2. Run blocking / candidate generation → `output/candidate_pairs.tsv`
3. Extract features for candidate pairs
4. Train/load matching model and predict → `output/matching_results.tsv`

## Run Individual Steps

```bash
# Data inspection
python inspect_data.py

# Validation
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Project Structure

```
src/
├── __init__.py       # Package init
├── config.py         # Paths, constants, hyperparameters
├── preprocess.py     # Data cleaning & normalization
├── blocking.py       # Candidate generation / blocking strategies
├── features.py       # Feature engineering for candidate pairs
├── model.py          # Matching model training & inference
├── evaluate.py       # F₀.₅ evaluation on validation split
└── pipeline.py       # End-to-end orchestration
```
