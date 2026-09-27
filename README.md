# Amazon ML Challenge 2026 — Business Entity Resolution

## 🎯 Challenge Overview

**Entity Resolution (ER)** across 3 independent data sources with noisy, inconsistent business records. Given business records from Source 1 (reference), Source 2, and Source 3, determine which records refer to the same real-world business entity.

- **Metric:** F₀.₅ (precision-heavy, macro-averaged per Source 1 entity)
- **Model constraint:** MIT/Apache 2.0 license, ≤ 8B parameters
- **No external data lookup** — only the provided training data

## 📁 Project Structure

```
amazon-ml-challenge-2026/
│
├── README.md                          # ← You are here
├── Documentation_template.md          # Methodology write-up template
├── inspect_data.py                    # EDA / data inspection script
├── .gitignore
│
├── dataset/                           # All data files (gitignored — too large)
│   ├── train/
│   │   ├── train_source1.tsv          # Source 1 training records (reference)
│   │   ├── train_source2.tsv          # Source 2 training records
│   │   ├── train_source3.tsv          # Source 3 training records
│   │   └── train_ground_truth.tsv     # Ground truth matching labels
│   └── test/
│       ├── test_source1.tsv           # Source 1 test records (predict these)
│       ├── test_source2.tsv           # Source 2 test records
│       └── test_source3.tsv           # Source 3 test records
│
├── code/                              # Your solution pipeline
│   └── business_entity_resolution/
│       ├── README.md                  # Reproduction instructions
│       ├── requirements.txt           # Pinned dependencies
│       └── src/                       # All source code
│           ├── __init__.py
│           ├── config.py              # Configuration & paths
│           ├── preprocess.py          # Data cleaning & normalization
│           ├── blocking.py            # Candidate generation / blocking
│           ├── features.py            # Feature engineering
│           ├── model.py               # Matching model (train & predict)
│           ├── evaluate.py            # F₀.₅ evaluation & validation split
│           └── pipeline.py            # End-to-end pipeline orchestration
│
├── output/                            # Generated output files
│   ├── matching_results.tsv           # Final matches (leaderboard upload)
│   └── candidate_pairs.tsv            # Blocking candidate set
│
├── utils/                             # Provided utility scripts
│   └── validate_submission.py         # Submission format validator
│
└── docs/                              # Reference documents
    ├── amazon-ml-challenge.pdf        # Full challenge description
    └── dataset-details.pdf            # Dataset documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Explore the Data

```bash
python inspect_data.py
```

### 3. Run the Pipeline

```bash
python code/business_entity_resolution/src/pipeline.py
```

### 4. Validate Output

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## 📊 Data Summary

| File | Records | Description |
|------|---------|-------------|
| `train_source1.tsv` | ~200 MB | Reference source (deduplicated) |
| `train_source2.tsv` | ~467 MB | Noisy duplicate source |
| `train_source3.tsv` | ~480 MB | Noisy duplicate source |
| `train_ground_truth.tsv` | ~121 MB | S1 → {S2, S3} matching labels |
| `test_source1.tsv` | ~167 MB | Test reference (predict these) |
| `test_source2.tsv` | ~486 MB | Test noisy source |
| `test_source3.tsv` | ~483 MB | Test noisy source |

**Columns:** `entity_id`, `business_name`, `business_address`, `country`

**Countries:** Training = {US, India}; Test adds **France** (zero-shot)

## 📐 Evaluation

**F₀.₅ Score** (precision-heavy, macro-averaged):

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- Singletons (no matches) score 1.0 when correctly predicted empty
- False merges are penalized more than missed matches

## 📦 Final Submission

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

## 💡 Tips

- Strong blocking strategy → upper-bound recall
- String similarity: Jaccard, Levenshtein, TF-IDF cosine
- Country-specific address patterns
- F₀.₅ rewards precision > recall — be conservative
- Don't neglect singletons — correct "no match" = 1.0 per entity
