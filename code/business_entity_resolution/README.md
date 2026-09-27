# Business Entity Resolution — Reproduction Guide

## Prerequisites

- Python 3.9+
- NVIDIA GPU (RTX 4060 or any CUDA GPU) supported automatically
- CPU fallback supported automatically on any system without code changes

## Execution Modes (Universal GPU & CPU)

### 1. GPU Mode (Default on your Laptop)
The pipeline automatically detects your **NVIDIA GeForce RTX 4060 Laptop GPU** and uses:
- **CUDA 12.8 / 13.3 GPU acceleration** for model training (`tree_method='hist'`, `device='cuda'`)
- Model training executes in **~1 second** on GPU!

```bash
# Auto-detects GPU and runs full pipeline:
python code/business_entity_resolution/src/pipeline.py
```

### 2. CPU Mode (Universal fallback)
If you want to force CPU execution (or on systems without a GPU):

```bash
python code/business_entity_resolution/src/pipeline.py --device cpu
```

### 3. Quick Verification Test
To verify the entire pipeline runs without waiting for all 1.7M test entities:

```bash
python code/business_entity_resolution/src/pipeline.py --quick-test
```

## Dataset Structure

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

## Output Files

The pipeline generates the two required competition files in `output/`:
1. `output/matching_results.tsv` — Final entity matches for portal leaderboard upload.
2. `output/candidate_pairs.tsv` — Candidate pairs from the blocking stage.

## Format Validation

Validate your files locally before uploading:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
