"""
Configuration module — paths, constants, device detection, and hyperparameters.

Supports:
- Universal execution: NVIDIA GPU (CUDA) or CPU fallback
- Memory-safe streaming and batch sizes
- All competition paths and evaluation constants
"""

from pathlib import Path
import os
import argparse

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ─── Dataset Paths ───────────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_DIR = DATASET_DIR / "test"
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# ─── Output Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# ─── Model Paths ─────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"

# ─── Data Constants ──────────────────────────────────────────────────────────
TSV_SEP = "\t"
ENTITY_ID_COL = "entity_id"
BUSINESS_NAME_COL = "business_name"
BUSINESS_ADDRESS_COL = "business_address"
COUNTRY_COL = "country"
SOURCE1_ENTITY_ID_COL = "source1_entity_id"
MATCHED_ENTITY_IDS_COL = "matched_entity_ids"
CANDIDATE_ENTITY_IDS_COL = "candidate_entity_ids"

# ─── Device Detection (Universal GPU + CPU Fallback) ─────────────────────────

def detect_device(force_device: str = None) -> dict:
    """
    Detect the compute device (GPU or CPU).

    Args:
        force_device: 'gpu', 'cuda', or 'cpu' to override auto-detection.

    Returns:
        dict with device settings.
    """
    if force_device:
        force_device = force_device.lower()
        if force_device in ("cpu",):
            return {
                "use_gpu": False,
                "xgb_device": "cpu",
                "torch_device": "cpu",
                "device_name": "CPU (Forced)",
            }

    # Auto-detect CUDA GPU
    use_gpu = False
    device_name = "CPU"
    xgb_device = "cpu"
    torch_device = "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            device_name = torch.cuda.get_device_name(0)
            torch_device = "cuda"
            xgb_device = "cuda"
    except ImportError:
        pass

    if not use_gpu:
        try:
            import xgboost as xgb
            if xgb.build_info().get("USE_CUDA", False):
                use_gpu = True
                xgb_device = "cuda"
                device_name = "CUDA (via XGBoost)"
        except Exception:
            pass

    return {
        "use_gpu": use_gpu,
        "xgb_device": xgb_device,
        "torch_device": torch_device,
        "device_name": device_name,
    }


DEVICE_CONFIG = detect_device()
USE_GPU = DEVICE_CONFIG["use_gpu"]
XGB_DEVICE = DEVICE_CONFIG["xgb_device"]
TORCH_DEVICE = DEVICE_CONFIG["torch_device"]

# ─── Pipeline Hyperparameters ────────────────────────────────────────────────
# Scale parameters for memory-safe execution on laptop
TRAIN_SAMPLE_SIZE = 60_000        # Number of S1 training records to sample for model training
VAL_SAMPLE_SIZE = 10_000          # Number of S1 validation records for threshold tuning
TEST_CHUNK_SIZE = 100_000         # Process test set in streaming chunks of S1 entities

# Blocking parameters
MAX_CANDIDATES_PER_S1 = 35        # Top candidate pairs per S1 record

# Matching threshold (F0.5 favors precision: default 0.55, auto-tuned during training)
DEFAULT_THRESHOLD = 0.55

# Validation split
VAL_SPLIT_RATIO = 0.2

# CPU parallelism
N_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
RANDOM_SEED = 42
