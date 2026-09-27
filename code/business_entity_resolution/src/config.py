"""
Configuration module — paths, constants, and hyperparameters.

All paths are relative to the project root (amazon-ml-challenge-2026/).
"""

from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
# Resolve from this file: src/ → business_entity_resolution/ → code/ → root
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

# ─── Hyperparameters (tune these) ────────────────────────────────────────────
# Blocking
BLOCKING_TOP_K = 50               # Max candidates per S1 entity from each blocking key
TFIDF_NGRAM_RANGE = (2, 4)        # Character n-gram range for TF-IDF
TFIDF_TOP_K = 100                 # Top-K similar candidates from TF-IDF

# Matching threshold
MATCH_THRESHOLD = 0.5             # Probability threshold for final match decision

# Validation
VAL_SPLIT_RATIO = 0.2             # Fraction of training data for validation
RANDOM_SEED = 42
