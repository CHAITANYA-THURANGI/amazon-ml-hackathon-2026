"""
End-to-end Entity Resolution Pipeline.

Supports:
- Universal hardware execution: NVIDIA GPU (CUDA) or CPU fallback
- Ultra-fast Inverted Index Blocking with column zip iteration (>50x faster than iterrows)
- Fast feature extraction and GPU-accelerated XGBoost
- Macro F_0.5 threshold optimization
- Streaming memory-safe test inference
- Automatic submission format validation
"""

import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure unbuffered output so logs print immediately in terminal
sys.stdout.reconfigure(line_buffering=True)

# Ensure src/ is in path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    MATCHING_RESULTS, CANDIDATE_PAIRS, OUTPUT_DIR,
    TRAIN_SAMPLE_SIZE, TEST_CHUNK_SIZE,
    detect_device,
)
from preprocess import clean_text
from blocking import InvertedIndexBlocker
from features import extract_features
from model import EntityMatchingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Entity Resolution Pipeline")
    parser.add_argument(
        "--device",
        choices=["gpu", "cuda", "cpu"],
        default=None,
        help="Compute device (default: auto-detect GPU/CPU)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=TRAIN_SAMPLE_SIZE,
        help=f"Number of S1 train samples for model training (default: {TRAIN_SAMPLE_SIZE:,})",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick verification pass with small sample size",
    )
    return parser.parse_args()


def build_candidate_index(s2_path: Path, s3_path: Path, max_records: int = None) -> InvertedIndexBlocker:
    """Build inverted index from candidate sources S2 and S3 using fast zip iteration."""
    blocker = InvertedIndexBlocker()
    total_loaded = 0

    print("  Indexing Source 2 records...", flush=True)
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=250_000, dtype=str):
        # Fast column zip iteration
        e_ids = chunk["entity_id"].fillna("").values
        names = chunk["business_name"].fillna("").values
        addrs = chunk["business_address"].fillna("").values
        cntrs = chunk["country"].fillna("").values

        for eid, nm, ad, ct in zip(e_ids, names, addrs, cntrs):
            blocker.add_record(eid, nm, ad, ct)
            total_loaded += 1
            if max_records and total_loaded >= max_records:
                break
        if max_records and total_loaded >= max_records:
            break

    print("  Indexing Source 3 records...", flush=True)
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=250_000, dtype=str):
        e_ids = chunk["entity_id"].fillna("").values
        names = chunk["business_name"].fillna("").values
        addrs = chunk["business_address"].fillna("").values
        cntrs = chunk["country"].fillna("").values

        for eid, nm, ad, ct in zip(e_ids, names, addrs, cntrs):
            blocker.add_record(eid, nm, ad, ct)
            total_loaded += 1
            if max_records and total_loaded >= (max_records * 2):
                break
        if max_records and total_loaded >= (max_records * 2):
            break

    pruned = blocker.prune_frequent_keys()
    print(f"  Indexed {len(blocker.records):,} total candidate records. Pruned {pruned:,} high-frequency keys.", flush=True)
    return blocker


def prepare_training_pairs(
    s1_path: Path,
    gt_path: Path,
    blocker: InvertedIndexBlocker,
    n_samples: int,
):
    """Generate labeled feature matrix from training S1 samples and ground truth."""
    print(f"  Loading ground truth and sampling {n_samples:,} S1 entities...", flush=True)
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_map = {}
    for sid, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        m = str(mids).strip()
        gt_map[sid] = set(m.split(",")) if (m and m != "nan") else set()

    s1_df = pd.read_csv(s1_path, sep="\t", nrows=n_samples, dtype=str)
    e_ids = s1_df["entity_id"].fillna("").values
    names = s1_df["business_name"].fillna("").values
    addrs = s1_df["business_address"].fillna("").values
    cntrs = s1_df["country"].fillna("").values

    X_list = []
    y_list = []

    print("  Extracting features for training candidate pairs...", flush=True)
    for s1_id, raw_n, raw_a, raw_c in tqdm(zip(e_ids, names, addrs, cntrs), total=len(s1_df), desc="  Pairs"):
        s1_n = clean_text(raw_n)
        s1_a = clean_text(raw_a)
        s1_c = raw_c.strip().lower()

        true_matches = gt_map.get(s1_id, set())

        # Retrieve blocking candidates
        cands = set(blocker.get_candidates(s1_n, s1_a, s1_c))

        # Always include true matches in training data for positive examples
        all_eval_cands = cands | (true_matches & blocker.records.keys())

        for cid in all_eval_cands:
            c_data = blocker.records.get(cid)
            if not c_data:
                continue
            c_n, c_a, c_c = c_data
            feat = extract_features(s1_n, s1_a, s1_c, c_n, c_a, c_c)
            X_list.append(feat)
            y_list.append(1 if cid in true_matches else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def run_pipeline():
    start_time = time.time()
    args = parse_args()

    # Device configuration
    dev_config = detect_device(args.device)
    device_name = dev_config["device_name"]
    use_gpu = dev_config["use_gpu"]

    print("=" * 65, flush=True)
    print("AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print(f"  Target Device:  {device_name} (GPU={use_gpu})", flush=True)
    print(f"  XGBoost Device: {dev_config['xgb_device'].upper()}", flush=True)
    print("=" * 65, flush=True)

    n_train_samples = 5_000 if args.quick_test else args.train_samples
    max_idx_records = 50_000 if args.quick_test else None

    # ── Stage 1: Build Candidate Index (S2 & S3) ─────────────────────────────
    print("\n[1/5] Building Candidate Blocking Index...", flush=True)
    t0 = time.time()
    blocker = build_candidate_index(TRAIN_SOURCE2, TRAIN_SOURCE3, max_records=max_idx_records)
    print(f"  ⏱ Index built in {time.time() - t0:.1f}s", flush=True)

    # ── Stage 2: Prepare Training Data & Split ────────────────────────────────
    print("\n[2/5] Generating Features & Labels for Model Training...", flush=True)
    t0 = time.time()
    X, y = prepare_training_pairs(TRAIN_SOURCE1, TRAIN_GROUND_TRUTH, blocker, n_train_samples)
    print(f"  Feature matrix shape: {X.shape}", flush=True)
    print(f"  Positive pairs: {y.sum():,} ({y.mean():.2%}) | Negative pairs: {len(y)-y.sum():,}", flush=True)

    # Train / Validation split
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    print(f"  ⏱ Feature preparation done in {time.time() - t0:.1f}s", flush=True)

    # ── Stage 3: Train Model & Optimize Threshold ────────────────────────────
    print("\n[3/5] Training Matching Model...", flush=True)
    t0 = time.time()
    model = EntityMatchingModel(device=dev_config["xgb_device"])
    model.train(X_train, y_train)
    model.tune_threshold(X_val, y_val)
    model.save()
    print(f"  ⏱ Model trained in {time.time() - t0:.1f}s", flush=True)

    # Free training memory before test phase
    del blocker, X, y, X_train, y_train, X_val, y_val
    import gc
    gc.collect()

    # ── Stage 4: Test Blocking & Inference (Streaming) ───────────────────────
    print("\n[4/5] Building Test Candidate Index & Running Inference...", flush=True)
    t0 = time.time()
    test_blocker = build_candidate_index(TEST_SOURCE2, TEST_SOURCE3, max_records=max_idx_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  Streaming test predictions to output files...", flush=True)
    # Open both output files for streaming
    with open(MATCHING_RESULTS, "w", encoding="utf-8") as f_match, \
         open(CANDIDATE_PAIRS, "w", encoding="utf-8") as f_cand:

        # Write required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        test_s1_reader = pd.read_csv(TEST_SOURCE1, sep="\t", chunksize=TEST_CHUNK_SIZE, dtype=str)
        total_s1 = 0
        total_matched = 0

        for chunk_idx, chunk in enumerate(test_s1_reader):
            if args.quick_test and chunk_idx >= 1:
                break

            print(f"  Processing Test Chunk {chunk_idx + 1} ({len(chunk):,} entities)...", flush=True)

            e_ids = chunk["entity_id"].fillna("").values
            names = chunk["business_name"].fillna("").values
            addrs = chunk["business_address"].fillna("").values
            cntrs = chunk["country"].fillna("").values

            for s1_id, raw_n, raw_a, raw_c in tqdm(zip(e_ids, names, addrs, cntrs), total=len(chunk), desc=f"  Chunk {chunk_idx+1}"):
                s1_n = clean_text(raw_n)
                s1_a = clean_text(raw_a)
                s1_c = raw_c.strip().lower()

                cands = test_blocker.get_candidates(s1_n, s1_a, s1_c)

                if not cands:
                    f_cand.write(f"{s1_id}\t\n")
                    f_match.write(f"{s1_id}\t\n")
                    total_s1 += 1
                    continue

                # Write candidate pairs
                f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

                # Extract features for candidate pairs
                pair_feats = []
                valid_cands = []
                for cid in cands:
                    c_data = test_blocker.records.get(cid)
                    if not c_data:
                        continue
                    c_n, c_a, c_c = c_data
                    pair_feats.append(extract_features(s1_n, s1_a, s1_c, c_n, c_a, c_c))
                    valid_cands.append(cid)

                if pair_feats:
                    X_test = np.array(pair_feats, dtype=np.float32)
                    preds = model.predict(X_test)
                    matched = [valid_cands[i] for i, p in enumerate(preds) if p == 1]
                    f_match.write(f"{s1_id}\t{','.join(matched)}\n")
                    if matched:
                        total_matched += 1
                else:
                    f_match.write(f"{s1_id}\t\n")

                total_s1 += 1

    print(f"  Generated outputs for {total_s1:,} entities ({total_matched:,} with matches).", flush=True)
    print(f"  ⏱ Test inference complete in {time.time() - t0:.1f}s", flush=True)

    # ── Stage 5: Validate Output ─────────────────────────────────────────────
    print("\n[5/5] Running Official Submission Validator...", flush=True)
    import subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(MATCHING_RESULTS),
        "--candidate", str(CANDIDATE_PAIRS),
        "--test-dir", str(TEST_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Warnings/Errors:", res.stderr)

    total_elapsed = time.time() - start_time
    print("=" * 65, flush=True)
    print(f"PIPELINE COMPLETED in {total_elapsed / 60:.1f} minutes", flush=True)
    print(f"Output files saved at: {OUTPUT_DIR}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_pipeline()
