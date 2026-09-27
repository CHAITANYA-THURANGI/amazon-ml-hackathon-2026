"""
Hold-Out Validation Evaluation Script for Amazon ML Challenge 2026.

Evaluates the trained model on an independent hold-out validation set
(unseen during training) to compute:
- Macro F_0.5 (Competition Scored Metric)
- Precision (Macro & Pairwise)
- Recall (Macro & Pairwise)
- Pairwise Accuracy & Specificity
- Singleton Identification Accuracy
- Full Confusion Matrix (TP, FP, FN, TN)
- Error Analysis Breakdown
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from model import EntityMatchingEnsemble, compute_f05
from blocking import InvertedIndexBlocker
from features import extract_features_precomputed
from preprocess import clean_text, extract_postal_code
import jellyfish


def evaluate_model_on_holdout(n_entities: int = 2_000, max_cands: int = 350_000):
    print("=" * 68)
    print(f"  EVALUATING TRAINED ENSEMBLE ON UNSEEN HOLD-OUT DATA ({n_entities:,} S1)")
    print("=" * 68)

    # 1. Load Trained Ensemble Model
    model_path = Path("models/matching_ensemble.joblib")
    if not model_path.exists():
        model_path = Path("models/matching_model.joblib")

    if not model_path.exists():
        print(f"❌ Error: Model artifact not found at {model_path}. Train model first.")
        return

    print(f"  [1/4] Loading model: {model_path.name}...")
    ensemble = EntityMatchingEnsemble()
    ensemble.load(model_path)
    th = ensemble.threshold
    print(f"        Decision Threshold: {th:.4f} (optimized for Macro F_0.5)")

    # 2. Load Ground Truth into O(1) hash map
    print("  [2/4] Loading ground truth lookup dictionary...")
    t0 = time.time()
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
    gt_raw = dict(zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")))
    print(f"        Loaded {len(gt_raw):,} ground truth mappings in {time.time()-t0:.2f}s")

    # 3. Load unseen hold-out S1 entities (rows 60,000 to 60,000+N)
    start_row = 60_000
    print(f"  [3/4] Sampling {n_entities:,} unseen validation S1 entities (rows {start_row:,} to {start_row+n_entities:,})...")
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", skiprows=range(1, start_row + 1), nrows=n_entities, dtype=str)

    val_gt = {}
    target_match_ids = set()
    for sid in s1_df["entity_id"]:
        m_str = gt_raw.get(sid, "").strip()
        if m_str:
            matches = set(m.strip() for m in m_str.split(",") if m.strip())
            val_gt[sid] = matches
            target_match_ids.update(matches)
        else:
            val_gt[sid] = set()

    n_singletons_true = sum(1 for m in val_gt.values() if len(m) == 0)
    n_non_singletons_true = len(val_gt) - n_singletons_true
    print(f"        Validation S1 entities: {len(val_gt):,}")
    print(f"        True Singletons (0 matches): {n_singletons_true:,} ({n_singletons_true/len(val_gt)*100:.1f}%)")
    print(f"        True Entities with Matches:  {n_non_singletons_true:,} ({n_non_singletons_true/len(val_gt)*100:.1f}%)")
    print(f"        Total True Target Matches:   {len(target_match_ids):,}")

    # 4. Index candidate sources S2 and S3
    print(f"\n  [4/4] Building evaluation candidate index (budget: {max_cands:,} records)...")
    t0 = time.time()
    blocker = InvertedIndexBlocker()
    found_targets = set()

    for s_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(s_path, sep="\t", chunksize=250_000, dtype=str):
            for eid, nm, ad, ct in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
                if eid in target_match_ids or len(blocker.records) < max_cands:
                    blocker.add_record(eid, nm, ad, ct)
                    if eid in target_match_ids:
                        found_targets.add(eid)
            if len(blocker.records) >= max_cands and len(found_targets) >= len(target_match_ids) * 0.95:
                break

    pruned = blocker.prune_and_compute_weights()
    print(f"        Index built in {time.time()-t0:.1f}s | {len(blocker.records):,} records indexed.")
    print(f"        Blocking Reachable Targets: {len(found_targets):,}/{len(target_match_ids):,} ({len(found_targets)/max(len(target_match_ids),1)*100:.1f}%)")

    # 5. Extract features & Evaluate
    print("\n  Executing ensemble inference & calculating evaluation metrics...")
    t0 = time.time()

    pair_y_true = []
    pair_y_pred = []

    entity_f05_scores = []
    entity_precisions = []
    entity_recalls = []

    correct_singletons = 0
    correct_non_singletons = 0

    for eid, r_name, r_addr, r_ct in zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"]):
        s1_n = clean_text(r_name)
        s1_a = clean_text(r_addr)
        s1_c = (r_ct or "").strip().lower()
        s1_p = extract_postal_code(s1_a, s1_c)

        w1 = s1_n.split()
        s1_first = w1[0] if w1 else ""
        s1_meta = jellyfish.metaphone(s1_first) if s1_first else ""
        s1_soundex = jellyfish.soundex(s1_first) if s1_first else ""

        actual = val_gt.get(eid, set())
        cands = blocker.get_candidates(s1_n, s1_a, s1_c)

        predicted = set()
        if cands:
            feats = []
            cand_valid = []
            for cid in cands:
                c_data = blocker.records.get(cid)
                if not c_data:
                    continue
                c_n, c_a, c_c, c_p, c_first, c_meta, c_soundex = c_data
                feats.append(extract_features_precomputed(
                    s1_n, s1_a, s1_c, s1_p, s1_first, s1_meta, s1_soundex,
                    c_n, c_a, c_c, c_p, c_first, c_meta, c_soundex
                ))
                cand_valid.append(cid)

            if feats:
                probas = ensemble.predict_proba(np.array(feats, dtype=np.float32))
                for cid, p_val in zip(cand_valid, probas):
                    is_true = 1 if cid in actual else 0
                    is_pred = 1 if p_val >= th else 0
                    pair_y_true.append(is_true)
                    pair_y_pred.append(is_pred)
                    if is_pred == 1:
                        predicted.add(cid)

        # Macro F_0.5 computation per entity
        if len(actual) == 0:
            if len(predicted) == 0:
                correct_singletons += 1
                entity_f05_scores.append(1.0)
            else:
                entity_f05_scores.append(0.0)
        else:
            tp = len(predicted & actual)
            p = tp / len(predicted) if predicted else 0.0
            r_val = tp / len(actual)
            entity_precisions.append(p)
            entity_recalls.append(r_val)
            entity_f05_scores.append(compute_f05(p, r_val))
            if tp > 0:
                correct_non_singletons += 1

    eval_time = time.time() - t0

    # 6. Aggregate Pairwise & Macro Metrics
    y_true_arr = np.array(pair_y_true)
    y_pred_arr = np.array(pair_y_pred)

    tp_pairs = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp_pairs = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn_pairs = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn_pairs = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())

    total_pairs = len(y_true_arr)
    accuracy_pairs = (tp_pairs + tn_pairs) / max(total_pairs, 1)
    precision_pairs = tp_pairs / max(tp_pairs + fp_pairs, 1)
    recall_pairs = tp_pairs / max(tp_pairs + fn_pairs, 1)
    specificity_pairs = tn_pairs / max(tn_pairs + fp_pairs, 1)

    macro_f05 = np.mean(entity_f05_scores)
    macro_precision = np.mean(entity_precisions) if entity_precisions else 0.0
    macro_recall = np.mean(entity_recalls) if entity_recalls else 0.0
    singleton_accuracy = correct_singletons / max(n_singletons_true, 1)

    # 7. Print Official Evaluation Report
    print("\n" + "=" * 68)
    print("        OFFICIAL COMPETITION METRICS & VALIDATION REPORT         ")
    print("=" * 68)
    print(f"  ★ Macro F_0.5 Score (Official Metric): {macro_f05:.4f}")
    print(f"  ★ Macro Precision:                     {macro_precision:.4f}  ({macro_precision*100:.2f}%)")
    print(f"  ★ Macro Recall:                        {macro_recall:.4f}  ({macro_recall*100:.2f}%)")
    print(f"  ★ Singleton Identification Accuracy:   {singleton_accuracy:.4f}  ({singleton_accuracy*100:.2f}%)")
    print(f"  ★ Pairwise Classification Accuracy:    {accuracy_pairs:.4f}  ({accuracy_pairs*100:.2f}%)")
    print(f"  ★ Pairwise Specificity (TN Rate):      {specificity_pairs:.4f}  ({specificity_pairs*100:.2f}%)")
    print("=" * 68)

    print("\n  Confusion Matrix (Candidate Pairs Evaluated):")
    print("  " + "-" * 55)
    print(f"                         Predicted MATCH    Predicted NO-MATCH")
    print(f"    Actual MATCH       TP: {tp_pairs:7,d}        FN: {fn_pairs:7,d}")
    print(f"    Actual NO-MATCH    FP: {fp_pairs:7,d}        TN: {tn_pairs:7,d}")
    print("  " + "-" * 55)
    print(f"    Total Pairs Evaluated:              {total_pairs:,}")
    print(f"    Pairwise Precision:                 {precision_pairs:.4%}")
    print(f"    Pairwise Recall:                    {recall_pairs:.4%}")
    print(f"    False Positive Rate (False Merges): {fp_pairs / max(fp_pairs + tn_pairs, 1):.4%}")
    print(f"    False Discovery Rate:               {fp_pairs / max(tp_pairs + fp_pairs, 1):.4%}")

    print("\n  Entity-Level Breakdown:")
    print(f"    - Evaluated Validation S1 Entities: {len(s1_df):,}")
    print(f"    - True Singletons Correctly Found:  {correct_singletons:,} / {n_singletons_true:,} ({singleton_accuracy*100:.1f}%)")
    print(f"    - True Entities with Matches Found: {correct_non_singletons:,} / {n_non_singletons_true:,} ({correct_non_singletons/max(n_non_singletons_true,1)*100:.1f}%)")
    print(f"    - Evaluation Speed:                 {len(s1_df) / max(eval_time, 0.01):.1f} entities/sec")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    evaluate_model_on_holdout()
