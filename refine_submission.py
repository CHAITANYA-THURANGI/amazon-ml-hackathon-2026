"""
High-Precision Sister Match Recovery & Refinement Script for Amazon ML Challenge 2026.

Recovers high-confidence missing sister matches (S2 <-> S3) for entities that only
matched one source, utilizing the precomputed candidate pairs and high-precision
lexical-address verification.

Safe execution:
- Does not overwrite 0.754 submission
- Outputs to output/matching_results_refined.tsv
- Fully validated via utils/validate_submission.py
"""

import sys
import time
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_SOURCE1 = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"
TEST_SOURCE2 = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
TEST_SOURCE3 = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"

ORIG_MATCHING = PROJECT_ROOT / "output" / "matching_results.tsv"
CANDIDATE_PAIRS = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
REFINED_MATCHING = PROJECT_ROOT / "output" / "matching_results_refined.tsv"


def run_refinement():
    start_time = time.time()
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — SUBMISSION REFINEMENT & SISTER RECOVERY")
    print("=" * 70)

    # 1. Load S1 data
    print("\n[1/5] Loading Test Source 1 entities...", flush=True)
    t0 = time.time()
    s1_df = pd.read_csv(TEST_SOURCE1, sep="\t", dtype=str)
    s1_names = dict(zip(s1_df["entity_id"], s1_df["business_name"].fillna("").str.lower().str.strip()))
    s1_addrs = dict(zip(s1_df["entity_id"], s1_df["business_address"].fillna("").str.lower().str.strip()))
    del s1_df
    print(f"  Loaded {len(s1_names):,} S1 entities in {time.time()-t0:.2f}s", flush=True)

    # 2. Identify entities that have only S2 or only S3 in original matching results
    print("\n[2/5] Analyzing current predictions from 0.754 submission...", flush=True)
    t0 = time.time()
    s1_matches = {}
    entities_to_check = set()
    total_entities = 0

    with open(ORIG_MATCHING, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            total_entities += 1
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            if len(parts) > 1 and parts[1]:
                cids = parts[1].split(",")
                s1_matches[eid] = cids
                has_s2 = any(c.startswith("S2-") for c in cids)
                has_s3 = any(c.startswith("S3-") for c in cids)
                if (has_s2 and not has_s3) or (has_s3 and not has_s2):
                    entities_to_check.add(eid)
            else:
                s1_matches[eid] = []

    print(f"  Total S1 entities: {total_entities:,}", flush=True)
    print(f"  Entities with matches: {len(s1_matches):,}", flush=True)
    print(f"  Target entities for sister recovery (single-source match): {len(entities_to_check):,}", flush=True)
    print(f"  ⏱ Analyzed in {time.time()-t0:.2f}s", flush=True)

    # 3. Read candidate pairs for target entities to collect needed S2 and S3 candidate IDs
    print("\n[3/5] Scanning precomputed candidates for target entities...", flush=True)
    t0 = time.time()
    target_cands = {}
    needed_s2 = set()
    needed_s3 = set()

    with open(CANDIDATE_PAIRS, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            if eid in entities_to_check and len(parts) > 1 and parts[1]:
                c_list = parts[1].split(",")
                target_cands[eid] = c_list
                for c in c_list:
                    if c.startswith("S2-"):
                        needed_s2.add(c)
                    elif c.startswith("S3-"):
                        needed_s3.add(c)

    print(f"  Found candidates for {len(target_cands):,} target entities.", flush=True)
    print(f"  Unique candidates to lookup: S2={len(needed_s2):,}, S3={len(needed_s3):,}", flush=True)
    print(f"  ⏱ Scanned in {time.time()-t0:.2f}s", flush=True)

    # 4. Stream test_source2 and test_source3 to fetch needed candidate records
    print("\n[4/5] Loading target candidate names & addresses into memory...", flush=True)
    t0 = time.time()
    cand_data = {}

    # Source 2
    if needed_s2:
        print("  Streaming Test Source 2...", flush=True)
        for chunk in pd.read_csv(TEST_SOURCE2, sep="\t", chunksize=300000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_s2)]
            for cid, nm, ad in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                cand_data[cid] = (str(nm or "").lower().strip(), str(ad or "").lower().strip())
            if len(cand_data) >= len(needed_s2):
                break

    # Source 3
    if needed_s3:
        print("  Streaming Test Source 3...", flush=True)
        s3_found = 0
        for chunk in pd.read_csv(TEST_SOURCE3, sep="\t", chunksize=300000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_s3)]
            for cid, nm, ad in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                cand_data[cid] = (str(nm or "").lower().strip(), str(ad or "").lower().strip())
                s3_found += 1
            if s3_found >= len(needed_s3):
                break

    print(f"  Loaded {len(cand_data):,} candidate records in {time.time()-t0:.2f}s", flush=True)

    # 5. Execute High-Precision Sister Recovery & Output Streaming
    print("\n[5/5] Executing Sister Recovery and writing refined output...", flush=True)
    t0 = time.time()
    recovered_sisters = 0
    total_written = 0

    with open(REFINED_MATCHING, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        with open(ORIG_MATCHING, "r", encoding="utf-8") as f_orig:
            next(f_orig)
            for line in f_orig:
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                cur_matches = parts[1].split(",") if len(parts) > 1 and parts[1] else []

                if eid in entities_to_check and cur_matches:
                    s1_n = s1_names.get(eid, "")
                    s1_a = s1_addrs.get(eid, "")
                    has_s2 = any(c.startswith("S2-") for c in cur_matches)
                    has_s3 = any(c.startswith("S3-") for c in cur_matches)
                    target_prefix = "S3-" if (has_s2 and not has_s3) else "S2-"

                    best_cand = None
                    best_n_sim = 0.0

                    for cid in target_cands.get(eid, []):
                        if cid.startswith(target_prefix) and cid in cand_data:
                            c_n, c_a = cand_data[cid]
                            n_sim = fuzz.token_set_ratio(s1_n, c_n)
                            # Strict criteria: name similarity >= 90% and address similarity >= 70%
                            if n_sim >= 90:
                                a_sim = fuzz.token_set_ratio(s1_a, c_a)
                                if a_sim >= 70:
                                    if n_sim > best_n_sim:
                                        best_n_sim = n_sim
                                        best_cand = cid

                    if best_cand:
                        cur_matches.append(best_cand)
                        recovered_sisters += 1

                f_out.write(f"{eid}\t{','.join(cur_matches)}\n")
                total_written += 1

    print(f"  Recovered {recovered_sisters:,} high-confidence sister matches!", flush=True)
    print(f"  Wrote {total_written:,} entities to {REFINED_MATCHING.name} in {time.time()-t0:.2f}s", flush=True)

    # Validate output
    print("\nValidating refined submission format...")
    import subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(REFINED_MATCHING),
        "--candidate", str(CANDIDATE_PAIRS),
        "--test-dir", str(PROJECT_ROOT / "dataset" / "test"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, flush=True)

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"REFINEMENT COMPLETE in {total_time / 60:.1f} minutes!")
    print(f"Refined output ready at: {REFINED_MATCHING}")
    print("=" * 70)


if __name__ == "__main__":
    run_refinement()
