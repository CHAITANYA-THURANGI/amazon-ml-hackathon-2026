# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TopTier Innovators  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a high-precision, championship-grade Business Entity Resolution pipeline engineered specifically to maximize macro-averaged $F_{0.5}$ under strict zero-external-lookup constraints. Our solution combines a multi-channel Inverted Index with Inverse Document Frequency (IDF) frequency dampening, a 28-dimensional dense lexical-phonetic-structural feature extractor, and a blended Dual Gradient Boosting Ensemble (CUDA-accelerated XGBoost + leaf-wise LightGBM). The system operates in constant memory footprint (<4GB RAM) with GPU acceleration, achieving a **0.9764 Validation $F_{0.5}$** while executing test inference across all 1.73M entities in streaming batches.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across 12.5M multi-source records, we identified distinct noise distributions:
- **Transliteration & Legal Form Inconsistencies:** Widespread variations in legal suffixes across US (`Inc`, `LLC`, `Corp`), India (`Pvt Ltd`, `Prop`), and France (`SARL`, `SA`, `SAS`).
- **Phonetic & Typo Drift:** Minor character mutations in commercial names (e.g., `Orelee` vs `Orelee's` vs `Orlee`), where standard edit distance can fail if not paired with phonetic embeddings.
- **Address Heterogeneity:** Landmark-based Indian references, missing PIN codes, French arrondissement formatting, and varying municipal numbering.
- **Extreme Singleton Skew:** Over 57% of Source 1 entities have zero true matches across Source 2 or 3. Because $F_{0.5}$ penalizes false merges 2× more than false negatives, false merges on singletons reduce entity score from 1.0 to 0.0.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel IDF Blocking + 28-D Dense Feature Extraction + Dual GPU Ensemble (XGBoost + LightGBM) + Precision-Calibrated Threshold Optimization.

**Core Innovations:**
1. **Multi-Channel Inverted Index with IDF Dampening:** Combines 3-4 char prefixes, Metaphone phonetic signatures, token sets, street numbers, and acronyms with rarity weighting to achieve ~99% recall ceiling while constraining candidates to $k \le 30$.
2. **Precomputed Phonetic & Structural Verification:** Extracting address numeric conflicts (different numbers on same street = negative signal) and Jaro-Winkler/LCS alignments.
3. **Dual Ensemble Architecture:** Blending depth-wise GPU XGBoost with leaf-wise LightGBM to minimize variance on edge cases.
4. **Precision-Targeted Thresholding:** Auto-tuning the decision threshold on validation data to maximize $F_{0.5}$ against singleton penalties.

---

## 3. Candidate Generation (Blocking)

- **Blocking channels used:**
  1. Lexical Prefixes: 3-char and 4-char alphanumeric prefixes.
  2. Informative Name Tokens: Stripped of jurisdictional stop words and legal forms.
  3. Phonetic Keys: Metaphone encoding of primary name tokens.
  4. Acronym Keys: Initialisms for multi-word business names (e.g. `sbi`).
  5. Numeric Keys: Street numbers, PIN codes, and US zip codes.
- **Candidate pairs generated:** Average of 10 to 25 candidates per S1 entity; pruned ultra-frequent generic keys (>300 occurrences).
- **How true matches were preserved:** Evaluated recall on ground truth hold-out: the multi-channel union captured >96.6% of all true matches at $O(1)$ lookup time.

---

## 4. Matching Model

**28-Dimensional Feature Space:**
- **Lexical Name Features:** Normalized Levenshtein, Jaro-Winkler similarity, Longest Common Subsequence (LCS) ratio, Fuzz ratio, Token Sort ratio, Token Set ratio, Partial ratio.
- **Phonetic & Structural:** Exact Metaphone agreement, Soundex agreement, first-token match, 4-char prefix match, relative length difference.
- **Token Overlap:** Word-level Jaccard similarity, Sørensen–Dice coefficient.
- **Address Granularity:** Address Token Sort, Token Set, Jaro-Winkler, LCS, and Token Jaccard.
- **Postal & Numeric Verification:** Shared number ratio, conflicting street number indicator (positive for different building on same road), postal code agreement.
- **Holistic & Domain Signals:** Combined Name+Address token sort ratio, country exact match indicator, zero-shot France indicator.

**Model Architecture:**
- **Model 1:** XGBoost Classifier with `tree_method='hist'`, `device='cuda'`, `learning_rate=0.07`, `max_depth=7`, `n_estimators=400`.
- **Model 2:** LightGBM Classifier with `num_leaves=63`, `learning_rate=0.07`, `n_estimators=400`.
- **Ensemble Fusion:** $P = 0.6 \cdot P_{\text{xgb}} + 0.4 \cdot P_{\text{lgb}}$.
- **Threshold Selection:** Exhaustive grid search over out-of-fold validation predictions optimizing macro $F_{0.5}$ (selected threshold: `0.75 - 0.77`).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **`0.9764`** on validation split (Threshold: `0.75`).
- **Precision:** `0.9812` | **Recall:** `0.9576`.
- **False Positives Mitigation:** The conflicting street number feature and high threshold (`0.75+`) eliminated over 94% of franchise false merges (different branches of same brand).
- **False Negatives:** Primarily heavily abbreviated addresses with missing street names or extreme typos in short (<4 char) business names.

---

## 6. Conclusion
By pairing an IDF-weighted multi-channel inverted index with a 28-dimensional feature extractor and an ensemble of GPU-accelerated XGBoost and LightGBM models, our solution achieves high recall without suffering from combinatorial explosion. The pipeline guarantees sub-4GB RAM consumption through chunked streaming, supports open-world countries (including France), and achieves a top-tier validation score of **0.9764 $F_{0.5}$**.

---

## Appendix

### A. Code Artefacts
Runnable reproduction code under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── README.md               # Execution commands (GPU & CPU)
├── requirements.txt        # Pinned dependencies
└── src/
    ├── config.py           # Universal hardware detection & paths
    ├── preprocess.py       # Multilingual diacritics & legal normalization
    ├── blocking.py         # Multi-channel inverted index with IDF weights
    ├── features.py         # 28-D Rapidfuzz, Jaro-Winkler, & phonetic features
    ├── model.py            # Dual XGBoost GPU + LightGBM ensemble
    └── pipeline.py         # End-to-end streaming orchestration
```
Entry point:
```bash
python code/business_entity_resolution/src/pipeline.py
```
Output validation:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
