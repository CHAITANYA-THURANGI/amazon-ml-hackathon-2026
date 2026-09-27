"""
High-Performance Feature Engineering Module for Entity Resolution.

Extracts 28 dense signals with precomputed phonetics and fast C++ Rapidfuzz metrics:
- Edit distance & prefix alignments
- Token intersection, Dice, and Jaccard
- Phonetic exact matches (O(1) comparison from precomputed codes)
- Address structural verification & number conflicts
- Holistic composite signals
"""

from typing import List, Set, Tuple
import re
from rapidfuzz import fuzz, distance

_RE_NUM = re.compile(r"\b\d{2,6}\b")


def extract_features_precomputed(
    s1_n: str, s1_a: str, s1_c: str, s1_p: str, s1_first: str, s1_meta: str, s1_soundex: str,
    c_n: str, c_a: str, c_c: str, c_p: str, c_first: str, c_meta: str, c_soundex: str,
) -> List[float]:
    """Compute 28-dimensional dense feature vector using precomputed phonetic & token metadata."""
    # ── 1. Name Features ──
    n_ratio = fuzz.ratio(s1_n, c_n) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
    n_set = fuzz.token_set_ratio(s1_n, c_n) / 100.0
    n_part = fuzz.partial_ratio(s1_n, c_n) / 100.0
    n_lev = distance.Levenshtein.normalized_similarity(s1_n, c_n)
    n_jw = distance.JaroWinkler.similarity(s1_n, c_n)
    n_lcs = distance.LCSseq.normalized_similarity(s1_n, c_n)

    w1 = s1_n.split()
    w2 = c_n.split()
    set1 = set(w1)
    set2 = set(w2)
    inter = len(set1 & set2)
    n_jacc = inter / max(len(set1 | set2), 1)
    n_dice = (2.0 * inter) / max(len(set1) + len(set2), 1)

    s1_pref = "".join(ch for ch in s1_n if ch.isalnum())[:4]
    c_pref = "".join(ch for ch in c_n if ch.isalnum())[:4]
    n_pref = 1.0 if (s1_pref and s1_pref == c_pref) else 0.0

    first_match = 1.0 if (s1_first and s1_first == c_first) else 0.0
    meta_match = 1.0 if (s1_meta and s1_meta == c_meta) else 0.0
    soundex_match = 1.0 if (s1_soundex and s1_soundex == c_soundex) else 0.0

    n_len_diff = abs(len(s1_n) - len(c_n)) / max(len(s1_n) + len(c_n), 1)

    # ── 2. Address Features ──
    a_ratio = fuzz.ratio(s1_a, c_a) / 100.0
    a_sort = fuzz.token_sort_ratio(s1_a, c_a) / 100.0
    a_set = fuzz.token_set_ratio(s1_a, c_a) / 100.0

    aw1 = set(s1_a.split())
    aw2 = set(c_a.split())
    a_jacc = len(aw1 & aw2) / max(len(aw1 | aw2), 1)
    a_jw = distance.JaroWinkler.similarity(s1_a, c_a)
    a_lcs = distance.LCSseq.normalized_similarity(s1_a, c_a)
    a_len_diff = abs(len(s1_a) - len(c_a)) / max(len(s1_a) + len(c_a), 1)

    nums1 = set(_RE_NUM.findall(s1_a))
    nums2 = set(_RE_NUM.findall(c_a))
    if nums1 and nums2:
        num_inter = len(nums1 & nums2)
        num_overlap = num_inter / len(nums1 | nums2)
        num_conflict = 1.0 if num_inter == 0 else 0.0
    elif not nums1 and not nums2:
        num_overlap = 0.5
        num_conflict = 0.0
    else:
        num_overlap = 0.25
        num_conflict = 0.0

    if s1_p and c_p:
        postal_match = 1.0 if s1_p == c_p else 0.0
    else:
        postal_match = 0.5

    # ── 3. Holistic & Metadata Features ──
    comb1 = f"{s1_n} {s1_a}"
    comb2 = f"{c_n} {c_a}"
    comb_sort = fuzz.token_sort_ratio(comb1, comb2) / 100.0

    cw1 = set(comb1.split())
    cw2 = set(comb2.split())
    comb_jacc = len(cw1 & cw2) / max(len(cw1 | cw2), 1)

    c_match = 1.0 if (s1_c and s1_c == c_c) else 0.0
    is_fr = 1.0 if (s1_c == "france" or c_c == "france") else 0.0
    prior = (n_jw * 0.4) + (a_jw * 0.3) + (comb_sort * 0.3)

    return [
        n_ratio,
        n_sort,
        n_set,
        n_part,
        n_lev,
        n_jw,
        n_lcs,
        n_jacc,
        n_dice,
        n_pref,
        first_match,
        meta_match,
        soundex_match,
        n_len_diff,
        a_ratio,
        a_sort,
        a_set,
        a_jacc,
        a_jw,
        a_lcs,
        a_len_diff,
        num_overlap,
        num_conflict,
        postal_match,
        comb_sort,
        comb_jacc,
        c_match,
        is_fr,
        prior,
    ]
