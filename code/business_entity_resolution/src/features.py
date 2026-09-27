"""
Feature engineering module for candidate pair matching.

Calculates high-precision similarity metrics using rapidfuzz (C++ SIMD-accelerated):
- Name fuzzy ratios (ratio, token sort, token set, partial)
- Normalized Levenshtein similarity
- Word token Jaccard similarity
- Prefix & character n-gram matching
- Address fuzzy and token overlap
- Numeric / postal code match
- Country consistency
"""

from typing import List, Tuple
from rapidfuzz import fuzz, distance
import re

FEATURE_NAMES = [
    "name_fuzz_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial_ratio",
    "name_lev_sim",
    "name_jaccard",
    "name_prefix_match",
    "addr_fuzz_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "num_overlap",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
]

_RE_NUM = re.compile(r"\b\d{2,6}\b")


def extract_features(
    s1_name: str, s1_addr: str, s1_country: str,
    c_name: str, c_addr: str, c_country: str
) -> List[float]:
    """Compute numerical feature vector for an entity pair."""
    # 1. Name features
    n_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    n_part = fuzz.partial_ratio(s1_name, c_name) / 100.0
    n_lev = distance.Levenshtein.normalized_similarity(s1_name, c_name)

    # Word Jaccard
    w1 = set(s1_name.split())
    w2 = set(c_name.split())
    n_jacc = len(w1 & w2) / max(len(w1 | w2), 1)

    # Prefix match (first 4 chars)
    s1_p = "".join(ch for ch in s1_name if ch.isalnum())[:4]
    c_p = "".join(ch for ch in c_name if ch.isalnum())[:4]
    n_pref = 1.0 if (s1_p and s1_p == c_p) else 0.0

    # 2. Address features
    a_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
    a_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    a_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0

    aw1 = set(s1_addr.split())
    aw2 = set(c_addr.split())
    a_jacc = len(aw1 & aw2) / max(len(aw1 | aw2), 1)

    # Numeric overlap (street / pin / zip)
    nums1 = set(_RE_NUM.findall(s1_addr))
    nums2 = set(_RE_NUM.findall(c_addr))
    num_match = len(nums1 & nums2) / max(len(nums1 | nums2), 1) if (nums1 or nums2) else 0.5

    # 3. Country match
    c_match = 1.0 if (s1_country and s1_country == c_country) else 0.0

    # 4. Length differences
    len_diff_n = abs(len(s1_name) - len(c_name)) / max(len(s1_name) + len(c_name), 1)
    len_diff_a = abs(len(s1_addr) - len(c_addr)) / max(len(s1_addr) + len(c_addr), 1)

    return [
        n_ratio,
        n_sort,
        n_set,
        n_part,
        n_lev,
        n_jacc,
        n_pref,
        a_ratio,
        a_sort,
        a_set,
        a_jacc,
        num_match,
        c_match,
        len_diff_n,
        len_diff_a,
    ]
