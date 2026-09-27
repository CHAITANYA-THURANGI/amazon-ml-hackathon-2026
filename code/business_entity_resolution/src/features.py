"""
Feature engineering for candidate pairs.

Computes similarity features between a Source 1 entity and each candidate
(S2/S3) entity for input to the matching model.

Feature categories:
- Name similarity (Jaccard, Levenshtein, token overlap, TF-IDF cosine)
- Address similarity (token overlap, edit distance, component matching)
- Country match
- Combined features
"""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


# ─── String Similarity Utilities ─────────────────────────────────────────────

def jaccard_similarity(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def token_sort_ratio(s1: str, s2: str) -> float:
    """Fuzzy token sort ratio (handles word reordering)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def token_set_ratio(s1: str, s2: str) -> float:
    """Fuzzy token set ratio (handles subsets and extra tokens)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def partial_ratio(s1: str, s2: str) -> float:
    """Fuzzy partial ratio (best substring match)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1 - normalized distance)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.normalized_distance(s1, s2)
    return 1.0 - dist


def char_ngram_overlap(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    ngrams1 = set(s1[i:i + n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i + n] for i in range(len(s2) - n + 1))
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_pair_features(s1_row: dict, cand_row: dict) -> dict:
    """
    Extract all similarity features for a (S1, candidate) pair.

    Args:
        s1_row: Dict with keys: name_clean, address_clean, country_clean, etc.
        cand_row: Dict with keys: name_clean, address_clean, country_clean, etc.

    Returns:
        dict: Feature name → value
    """
    s1_name = s1_row.get("name_clean", "")
    s1_addr = s1_row.get("address_clean", "")
    s1_country = s1_row.get("country_clean", "")

    c_name = cand_row.get("name_clean", "")
    c_addr = cand_row.get("address_clean", "")
    c_country = cand_row.get("country_clean", "")

    features = {}

    # ── Name features ──
    features["name_jaccard"] = jaccard_similarity(s1_name, c_name)
    features["name_levenshtein"] = normalized_levenshtein(s1_name, c_name)
    features["name_token_sort"] = token_sort_ratio(s1_name, c_name)
    features["name_token_set"] = token_set_ratio(s1_name, c_name)
    features["name_partial"] = partial_ratio(s1_name, c_name)
    features["name_char3gram"] = char_ngram_overlap(s1_name, c_name, n=3)
    features["name_char4gram"] = char_ngram_overlap(s1_name, c_name, n=4)

    # ── Address features ──
    features["addr_jaccard"] = jaccard_similarity(s1_addr, c_addr)
    features["addr_levenshtein"] = normalized_levenshtein(s1_addr, c_addr)
    features["addr_token_sort"] = token_sort_ratio(s1_addr, c_addr)
    features["addr_token_set"] = token_set_ratio(s1_addr, c_addr)
    features["addr_partial"] = partial_ratio(s1_addr, c_addr)
    features["addr_char3gram"] = char_ngram_overlap(s1_addr, c_addr, n=3)

    # ── Country features ──
    features["country_match"] = 1.0 if s1_country == c_country else 0.0

    # ── Combined features ──
    combined_s1 = s1_name + " " + s1_addr
    combined_c = c_name + " " + c_addr
    features["combined_jaccard"] = jaccard_similarity(combined_s1, combined_c)
    features["combined_token_sort"] = token_sort_ratio(combined_s1, combined_c)

    # ── Length features ──
    features["name_len_ratio"] = (
        min(len(s1_name), len(c_name)) / max(len(s1_name), len(c_name))
        if s1_name and c_name else 0.0
    )
    features["addr_len_ratio"] = (
        min(len(s1_addr), len(c_addr)) / max(len(s1_addr), len(c_addr))
        if s1_addr and c_addr else 0.0
    )

    return features


def get_feature_names() -> list:
    """Return the ordered list of feature names."""
    # Extract from a dummy pair to get consistent ordering
    dummy = extract_pair_features(
        {"name_clean": "a", "address_clean": "b", "country_clean": "us"},
        {"name_clean": "c", "address_clean": "d", "country_clean": "us"},
    )
    return list(dummy.keys())
