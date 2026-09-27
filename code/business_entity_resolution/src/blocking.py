"""
Blocking / Candidate Generation — reduces the comparison space.

Strategies:
- Country-based blocking (only compare records from the same country)
- TF-IDF character n-gram similarity on concatenated name+address
- Exact/phonetic name key blocking

The output is a candidate set: for each S1 entity, a list of S2/S3 entity IDs
that are plausible matches worth scoring with the full feature set.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from tqdm import tqdm

from config import (
    ENTITY_ID_COL,
    TFIDF_NGRAM_RANGE,
    TFIDF_TOP_K,
)


def create_blocking_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate normalized name + address into a single text field for TF-IDF."""
    return df["name_clean"].fillna("") + " " + df["address_clean"].fillna("")


def tfidf_blocking(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    batch_size: int = 500,
) -> dict:
    """
    TF-IDF character n-gram blocking.

    For each S1 entity, find the top-K most similar candidates from S2/S3
    based on TF-IDF cosine similarity of name+address text.

    Args:
        s1_df: Source 1 DataFrame with 'name_clean' and 'address_clean' columns.
        candidates_df: Combined S2+S3 DataFrame.
        top_k: Number of top candidates to return per S1 entity.
        ngram_range: Character n-gram range for TF-IDF.
        batch_size: Process S1 entities in batches to manage memory.

    Returns:
        dict: {s1_entity_id: [candidate_entity_ids]}
    """
    print(f"  Building TF-IDF index (ngram={ngram_range})...")

    # Create text representations
    s1_text = create_blocking_text(s1_df)
    cand_text = create_blocking_text(candidates_df)

    # Fit TF-IDF on all text, transform separately
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=100_000,
        sublinear_tf=True,
    )

    all_text = pd.concat([s1_text, cand_text], ignore_index=True)
    vectorizer.fit(all_text)

    cand_vectors = vectorizer.transform(cand_text)
    cand_ids = candidates_df[ENTITY_ID_COL].values

    # Process S1 in batches
    blocking_result = {}
    n_batches = (len(s1_df) + batch_size - 1) // batch_size

    print(f"  Scoring {len(s1_df)} S1 entities against {len(candidates_df)} candidates...")

    for batch_idx in tqdm(range(n_batches), desc="  TF-IDF blocking"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(s1_df))

        batch_text = s1_text.iloc[start:end]
        batch_vectors = vectorizer.transform(batch_text)
        batch_ids = s1_df[ENTITY_ID_COL].values[start:end]

        # Cosine similarity: (batch_size, n_candidates)
        sims = cosine_similarity(batch_vectors, cand_vectors)

        for i, s1_id in enumerate(batch_ids):
            # Get top-K candidate indices
            if top_k < len(cand_ids):
                top_indices = np.argpartition(sims[i], -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(sims[i][top_indices])[::-1]]
            else:
                top_indices = np.argsort(sims[i])[::-1]

            # Filter out zero-similarity candidates
            valid = [
                cand_ids[j] for j in top_indices if sims[i][j] > 0.0
            ]
            blocking_result[s1_id] = valid

    return blocking_result


def country_aware_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
) -> dict:
    """
    Country-aware blocking: only compare S1 entities to S2/S3 records
    from the same country (or all if country is missing).

    Returns:
        dict: {s1_entity_id: [candidate_entity_ids]}
    """
    candidates_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Group by country
    countries = s1_df["country_clean"].unique()
    all_blocking = {}

    for country in countries:
        print(f"\n  Processing country: '{country}'")

        s1_country = s1_df[s1_df["country_clean"] == country]
        cand_country = candidates_df[candidates_df["country_clean"] == country]

        if len(cand_country) == 0:
            # No candidates for this country — mark all as empty
            for s1_id in s1_country[ENTITY_ID_COL]:
                all_blocking[s1_id] = []
            continue

        country_blocking = tfidf_blocking(s1_country, cand_country, top_k=top_k)
        all_blocking.update(country_blocking)

    # Handle S1 entities not yet in results (e.g., empty country)
    for s1_id in s1_df[ENTITY_ID_COL]:
        if s1_id not in all_blocking:
            all_blocking[s1_id] = []

    return all_blocking


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
) -> dict:
    """
    Main entry point for candidate generation.

    Combines multiple blocking strategies and returns the union of candidates.

    Returns:
        dict: {s1_entity_id: list[candidate_entity_ids]}
    """
    print("=" * 60)
    print("CANDIDATE GENERATION (BLOCKING)")
    print("=" * 60)

    candidates = country_aware_blocking(s1_df, s2_df, s3_df, top_k=top_k)

    # Summary stats
    n_total = sum(len(v) for v in candidates.values())
    n_empty = sum(1 for v in candidates.values() if len(v) == 0)
    print(f"\n  Total S1 entities: {len(candidates)}")
    print(f"  Total candidate pairs: {n_total:,}")
    print(f"  Avg candidates per S1: {n_total / max(len(candidates), 1):.1f}")
    print(f"  S1 entities with no candidates: {n_empty}")

    return candidates
