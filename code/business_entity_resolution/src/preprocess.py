"""
Data preprocessing — loading, cleaning, and normalizing business records.

Handles:
- Loading TSV files with proper encoding
- Business name normalization (lowercasing, suffix standardization, punctuation)
- Address normalization (abbreviation expansion, whitespace, component cleaning)
- Country-aware preprocessing
"""

import re
import pandas as pd
from config import (
    TSV_SEP,
    ENTITY_ID_COL,
    BUSINESS_NAME_COL,
    BUSINESS_ADDRESS_COL,
    COUNTRY_COL,
)


# ─── Name Normalization ─────────────────────────────────────────────────────

# Common legal suffix mappings
LEGAL_SUFFIXES = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bco\b": "company",
    r"\bllp\b": "limited liability partnership",
    r"\bplc\b": "public limited company",
    r"\bhldgs?\b": "holdings",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\bsvcs?\b": "services",
    r"\bengr?g?\b": "engineering",
    r"\btech\b": "technology",
    r"\bgrp\b": "group",
    r"\bassoc\b": "associates",
    r"\bindust?\b": "industries",
}


def normalize_name(name: str) -> str:
    """Normalize a business name for comparison."""
    if not isinstance(name, str) or not name.strip():
        return ""

    text = name.lower().strip()

    # Replace & with "and"
    text = text.replace("&", " and ")

    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Expand legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Address Normalization ───────────────────────────────────────────────────

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpky?\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bdist\b": "district",
    r"\bnagar\b": "nagar",
}


def normalize_address(address: str) -> str:
    """Normalize a business address for comparison."""
    if not isinstance(address, str) or not address.strip():
        return ""

    text = address.lower().strip()

    # Remove punctuation except alphanumeric, spaces, and hyphens
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Expand abbreviations
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_source(path: str) -> pd.DataFrame:
    """Load a source TSV file and add normalized columns."""
    df = pd.read_csv(path, sep=TSV_SEP, dtype=str)

    # Fill NaN with empty strings
    for col in [BUSINESS_NAME_COL, BUSINESS_ADDRESS_COL, COUNTRY_COL]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Add normalized columns
    df["name_clean"] = df[BUSINESS_NAME_COL].apply(normalize_name)
    df["address_clean"] = df[BUSINESS_ADDRESS_COL].apply(normalize_address)
    df["country_clean"] = df[COUNTRY_COL].str.lower().str.strip()

    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load the ground truth TSV file."""
    gt = pd.read_csv(path, sep=TSV_SEP, dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    return gt


if __name__ == "__main__":
    # Quick test
    from config import TRAIN_SOURCE1

    df = load_source(TRAIN_SOURCE1)
    print(f"Loaded {len(df)} records from Source 1")
    print(df.head())
    print(f"\nNormalized name sample: '{df['name_clean'].iloc[0]}'")
    print(f"Normalized address sample: '{df['address_clean'].iloc[0]}'")
