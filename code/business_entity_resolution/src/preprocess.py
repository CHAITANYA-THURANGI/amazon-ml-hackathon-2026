"""
Data preprocessing and normalization module.

Handles fast, robust text normalization:
- Lowercasing, ASCII cleanup
- Business suffix standardization
- Address tokenization and abbreviation expansion
- Number and postal code extraction for blocking
"""

import re
import unicodedata
import pandas as pd
from typing import Set, List

# Pre-compiled regex patterns for speed
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_SPACES = re.compile(r"\s+")
_RE_NUMBERS = re.compile(r"\b\d{2,6}\b")

# Common business suffixes to normalize or strip from primary match keys
LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "associates": "assoc",
    "technologies": "tech",
    "technology": "tech",
    "international": "intl",
    "manufacturing": "mfg",
    "services": "svcs",
    "service": "svcs",
}

STOP_WORDS = {
    "the", "and", "of", "for", "in", "at", "on", "to", "a", "an",
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "plc", "llp", "sa", "gmbh",
}


def clean_text(text: str) -> str:
    """Normalize text: strip accents, lowercase, remove punctuation, collapse spaces."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Strip unicode accents
    norm = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    norm = norm.lower()
    norm = norm.replace("&", " and ")
    norm = _RE_PUNCT.sub(" ", norm)
    return _RE_SPACES.sub(" ", norm).strip()


def extract_name_tokens(clean_name: str) -> Set[str]:
    """Extract informative words from a cleaned business name."""
    words = clean_name.split()
    tokens = set()
    for w in words:
        if len(w) >= 3 and w not in STOP_WORDS:
            tokens.add(LEGAL_SUFFIX_MAP.get(w, w))
    return tokens


def extract_numbers(clean_address: str) -> List[str]:
    """Extract street numbers, pincodes, or zip codes (2-6 digits)."""
    return _RE_NUMBERS.findall(clean_address)


def normalize_record(name: str, address: str, country: str) -> dict:
    """Produce normalized representation of a business record."""
    cn = clean_text(name)
    ca = clean_text(address)
    cc = (country or "").strip().lower()
    return {
        "name_clean": cn,
        "address_clean": ca,
        "country_clean": cc,
        "tokens": extract_name_tokens(cn),
        "numbers": extract_numbers(ca),
    }
