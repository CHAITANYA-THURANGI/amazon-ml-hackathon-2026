"""
Data preprocessing and normalization module.

Handles fast, robust, multilingual text normalization:
- Lowercasing, ASCII conversion, diacritics removal (crucial for French/transliterated entities)
- Multi-jurisdiction legal suffix normalization (US, India, France)
- Tokenization, acronym extraction, phonetic encoding
- Address tokenization and numeric/postal code parsing
"""

import re
import unicodedata
from typing import Set, List, Tuple
import jellyfish

# Pre-compiled regex patterns for speed
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_SPACES = re.compile(r"\s+")
_RE_NUMBERS = re.compile(r"\b\d{2,6}\b")
_RE_POSTAL_US = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_RE_POSTAL_IN_FR = re.compile(r"\b\d{5,6}\b")

# Multi-jurisdiction legal suffix mappings (US, India, France)
LEGAL_SUFFIX_MAP = {
    # US / General
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "company": "co",
    "associates": "assoc",
    "technologies": "tech",
    "technology": "tech",
    "international": "intl",
    "manufacturing": "mfg",
    "services": "svcs",
    "service": "svcs",
    "enterprises": "ent",
    "enterprise": "ent",
    # India
    "private": "pvt",
    "proprietor": "prop",
    # France
    "societe": "soc",
    "anonyme": "sa",
    "responsabilite": "resp",
    "limitee": "lim",
}

STOP_WORDS = {
    "the", "and", "of", "for", "in", "at", "on", "to", "a", "an",
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "plc", "llp",
    "sa", "sarl", "sas", "eurl", "gmbh", "de", "la", "le", "les", "du", "et",
}


def clean_text(text: str) -> str:
    """Normalize text: strip accents, lowercase, remove punctuation, collapse spaces."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Strip unicode accents (é -> e, ç -> c, etc.)
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


def extract_acronym(clean_name: str) -> str:
    """Extract acronym from words (e.g. 'State Bank of India' -> 'sbi')."""
    words = [w for w in clean_name.split() if w not in STOP_WORDS]
    if len(words) >= 2:
        return "".join(w[0] for w in words if w[0].isalnum())
    return ""


def extract_numbers(clean_address: str) -> List[str]:
    """Extract street numbers, pincodes, or zip codes (2-6 digits)."""
    return _RE_NUMBERS.findall(clean_address)


def extract_postal_code(clean_address: str, country: str) -> str:
    """Extract likely postal code based on country patterns."""
    c = (country or "").lower()
    if c == "us":
        m = _RE_POSTAL_US.search(clean_address)
        return m.group(0)[:5] if m else ""
    else:  # India (6 digits), France (5 digits)
        matches = _RE_POSTAL_IN_FR.findall(clean_address)
        if matches:
            # Pincode usually appears towards the end of address
            return matches[-1]
    return ""


def get_phonetic_code(token: str) -> str:
    """Compute Metaphone representation for a token."""
    if not token or len(token) < 2:
        return ""
    try:
        return jellyfish.metaphone(token)
    except Exception:
        return ""
