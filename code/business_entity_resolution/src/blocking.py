"""
Advanced Inverted Index Blocker with Multi-Channel Phonetic & IDF-Weighted Candidate Retrieval.

Key Channels:
1. Lexical Prefixes (3-4 characters)
2. Phonetic Codes (Metaphone on primary tokens)
3. Informative Name Tokens
4. Street Numbers & Postal Codes
5. Acronym Matching

Stores precomputed token & phonetic metadata for ultra-fast downstream feature extraction.
"""

from collections import defaultdict
import math
from typing import Dict, List, Set, Tuple
import jellyfish
from preprocess import (
    clean_text, extract_name_tokens, extract_numbers,
    extract_postal_code, extract_acronym, get_phonetic_code,
)
from config import MAX_CANDIDATES_PER_KEY, MAX_CANDIDATES_PER_S1


class InvertedIndexBlocker:
    """High-recall, IDF-ranked inverted index blocker with precomputed metadata."""

    def __init__(self, max_key_freq: int = MAX_CANDIDATES_PER_KEY, top_k: int = MAX_CANDIDATES_PER_S1):
        self.max_key_freq = max_key_freq
        self.top_k = top_k
        self.index = defaultdict(list)
        # Compact storage: entity_id -> (clean_name, clean_address, country, postal_code, first_tok, meta_code, soundex_code)
        self.records: Dict[str, Tuple[str, str, str, str, str, str, str]] = {}
        # Precomputed IDF weights for keys
        self.key_weights: Dict[str, float] = {}

    def get_keys(self, clean_name: str, clean_address: str, country: str) -> Set[str]:
        """Generate multi-channel blocking keys for an entity."""
        keys = set()
        c = (country or "unk").lower()

        # 1. Lexical prefix keys (length 3 and 4)
        alphanumeric_name = "".join(ch for ch in clean_name if ch.isalnum())
        if len(alphanumeric_name) >= 3:
            keys.add(f"{c}:p3:{alphanumeric_name[:3]}")
        if len(alphanumeric_name) >= 4:
            keys.add(f"{c}:p4:{alphanumeric_name[:4]}")

        # 2. Informative name tokens
        tokens = extract_name_tokens(clean_name)
        for tok in tokens:
            keys.add(f"{c}:tok:{tok}")

        # 3. Phonetic Metaphone key on primary token
        if tokens:
            first_tok = next(iter(tokens))
            ph = get_phonetic_code(first_tok)
            if ph:
                keys.add(f"{c}:ph:{ph}")

        # 4. Acronym key
        acr = extract_acronym(clean_name)
        if len(acr) >= 2:
            keys.add(f"{c}:acr:{acr}")

        # 5. Numeric street numbers & postal codes
        for num in extract_numbers(clean_address):
            keys.add(f"{c}:num:{num}")

        return keys

    def add_record(self, entity_id: str, name: str, address: str, country: str):
        """Index a candidate record and precompute its phonetic & token signatures."""
        cn = clean_text(name)
        ca = clean_text(address)
        cc = (country or "").strip().lower()
        pc = extract_postal_code(ca, cc)

        words = cn.split()
        first_tok = words[0] if words else ""
        meta_code = jellyfish.metaphone(first_tok) if first_tok else ""
        soundex_code = jellyfish.soundex(first_tok) if first_tok else ""

        self.records[entity_id] = (cn, ca, cc, pc, first_tok, meta_code, soundex_code)

        for key in self.get_keys(cn, ca, cc):
            if len(self.index[key]) < self.max_key_freq:
                self.index[key].append(entity_id)

    def prune_and_compute_weights(self):
        """Prune ultra-frequent keys and compute IDF weights for ranking."""
        pruned = 0
        keys_to_remove = [k for k, v in self.index.items() if len(v) >= self.max_key_freq]
        for k in keys_to_remove:
            del self.index[k]
            pruned += 1

        total_records = max(len(self.records), 1)
        for k, v in self.index.items():
            freq = len(v)
            self.key_weights[k] = math.log1p(total_records / freq)

        return pruned

    def get_candidates(self, clean_name: str, clean_address: str, country: str) -> List[str]:
        """Retrieve and rank top candidates using IDF-weighted key overlap."""
        keys = self.get_keys(clean_name, clean_address, country)
        cand_scores = defaultdict(float)

        for key in keys:
            hits = self.index.get(key)
            if hits:
                weight = self.key_weights.get(key, 1.0)
                for cid in hits:
                    cand_scores[cid] += weight

        if not cand_scores:
            return []

        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_cands[:self.top_k]]
