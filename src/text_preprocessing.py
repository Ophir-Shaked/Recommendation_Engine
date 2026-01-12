# Text cleaning + tokenization + Transformers 

import re                                    # Regular expressions for pattern-based cleanup
from typing import Any, List, Optional       # Type hints (Any/List/Optional)

import numpy as np                           # Numerical arrays (used in sklearn signatures + later embedding code)
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for sklearn-compatible transformers


# Regex patterns (compiled once for speed)


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)        # Match URLs like http(s)://... or www....
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)   # Match simple email patterns like a@b.com
_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")                               # Match integers or decimals (e.g., 12 or 12.5)

# Helper: conversion to string

def simple_clean_text(s: Any) -> str:
    """
    Convert input to a safe string:
    - If None -> empty string
    - Else -> str(s)
    """
    return "" if s is None else str(s)

# Normalization (the core cleanup)

def normalize_text(s: Any, cfg: "Config") -> str:
    """
    Normalize text consistently (used by TF-IDF and Word2Vec tokenization):

    Steps:
    1) Convert to safe string (handle None)
    2) Lowercase everything
    3) Replace URLs/emails/numbers with special tokens (URL/EMAIL/NUM)
    4) Remove everything except: a-z, 0-9, space, underscore, dash, plus
    5) Collapse multiple spaces into one, trim ends
    """
    s = simple_clean_text(s)                 # Ensure we never crash on None / non-string inputs
    s = s.lower()                            # Lowercase for normalization

    s = _URL_RE.sub(" URL ", s)              # Replace any URL with token "URL"
    s = _EMAIL_RE.sub(" EMAIL ", s)          # Replace any email with token "EMAIL"
    s = _NUM_RE.sub(" NUM ", s)              # Replace any number with token "NUM"

    s = re.sub(r"[^a-z0-9 _\-\+]", " ", s)   # Remove unwanted chars (keep letters/digits/_-+ and spaces)
    s = re.sub(r"\s+", " ", s).strip()       # Collapse whitespace and trim
    return s

# Tokenization for Word2Vec

def tokenize(s: Any, cfg: "Config") -> List[str]:
    """
    Tokenize normalized text into simple whitespace tokens.

    Rules:
    - First run normalize_text()
    - Split by spaces
    - Keep tokens of length >= 2
    - Optionally cap number of tokens per document (cfg.w2v_max_tokens_per_doc)
    """
    s = normalize_text(s, cfg)               # Normalize text first (same normalization used everywhere)
    toks = [t for t in s.split(" ") if len(t) >= 2]  # Keep only tokens with length >= 2

    max_toks = int(cfg.w2v_max_tokens_per_doc or 0)  # Max tokens cap (0 => no cap)
    if max_toks > 0 and len(toks) > max_toks:        # If cap is enabled and we exceed it...
        toks = toks[:max_toks]                       # ...truncate to the cap
    return toks

# Sklearn transformer: raw texts -> normalized texts

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that converts a list of raw texts
    into normalized texts (strings), so it can be used inside a Pipeline.
    """

    def __init__(self, cfg: "Config"):
        self.cfg = cfg                              # Store config so normalization uses the same parameters

    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        return self                                 # Nothing to learn, so just return self

    def transform(self, X: List[str]) -> List[str]:
        # Apply normalize_text to each input string
        return [normalize_text(x, self.cfg) for x in X]

#  print so we know the cell executed successfully
print("[OK] normalize_text/tokenize/TextPreprocessor are defined.")
