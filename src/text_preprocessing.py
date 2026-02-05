# Text cleaning + tokenization + stemming/lemmatization


import re                                    # Regular expressions for pattern-based cleanup
from typing import Any, List, Optional       # Type hints (Any/List/Optional)

import numpy as np                           # Numerical arrays (used in sklearn signatures + later embedding code)
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for sklearn-compatible transformers

# Optional NLP tools (NLTK) for stemming / lemmatization
# If you don't have nltk installed:
#   pip install nltk
#
# Lemmatization requires downloads:
#   import nltk
#   nltk.download("wordnet")
#   nltk.download("omw-1.4")
try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
    PorterStemmer = None
    WordNetLemmatizer = None


# Regex patterns (compiled once for speed)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)        # Match URLs like http(s)://... or www....
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)   # Match simple email patterns like a@b.com
_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")                               # Match integers or decimals (e.g., 12 or 12.5)


def simple_clean_text(s: Any) -> str:
    """
    Convert input to a safe string:
    - If None -> empty string
    - Else -> str(s)
    """
    return "" if s is None else str(s)


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


def _get_cfg_bool(cfg: "Config", name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(cfg, name))
    except Exception:
        return default


def _apply_stem_or_lemma(norm_s: str, cfg: "Config") -> str:
    """
    Apply optional stemming/lemmatization to an already-normalized string.

    IMPORTANT:
    - This is intended for TF-IDF path only (used in TextPreprocessor).
    - Word2Vec uses tokenize() below and will NOT apply stemming/lemmatization
      unless you explicitly add it there.
    """
    use_stem = _get_cfg_bool(cfg, "tfidf_use_stemming", False)
    use_lemma = _get_cfg_bool(cfg, "tfidf_use_lemmatization", False)

    # By design, allow at most one of them to be True.
    if use_stem and use_lemma:
        raise ValueError("Config error: choose only one of tfidf_use_stemming or tfidf_use_lemmatization.")

    if not use_stem and not use_lemma:
        return norm_s

    if not _NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is not available but stemming/lemmatization was requested. "
            "Install it with: pip install nltk"
        )

    toks = [t for t in norm_s.split(" ") if t]
    if not toks:
        return ""

    if use_stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(t) for t in toks]
        return " ".join(toks)

    # use_lemma
    lemm = WordNetLemmatizer()
    # Note: default lemmatize POS is noun; for better results you'd add POS tagging.
    toks = [lemm.lemmatize(t) for t in toks]
    return " ".join(toks)


def tokenize(s: Any, cfg: "Config") -> List[str]:
    """
    Tokenize normalized text into simple whitespace tokens (used by Word2Vec).

    Rules:
    - First run normalize_text()
    - Split by spaces
    - Keep tokens of length >= 2
    - Optionally cap number of tokens per document (cfg.w2v_max_tokens_per_doc)

    NOTE:
    - This DOES NOT apply stemming/lemmatization (by design).
      If you want stemming/lemmatization for Word2Vec too, you can optionally
      apply _apply_stem_or_lemma() here behind separate flags.
    """
    s = normalize_text(s, cfg)                               # Normalize text first
    toks = [t for t in s.split(" ") if len(t) >= 2]         # Keep only tokens with length >= 2

    max_toks = int(getattr(cfg, "w2v_max_tokens_per_doc", 0) or 0)  # Max tokens cap (0 => no cap)
    if max_toks > 0 and len(toks) > max_toks:                         # If cap is enabled and exceeded
        toks = toks[:max_toks]                                        # Truncate
    return toks


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that converts a list of raw texts
    into normalized texts (strings), so it can be used inside a Pipeline.

    In this version, it can optionally apply stemming or lemmatization
    specifically for TF-IDF (controlled by cfg.tfidf_use_stemming / cfg.tfidf_use_lemmatization).
    """

    def __init__(self, cfg: "Config"):
        self.cfg = cfg

    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        return self

    def transform(self, X: List[str]) -> List[str]:
        # 1) Normalize text (shared normalization)
        out = [normalize_text(x, self.cfg) for x in X]

        # 2) Optional stemming / lemmatization for TF-IDF
        #    (This affects TF-IDF pipeline because TextPreprocessor is used there.)
        if _get_cfg_bool(self.cfg, "tfidf_use_stemming", False) or _get_cfg_bool(self.cfg, "tfidf_use_lemmatization", False):
            out = [_apply_stem_or_lemma(s, self.cfg) for s in out]

        return out


print("[OK] normalize_text/tokenize/TextPreprocessor (with optional TF-IDF stemming/lemmatization) are defined.")
