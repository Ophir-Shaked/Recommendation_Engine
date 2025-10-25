# -*- coding: utf-8 -*-  # Encoding declaration to safely handle Unicode characters (cross-platform compatibility).

from __future__ import annotations  # Allows forward type references (useful for self-referential annotations).
from typing import List, Optional  # Typing aliases for lists and optional returns (improves clarity and tooling).
import numpy as np  # Fast numerical library for dense matrix/vector operations.

# -------------------- Import configuration constants --------------------
from .env_and_imports import (
    TF_MIN_DF, TF_MAX_DF, TF_BODY_MAXFEAT,   # TF-IDF frequency and vocabulary limits.
    GLOBAL_RS_SVD, LSA_DIM, W2V_EPOCHS, W2V_NEGATIVE,  # Random seeds, dimensions, epochs, and negative-sampling constants.
    GENSIM_AVAILABLE  # Flag signaling if gensim is installed and can be used for Word2Vec.
)

# -------------------- Import sklearn components --------------------
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts raw text → sparse TF-IDF matrix.
from sklearn.decomposition import TruncatedSVD               # Performs LSA (dimensionality reduction on TF-IDF).
from sklearn.preprocessing import Normalizer                 # Normalizes vectors to unit length for cosine similarity.
from sklearn.pipeline import make_pipeline                   # Convenience to chain SVD + Normalizer into one callable.

# -------------------- Optional gensim import --------------------
try:
    from gensim.models import Word2Vec  # type: ignore
    # If gensim is present, true neural Word2Vec embeddings can be trained.
except Exception:
    Word2Vec = None  # type: ignore
    # Fallback — gensim not available (e.g. minimal Colab runtime). We’ll use pseudo-vectors from LSA instead.

# ========================================================================
#                           GLOBAL EMBEDDER
# ========================================================================
class GlobalEmbedder:
    """TF-IDF + (optional) LSA embedder for corpus-level text representation."""
    def __init__(self, use_lsa: bool = True, lsa_dim: int = 256):
        self.use_lsa = use_lsa        # Flag toggling LSA dimensionality reduction.
        self.lsa_dim = lsa_dim        # Target dimension for SVD projection.
        self.vec = TfidfVectorizer(   # Configure TF-IDF vectorizer.
            min_df=TF_MIN_DF, max_df=TF_MAX_DF,  # Prune rare and overly common terms.
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english",  # Unigrams+bigrams, log(TF) weighting.
            max_features=TF_BODY_MAXFEAT  # Cap vocabulary size to avoid OOM on large corpora.
        )
        self.pipe = None              # Placeholder for combined SVD+Normalizer pipeline.
        self.svd: Optional[TruncatedSVD] = None  # Will hold fitted SVD once trained.
        self.norm: Optional[Normalizer] = None   # Will hold Normalizer instance.

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit TF-IDF (and optionally LSA) to a corpus, returning document embeddings."""
        X = self.vec.fit_transform(texts)  # Learn vocabulary + compute TF-IDF matrix.
        if self.use_lsa:
            # Apply Latent Semantic Analysis for dense, low-rank semantic vectors.
            self.svd = TruncatedSVD(n_components=self.lsa_dim, random_state=GLOBAL_RS_SVD)
            self.norm = Normalizer(copy=False)
            self.pipe = make_pipeline(self.svd, self.norm)
            return self.pipe.fit_transform(X)  # Fit + transform in one step.
        return X  # If no LSA requested, return sparse TF-IDF matrix.

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using the fitted model."""
        X = self.vec.transform(texts)      # Reuse learned vocabulary.
        if self.pipe is not None:
            X = self.pipe.transform(X)     # Apply SVD + normalization if available.
        return X

    def term_vector(self, term: str) -> Optional[np.ndarray]:
        """Return LSA vector for a single token, if available."""
        if (self.svd is None) or (self.vec is None):
            return None  # Model not yet fitted.
        vocab = self.vec.vocabulary_       # Token → column index mapping.
        idx = vocab.get(term.lower())      # Case-insensitive lookup.
        if idx is None:
            return None  # Token not found in training vocabulary.
        return self.svd.components_[:, idx].copy()  # Retrieve SVD projection column for that term.

# ========================================================================
#                           WORD2VEC / PSEUDO2VEC
# ========================================================================
class W2VEmbedder:
    """Train a Word2Vec model if gensim is available, otherwise fall back to LSA-term pseudo2vec."""
    def __init__(self, global_embedder: GlobalEmbedder, dim: int = 64, window: int = 6,
                 min_count: int = 2, sg: int = 1):
        self.dim = dim                  # Embedding dimension (vector size).
        self.window = window            # Context window size for co-occurrence.
        self.min_count = min_count      # Minimum frequency threshold for vocabulary inclusion.
        self.sg = sg                    # Model type: 1=Skip-gram (default), 0=CBOW.
        self.global_emb = global_embedder  # Reference to GlobalEmbedder for fallback pseudo-vectors.
        self.model = None               # Will hold trained gensim Word2Vec model.
        self.use_gensim = False         # Indicates whether gensim Word2Vec was trained successfully.

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Basic tokenizer stripping punctuation and lowercasing."""
        # Retains only alphanumeric characters; replaces others with space; filters 1-char fragments.
        return [t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split() if len(t) > 1]

    def fit(self, texts: List[str]):
        """Train Word2Vec model if possible, otherwise enable pseudo-vector fallback."""
        sents = [self._tokenize(t) for t in texts]  # Convert corpus to tokenized sentences.
        if GENSIM_AVAILABLE:
            try:
                self.model = Word2Vec(
                    sentences=sents,          # Tokenized input sentences.
                    vector_size=self.dim,     # Dimensionality of embeddings.
                    window=self.window,       # Context window for co-occurrence learning.
                    min_count=self.min_count, # Discard infrequent tokens.
                    sg=self.sg,               # Skip-gram vs CBOW.
                    workers=1,                # Single-threaded for reproducibility.
                    epochs=W2V_EPOCHS,        # Training epochs (constant from env_and_imports).
                    negative=W2V_NEGATIVE     # Number of negative samples.
                )
                self.use_gensim = True
                print("[W2V] Trained gensim.Word2Vec")  # Log successful training.
                return
            except Exception as e:
                print(f"[W2V] gensim training failed, fallback: {e}")  # Safe fallback on error.
        # If gensim unavailable or training failed, revert to LSA pseudo2vec.
        self.use_gensim = False
        self.model = None
        print("[W2V] Using LSA-term pseudo2vec (fallback)")

    def doc_vector(self, text: str) -> np.ndarray:
        """Compute a document vector as mean of word vectors (gensim or pseudo)."""
        toks = self._tokenize(text)
        if not toks:
            # Empty doc → zero vector of appropriate dimension.
            return np.zeros(self.dim if self.use_gensim else LSA_DIM, dtype=float)
        vecs = []
        if self.use_gensim and self.model is not None:
            # Gensim path — average true word embeddings.
            for w in toks:
                if w in self.model.wv:
                    vecs.append(self.model.wv[w])
            if not vecs:
                return np.zeros(self.model.vector_size, dtype=float)  # Handle unseen words.
            v = np.mean(vecs, axis=0)
            return v / (np.linalg.norm(v) + 1e-9)  # Normalize to unit length.
        else:
            # Fallback path — average LSA term vectors from GlobalEmbedder.
            for w in toks:
                tv = self.global_emb.term_vector(w)
                if tv is not None:
                    vecs.append(tv)
            if not vecs:
                return np.zeros(LSA_DIM, dtype=float)
            v = np.mean(vecs, axis=0)
            return v / (np.linalg.norm(v) + 1e-9)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform multiple documents into stacked embedding matrix."""
        # Compute vector for each text and stack vertically into float32 array.
        return np.vstack([self.doc_vector(t) for t in texts]).astype(np.float32)
