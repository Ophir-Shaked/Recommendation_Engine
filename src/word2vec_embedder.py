#  Word2Vec document embedder 

from typing import List, Optional, Any             # Type hints for readability and safety

import numpy as np                                 # Arrays + random generator + vector math
from sklearn.base import BaseEstimator, TransformerMixin  # Make this compatible with sklearn Pipelines
from gensim.models import Word2Vec                 # Gensim Word2Vec implementation


class Word2VecDocEmbedder(BaseEstimator, TransformerMixin):
    """
    Fold-safe Word2Vec embedder.

    Why "fold-safe"?
    - In cross-validation, each fold should train ONLY on that fold's training data.
    - This class trains Word2Vec inside fit(), so every fold gets its own Word2Vec model.
    - transform() embeds each document as the mean of its word vectors.

    Output:
    - transform(X) returns a NumPy array of shape (n_docs, w2v_dim)
    """

    def __init__(self, cfg: "Config"):
        self.cfg = cfg                              # Store config (dimensions, window, seed, etc.)
        self.w2v_: Optional[Word2Vec] = None        # Will hold the trained Word2Vec model after fit()

    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        """
        Train Word2Vec on the current fold's training texts (X).
        y is ignored, but included for sklearn compatibility.
        """

        #  Tokenize each document into a list of tokens
        #    tokenize() comes from CELL 3 (so CELL 3 must run before this cell)
        sents = [tokenize(x, self.cfg) for x in X]  # List[List[str]] (one list of tokens per doc)

        #  Remove empty token lists (docs that become empty after normalization)
        sents = [s for s in sents if len(s) > 0]

        #  If nothing usable remains, keep model as None and exit gracefully
        if len(sents) == 0:
            self.w2v_ = None                        # No model can be trained
            return self                          

        #  Optional: subsample documents for speed (cap training docs)
        max_docs = int(self.cfg.w2v_max_train_docs or 0)  # 0 means no cap
        if max_docs > 0 and len(sents) > max_docs:
            rng = np.random.default_rng(int(self.cfg.seed))       # Reproducible random generator
            idx = rng.choice(len(sents), size=max_docs, replace=False)  # Choose a subset
            sents = [sents[i] for i in idx]                       # Keep only the sampled sentences

        # 5) Create a Word2Vec model instance with config hyperparameters
        self.w2v_ = Word2Vec(
            vector_size=int(self.cfg.w2v_dim),     # Embedding dimension
            window=int(self.cfg.w2v_window),       # Context window size
            min_count=int(self.cfg.w2v_min_count), # Ignore rare words below this count
            sg=int(self.cfg.w2v_sg),               # 1=skip-gram, 0=CBOW
            negative=int(self.cfg.w2v_negative),   # Negative sampling count
            workers=int(self.cfg.w2v_workers),     # Threads/workers (Colab often set to 1 for stability)
            seed=int(self.cfg.seed),               # Seed for reproducibility
        )

        #  Build vocabulary from the training sentences
        self.w2v_.build_vocab(sents)

        #  Train the model for a number of epochs
        self.w2v_.train(
            sents,                                 # Training data: list of token lists
            total_examples=len(sents),              # How many sentences
            epochs=int(self.cfg.w2v_epochs)         # Training epochs
        )

        #  Return self (sklearn style)
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        """
        Convert each document into a fixed-size vector by averaging word vectors.

        If the model was not trained (w2v_ is None), returns zeros.
        """

        #  Prepare output matrix (n_docs x dim) filled with zeros by default
        n = len(X)                                  # Number of documents
        dim = int(self.cfg.w2v_dim)                 # Embedding dimension
        out = np.zeros((n, dim), dtype=np.float32)  # Default: all zeros

        #  If no model exists (fit() couldn't train), return zeros safely
        if self.w2v_ is None:
            return out

        #   the KeyedVectors object (fast word->vector lookup)
        kv = self.w2v_.wv

        #  For each document, tokenize and average the vectors of known tokens
        for i, txt in enumerate(X):
            toks = tokenize(txt, self.cfg)                 # Tokenize this doc (same tokenize as training)
            vecs = [kv[t] for t in toks if t in kv]        # Collect vectors only for words in vocabulary

            # If we got at least one vector, average them into a doc embedding
            if vecs:
                out[i] = np.asarray(vecs, dtype=np.float32).mean(axis=0)

        #  Return the (n_docs, dim) matrix
        return out


#  print so we know the cell executed successfully
print("[OK] Word2VecDocEmbedder is defined.")
