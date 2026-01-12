#  Data + Reps/Models + Featurize 

import os                                               # File system utilities (folders, paths)
from typing import Dict, List, Tuple, Optional, Any      # Type hints for clarity

import numpy as np                                      # Numerical arrays and vector ops
from datasets import load_dataset                        # HuggingFace datasets loader

#  scikit-learn utilities 
from sklearn.base import clone                           # Clone estimators/pipelines (fresh model per fold)
from sklearn.pipeline import Pipeline                    # Chain transformers + estimators
from sklearn.preprocessing import Normalizer             # Normalize vectors (L2 norm)
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text -> TF-IDF sparse vectors
from sklearn.decomposition import TruncatedSVD           # LSA/latent semantic analysis projection

#  Models 
from sklearn.linear_model import LogisticRegression      # Linear model baseline
from sklearn.svm import LinearSVC                        # Linear SVM baseline
from sklearn.ensemble import HistGradientBoostingClassifier  # Strong non-linear baseline (trees)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist (safe to call repeatedly)."""
    os.makedirs(path, exist_ok=True)                     # exist_ok=True prevents error if already exists


def load_dataset_hf(cfg: "Config") -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Load the dataset and return:
      - texts: List[str]
      - y: np.ndarray of labels
      - times: Optional[np.ndarray] (timestamps if available and enabled)

    NO-LEAKAGE RULE:
    - We use ONLY the dataset's official TRAIN split for cross-validation.
    - We do NOT touch the official TEST split during CV (prevents leakage).
    """
    ds = load_dataset(cfg.dataset_name)                  # Download/cache dataset locally via HF

    # Choose the train split if available; otherwise use the first available split (fallback)
    full = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    # Shuffle the dataset so the first N docs are not biased by original ordering
    full = full.shuffle(seed=int(cfg.seed))

    # Cap dataset size for speed/memory (use only first n after shuffle)
    n = min(int(cfg.max_docs), len(full))
    full = full.select(range(n))

    # Extract raw texts (safe conversion) and labels
    # NOTE: simple_clean_text() must be defined earlier (CELL 3)
    texts = [simple_clean_text(x) for x in full[cfg.text_field]]
    y = np.asarray(full[cfg.label_field], dtype=int)

    # Optional: attempt to use real timestamps (many toy datasets do not have them)
    times: Optional[np.ndarray] = None                   # Default: no real time info
    if bool(cfg.topk_use_real_time):                     # Only attempt if the user enabled it
        # Candidate column names that might represent time
        for cand in ["timestamp", "time", "date", "published", "created_at", "created", "published_at"]:
            if cand in full.column_names:                # If the dataset has such a column
                try:
                    raw = full[cand]                     # Raw time column (could be ints, floats, strings, etc.)

                    # If numeric timestamps -> use them
                    if isinstance(raw[0], (int, float, np.integer, np.floating)):
                        times = np.asarray(raw, dtype=np.float32)

                    # Else (strings/dates) -> fallback to a synthetic increasing timeline
                    else:
                        times = np.arange(len(texts), dtype=np.float32)

                except Exception:
                    # If anything breaks, fallback to synthetic time
                    times = np.arange(len(texts), dtype=np.float32)

                break                                    # Stop after finding the first time-like column

    # Print summary so you can verify what was loaded
    print(f"[DATA] Loaded {len(texts)} docs from '{cfg.dataset_name}' (cap={cfg.max_docs})")
    return texts, y, times


def make_binary_target(y: np.ndarray, pos_class: int) -> np.ndarray:
    """
    Convert multi-class labels into a binary target:
      - 1 if original label == pos_class
      - 0 otherwise
    """
    yb = (y == int(pos_class)).astype(int)               # Vectorized conversion to 0/1
    print(f"[DATA] Binary target: pos_class={pos_class} | pos_rate={float(np.mean(yb)):.4f}")
    return yb


def build_representations(cfg: "Config") -> Dict[str, Any]:
    """
    Build "representations" = text->feature pipelines.

    IMPORTANT (no leakage):
    - These objects will be cloned and fitted ONLY on the fold's TRAIN texts inside featurize().
    """
    reps: Dict[str, Any] = {}                            # Will store representation pipelines by name

    # TF-IDF + LSA (TruncatedSVD) pipeline:
    # - clean: normalize text
    # - tfidf: build sparse term-document matrix
    # - svd: reduce to dense LSA space
    # - norm: L2 normalize for cosine-friendly geometry
    reps["tfidf_lsa"] = Pipeline([
        ("clean", TextPreprocessor(cfg)),                # TextPreprocessor defined in CELL 3
        ("tfidf", TfidfVectorizer(
            max_features=int(cfg.tfidf_max_features),    # Limit vocabulary size
            ngram_range=tuple(cfg.tfidf_ngram_range),    # E.g., (1,2) = unigrams + bigrams
            min_df=int(cfg.tfidf_min_df),                # Ignore extremely rare terms
            stop_words=cfg.tfidf_stop_words              # Optional stop words ("english")
        )),
        ("svd", TruncatedSVD(
            n_components=int(cfg.lsa_dim),               # Output dimension for LSA space
            random_state=int(cfg.random_state)           # Reproducibility
        )),
        ("norm", Normalizer(copy=False)),                # L2 normalize vectors
    ])

    # Optional Word2Vec document average embedding pipeline
    if bool(RUN_W2V):                                    # RUN_W2V is a global switch
        reps["w2v_avg"] = Pipeline([
            ("w2v", Word2VecDocEmbedder(cfg)),            # Trains Word2Vec per fold 
            ("norm", Normalizer(copy=False)),             # Normalize embeddings
        ])

    # Print enabled representations so you know exactly what's used
    print("[REPS] Enabled:", sorted(reps.keys()))
    return reps


def build_models(cfg: "Config") -> Dict[str, Any]:
    """
    Build a dict of classification models.
    These are "template" models that will be cloned for each fold/rep combo.
    """
    models: Dict[str, Any] = {
        "logreg": LogisticRegression(
            max_iter=3000,                               # Enough iterations for convergence
            class_weight=cfg.class_weight,               
            solver="liblinear"                           # Good for small/medium sparse-ish problems
        ),
        "linearsvc": LinearSVC(
            class_weight=cfg.class_weight                # Linear SVM (no predict_proba by default)
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=3,                                 # Tree depth
            learning_rate=0.1,                            # Boosting learning rate
            max_iter=200,                                 # Number of boosting stages
            random_state=int(cfg.seed)                    # Reproducibility
        ),
    }

    # Print enabled models so you know exactly what's used
    print("[MODELS] Enabled:", sorted(models.keys()))
    return models


def featurize(rep_obj: Any, texts_train: List[str], texts_test: List[str]) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    NO-LEAKAGE FEATURE CREATION:
    - Clone the representation object (fresh state)
    - Fit ONLY on texts_train
    - Transform both train and test with that fitted transform

    Returns:
    - fitted: the fitted representation pipeline (useful if you want to reuse it)
    - X_tr: train features (dense np.ndarray float32)
    - X_te: test features  (dense np.ndarray float32)
    """
    fitted = clone(rep_obj)                              # New unfitted copy (important for CV safety)

    # Fit on training texts only
    X_tr = fitted.fit_transform(texts_train)

    # Transform test texts using the fitted representation (no fitting on test!)
    X_te = fitted.transform(texts_test)

    # Convert sparse matrices to dense arrays if needed (HGB typically wants dense)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()

    # Cast to float32 to reduce memory usage
    return fitted, X_tr.astype(np.float32), X_te.astype(np.float32)


# Final print to confirm cell executed without NameErrors
print("[OK] Data/Reps/Models/Featurize are defined.")
