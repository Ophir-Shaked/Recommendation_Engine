# global switches

import json
from dataclasses import dataclass, asdict  # dataclass for clean configuration objects
from typing import Tuple, Optional   # typing helpers for clearer type hints


@dataclass
class Config:
    """
    Central configuration object for the whole notebook.
    We keep ALL hyperparameters and switches here so it is easy to reproduce runs.
    """

    # Dataset parameters
    dataset_name: str = "ag_news"    # HuggingFace dataset name (e.g., "ag_news")
    text_field: str = "text"         # Column name that contains the text
    label_field: str = "label"       # Column name that contains labels (multi-class labels in AG News)
    max_docs: int = 40000            # Limit number of docs (for speed)
    seed: int = 42                   # Random seed for reproducibility

    # Binary target settings (optional)
    make_binary: bool = True         # If True: convert multi-class y -> binary y
    pos_class: int = 0               # Which original class becomes positive (=1) after binarization

    # CV / evaluation settings
    outer_folds: int = 2             # Outer CV folds (test measurement)
    inner_val_size: float = 0.1      # Inner split size for threshold tuning (validation fraction)

    # TF-IDF + LSA representation hyperparameters

    tfidf_max_features: int = 60000  # TF-IDF vocab cap (large but ok)
    tfidf_ngram_range: Tuple[int, int] = (1, 2)  # Use unigrams + bigrams
    tfidf_min_df: int = 2            # Ignore terms appearing in <2 documents (remove noise)
    tfidf_stop_words: Optional[str] = "english"  # Remove English stopwords
    lsa_dim: int = 128               # TruncatedSVD dimension (LSA embedding size)

    # Word2Vec parameters (small-ish for runtime stability)
    w2v_dim: int = 50                # Word2Vec embedding dimension
    w2v_window: int = 5              # Context window size
    w2v_min_count: int = 2           # Ignore very rare words
    w2v_sg: int = 1                  # 1 = skip-gram, 0 = CBOW
    w2v_negative: int = 10           # Negative sampling count
    w2v_epochs: int = 2              # Training epochs (low for speed)
    w2v_workers: int = 1             # Threads/workers (keep 1 in Colab to avoid instability)
    w2v_max_train_docs: int = 8000   # Train W2V only on subset of docs (speed)
    w2v_max_tokens_per_doc: int = 2000  # Token cap per document (speed & safety)

    # Models configuration
    class_weight: str = "balanced"   # Handle class imbalance by reweighting

    # Overfitting detection rule (simple heuristic)
    enable_overfit_flag: bool = True     # Enable/disable overfitting flag
    overfit_acc_gap_thr: float = 0.1     # Flag model if train_acc - val_acc > 0.1

    # Winner selection logic configuration
    primary_metric: str = "val_pr_auc"   # Winner chosen primarily by validation PR-AUC
    tie_metrics: Tuple[str, ...] = ("val_roc_auc", "val_f1", "val_mcc", "val_bal_acc", "val_acc")  # tie-break metrics
    tie_eps: float = 1e-6               # epsilon tolerance when comparing floating metrics

    # Top-K retrieval demo (recommender step)
    topk_k: int = 20                 # Recommend top-K documents
    topk_theta: float = 0.01         # Time decay coefficient
    topk_history_size: int = 10      # How many docs the user "read" (=history)
    topk_use_real_time: bool = False # Most HF datasets have no timestamps -> use fake time (=index)

    # Output folders (saved files)
    out_dir: str = "./artifacts"     # Where all CSV/plots/JSON outputs go
    cache_dir: str = "./cache"       # Optional: place to cache intermediate artifacts

    random_state: int = 42           # Used for scikit objects that want random_state


# Global switches (not in dataclass)

RUN_W2V = True                       # If False, skip Word2Vec rep completely (faster)
W2V_ONLY_FOR_MODEL = "logreg"        # Only allow W2V representation for ONE model (saves time)

# Create config instance (uses all defaults above)
cfg = Config()

# Print configuration so the run is documented in notebook output
print("[CONFIG] RUN_W2V:", RUN_W2V, "| W2V_ONLY_FOR_MODEL:", W2V_ONLY_FOR_MODEL)
print("[CONFIG] cfg:")
print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))  # asdict() converts dataclass -> dict
