# config.py 
# global switches

import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional


@dataclass
class Config:
    """
    Central configuration
    """


    dataset_name: str = "ag_news"        # HuggingFace dataset name
    text_field: str = "text"             # Column name that contains the text
    label_field: str = "label"           # Column name that contains labels
    max_docs: int = 40000                # Limit number of docs (for speed)
    seed: int = 42                       # Random seed for reproducibility


    make_binary: bool = True             # If True: convert multi-class y -> binary y
    pos_class: int = 0                   # Which original class becomes positive (=1)

    
    # CV / evaluation settings
    outer_folds: int = 2                 # Outer CV folds (test measurement)
    inner_val_size: float = 0.1          # Inner split size for threshold tuning

    # TF-IDF + LSA representation hyperparameters
    tfidf_max_features: int = 60000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    tfidf_stop_words: Optional[str] = "english"
    lsa_dim: int = 128

    # TF-IDF optional stemming/lemmatization switches (used by TextPreprocessor)
    # Rule: at most ONE of these should be True.
    tfidf_use_stemming: bool = False
    tfidf_use_lemmatization: bool = False

    # Word2Vec parameters
    w2v_dim: int = 50
    w2v_window: int = 5
    w2v_min_count: int = 2
    w2v_sg: int = 1
    w2v_negative: int = 10
    w2v_epochs: int = 2
    w2v_workers: int = 1
    w2v_max_train_docs: int = 8000
    w2v_max_tokens_per_doc: int = 2000

    # Models configuration

    class_weight: str = "balanced"


    # Overfitting detection rule (simple heuristic)

    enable_overfit_flag: bool = True
    overfit_acc_gap_thr: float = 0.1


    # Winner selection logic configuration

    primary_metric: str = "val_pr_auc"
    tie_metrics: Tuple[str, ...] = ("val_roc_auc", "val_f1", "val_mcc", "val_bal_acc", "val_acc")
    tie_eps: float = 1e-6

    # Top-K retrieval demo (recommender step)

    topk_k: int = 20
    topk_theta: float = 0.01
    topk_history_size: int = 10
    topk_use_real_time: bool = False


    # Output folders

    out_dir: str = "./artifacts"
    cache_dir: str = "./cache"


    # Random state used by sklearn objects that accept random_state

    random_state: int = 42



# Global switches
RUN_W2V = True
W2V_ONLY_FOR_MODEL = "logreg"   # allow W2V only for one model to save runtime



# Create config instance + sanity checks

cfg = Config()

# Sanity check: cannot enable both stemming and lemmatization
if cfg.tfidf_use_stemming and cfg.tfidf_use_lemmatization:
    raise ValueError("Config error: set at most ONE of tfidf_use_stemming or tfidf_use_lemmatization to True.")


print("[CONFIG] RUN_W2V:", RUN_W2V, "| W2V_ONLY_FOR_MODEL:", W2V_ONLY_FOR_MODEL)
print("[CONFIG] cfg:")
print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
