# -*- coding: utf-8 -*-  # Encoding declaration for safe Unicode compatibility across systems.

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .env_and_imports import HNM_MAX_ITER, HNM_BASE_LR_C  # Constants controlling LR iterations and regularization.

# -------------------- Hard-negative mining --------------------
def mine_hard_negatives(X_tr: np.ndarray,
                        y_tr: np.ndarray,
                        keep_pos_frac: float = 1.0,
                        hard_neg_multiplier: float = 6.0,
                        seed: int = 42,
                        sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Performs hard-negative mining to rebalance the training set by selecting the most confusing negatives.

    Parameters
    ----------
    X_tr : np.ndarray
        Training feature matrix.
    y_tr : np.ndarray
        Binary target vector (1=positive, 0=negative).
    keep_pos_frac : float, optional
        Fraction of positives to retain (default 1.0 keeps all positives).
    hard_neg_multiplier : float, optional
        Multiplier determining how many hard negatives to include per positive.
    seed : int, optional
        Random seed for reproducibility.
    sample_weight : np.ndarray, optional
        Optional sample weights to carry through to the new subset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        Subsampled features, labels, and optional weights after mining.
    """

    rng = np.random.RandomState(seed)  # Deterministic RNG for reproducibility.

    # Normalize features for linear separability and stable gradient descent.
    scaler = StandardScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)

    # Train a lightweight logistic regression classifier as a scoring base.
    base = LogisticRegression(
        max_iter=HNM_MAX_ITER,
        solver="liblinear",
        class_weight="balanced",
        C=HNM_BASE_LR_C
    )
    base.fit(Xs, y_tr, sample_weight=sample_weight)

    # Predict probabilities for all training samples (score = P(y=1)).
    p = base.predict_proba(Xs)[:, 1]

    # Split indices for positives and negatives.
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]

    # Optionally downsample positives if keep_pos_frac < 1.0
    if keep_pos_frac < 1.0:
        kpos = max(1, int(keep_pos_frac * len(pos_idx)))  # Ensure at least one positive remains.
        pos_keep = rng.choice(pos_idx, size=kpos, replace=False)
    else:
        pos_keep = pos_idx

    # Select a proportional number of negatives — focus on those with high predicted positive probability.
    k_neg = max(len(pos_keep), int(hard_neg_multiplier * len(pos_keep)))

    # If dataset small, take all negatives; else, choose the hardest ones (highest p).
    if k_neg >= len(neg_idx):
        hard_neg = neg_idx
    else:
        order = np.argsort(-p[neg_idx])  # Sort descending by model confidence.
        hard_neg = neg_idx[order[:k_neg]]

    # Combine retained positives and selected hard negatives, ensuring uniqueness.
    keep = np.unique(np.concatenate([pos_keep, hard_neg]))

    # Return filtered training set (and weights if provided).
    if sample_weight is None:
        return X_tr[keep], y_tr[keep], None
    else:
        return X_tr[keep], y_tr[keep], sample_weight[keep]
