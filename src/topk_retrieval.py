# Top-K retrieval (cosine * exp(-theta*dt)) 

from typing import List, Optional                         # Type hints for clarity (texts/history may be lists)

import numpy as np                                        # Vector math / argpartition
import pandas as pd                                       # Output DataFrame for recommendations

# Time handling

def get_doc_times(cfg: "Config", n_docs: int, times_from_data: Optional[np.ndarray]) -> np.ndarray:
    """
    Return a 1D float array of "times" for each document.
    - If dataset provides real timestamps (and length matches n_docs): use them.
    - Otherwise: use a synthetic time index [0,1,2,...] (acts like "older -> newer").
    """
    if times_from_data is not None:                        # If caller passed times from dataset
        t = np.asarray(times_from_data, dtype=np.float32)  # Force float32 array
        if len(t) == n_docs:                               # Validate length matches number of docs
            return t                                       # Use real times
    return np.arange(n_docs, dtype=np.float32)             # Fallback: synthetic time index

# User profile construction

def build_user_profile(X_norm: np.ndarray, history_idx: List[int]) -> np.ndarray:
    """
    Build user vector q:
    - Take mean of normalized vectors for docs in history
    - Normalize q to unit length (so dot product becomes cosine similarity)
    """
    if len(history_idx) == 0:                              # If user has no history
        return np.zeros((X_norm.shape[1],), dtype=np.float32)  # Return zero vector (no preference)

    idx = np.asarray(history_idx, dtype=int)               # Convert to numpy indices
    q = X_norm[idx].mean(axis=0).astype(np.float32)        # Average normalized vectors (centroid)

    n = float(np.linalg.norm(q))                           # Compute norm for re-normalization
    if n > 0:                                              # Avoid divide-by-zero
        q /= n                                             # Normalize q to unit length

    return q                                               # Return profile vector

# Time-decay weighting

def time_decay_weights(doc_times: np.ndarray, now_time: float, theta: float) -> np.ndarray:
    """
    Compute time weights:
      w_i = exp(-theta * dt_i)
      dt_i = max(now_time - doc_time_i, 0)
    Intuition:
      - Larger theta => stronger preference for newer docs
      - If doc is old (large dt) => smaller weight
    """
    dt = (now_time - doc_times).astype(np.float32)         # Delta time (newest - doc_time)
    dt = np.maximum(dt, 0.0)                               # Clip negatives to 0 (safety)
    return np.exp(-float(theta) * dt).astype(np.float32)   # Elementwise exp decay


# Main Top-K recommender

def top_k_time_aware(
    texts: List[str],                                     # Raw documents (strings)
    X_norm: np.ndarray,                                   # Normalized embedding matrix (n_docs x d)
    doc_times: np.ndarray,                                # Document times (n_docs,)
    history_idx: List[int],                               # Indices of "already read" docs
    top_k: int,                                           # How many recommendations
    theta: float,                                         # Time decay strength
) -> pd.DataFrame:
    """
    Recommend top_k documents not in history using:
      score_i = cosine(q, doc_i) * exp(-theta * dt_i)

    IMPORTANT:
    - X_norm must already be row-normalized
    - q is built from history docs and normalized
    => cosine similarity = dot product (X_norm @ q)
    """
    n = len(texts)                                        # Total number of docs

    #  Candidate mask: True = can recommend, False = in history 
    cand_mask = np.ones(n, dtype=bool)                    # Start with all docs eligible
    if history_idx:                                       # If there is any history
        cand_mask[np.asarray(history_idx, dtype=int)] = False  # Exclude already-read docs

    #  Build user profile vector q 
    q = build_user_profile(X_norm, history_idx)           # User centroid (unit vector)

    # Cosine similarity (dot product because vectors are normalized) 
    sims = (X_norm @ q).astype(np.float32)                # Similarity per doc (n_docs,)

    #  Time decay weights 
    now_time = float(np.max(doc_times)) if len(doc_times) else 0.0  # Define "now" as newest doc time
    w = time_decay_weights(doc_times, now_time=now_time, theta=float(theta))  # w_i per doc

    #  Final score (exclude history using very negative value) 
    score = np.where(cand_mask, sims * w, -1e9).astype(np.float32)  # Score_i, history gets -inf-ish

    #  Select Top-K efficiently without full sort 
    k = int(min(top_k, n))                               # Make sure k <= n
    idx = np.argpartition(-score, kth=max(0, k - 1))[:k] # Get indices of top-k (unordered)
    idx = idx[np.argsort(-score[idx])]                   # Sort those top-k by score descending

    #  Build a clean DataFrame for inspection / saving 
    return pd.DataFrame({
        "doc_idx": idx.astype(int),                       # Document index in original list
        "score": score[idx].astype(float),                # Final time-aware score
        "cosine": sims[idx].astype(float),                # Pure cosine similarity
        "time_weight": w[idx].astype(float),              # Decay multiplier
        "doc_time": doc_times[idx].astype(float),         # Document time (real or synthetic)
        "text": [texts[i] for i in idx],                  # Raw document text (for humans)
    })

# Confirmation print
print("[OK] Top-K retrieval functions are defined (get_doc_times, build_user_profile, time_decay_weights, top_k_time_aware).")
