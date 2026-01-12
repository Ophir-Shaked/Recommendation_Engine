#  Metrics + Threshold + Plots + Winner logic

import os                                                     # Path building for saved artifacts
from typing import Dict, Tuple, List, Any                      # Type hints (readability + safety)

import numpy as np                                             # Numeric arrays
import pandas as pd                                            # DataFrames for saving error analysis + ECE bins
from scipy.special import expit                                # Sigmoid function (turn margins into probs)

#  Sklearn metrics 
from sklearn.metrics import (
    roc_auc_score,                                             # ROC-AUC metric
    average_precision_score,                                   # PR-AUC (aka average precision)
    accuracy_score,                                            # Accuracy
    f1_score,                                                  # F1 score
    precision_score,                                           # Precision
    recall_score,                                              # Recall
    matthews_corrcoef,                                         # MCC (good for imbalance)
    balanced_accuracy_score,                                   # Balanced accuracy
    roc_curve,                                                 # For ROC plot
    precision_recall_curve                                     # For PR plot
)

#  Calibration 
from sklearn.calibration import calibration_curve              # Reliability curve helper

#  Plotting 
import matplotlib.pyplot as plt                                # Save PNG plots


# Metric helpers (safe functions)

def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Compute ROC-AUC safely (if there is only one class -> ROC-AUC fails).
    Returns NaN in edge cases.
    """
    try:
        return float(roc_auc_score(y, s))                       # Standard ROC-AUC
    except Exception:
        return float("nan")                                     # Safe fallback

def safe_pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Compute PR-AUC safely (Average Precision).
    Returns NaN in edge cases.
    """
    try:
        return float(average_precision_score(y, s))             # Average precision
    except Exception:
        return float("nan")                                     # Safe fallback



# Model score extraction (make all models output "probability-like")

def model_scores(estimator: Any, X: Any) -> np.ndarray:
    """
    Convert model outputs into score vector in [0,1].
    - If model supports predict_proba: use P(class=1)
    - If model supports decision_function: apply sigmoid to convert margins to pseudo-probabilities
    - Else fallback to predict (not ideal, but safe)
    """
    if hasattr(estimator, "predict_proba"):                     # Logistic regression etc.
        return estimator.predict_proba(X)[:, 1].astype(float)   # Probability of positive class

    if hasattr(estimator, "decision_function"):                 # SVM etc.
        margins = estimator.decision_function(X).astype(float)  # Raw decision margin
        return expit(margins)                                   # Convert to 0..1 via sigmoid

    return estimator.predict(X).astype(float)                   # Last resort



# Threshold selection

def choose_threshold_best_f1(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """
    Pick a classification threshold that maximizes F1 on validation data.
    - We search across candidate thresholds derived from scores.
    """
    uniq = np.unique(s_val)                                     # Unique score values
    if len(uniq) < 2:                                           # Not enough variation
        return 0.5                                              

    # If too many unique values, use quantiles for speed
    thr_list = uniq if len(uniq) <= 800 else np.unique(
        np.quantile(s_val, np.linspace(0, 1, 801))               # 801 thresholds from 0%..100%
    )

    best_thr = 0.5                                              # Initialize best threshold
    best_f1v = -1.0                                             # Initialize best F1

    for thr in thr_list:                                        # Try each threshold
        yhat = (s_val >= thr).astype(int)                       # Turn scores into predictions
        f1v = f1_score(y_val, yhat, zero_division=0)             # Compute F1 at this threshold
        if f1v > best_f1v:                                      # Update best if improved
            best_f1v = float(f1v)
            best_thr = float(thr)

    return best_thr                                             # Return best threshold found



# Core metric computation

def compute_metrics(y_true: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Given true labels + predicted scores + threshold, compute metrics dict.
    """
    yhat = (s >= thr).astype(int)                               # Binary predictions (0/1)

    out = {                                                     # Collect metrics into a dict
        "roc_auc": safe_roc_auc(y_true, s),                     # ROC-AUC (threshold-free)
        "pr_auc":  safe_pr_auc(y_true, s),                      # PR-AUC (threshold-free)
        "acc":     float(accuracy_score(y_true, yhat)),         # Accuracy
        "f1":      float(f1_score(y_true, yhat, zero_division=0)),  # F1 score
        "precision": float(precision_score(y_true, yhat, zero_division=0)),  # Precision
        "recall":    float(recall_score(y_true, yhat, zero_division=0)),     # Recall
    }

    # MCC is sometimes undefined; handle safely
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, yhat))     # Matthews correlation coefficient
    except Exception:
        out["mcc"] = float("nan")

    # Balanced accuracy sometimes fails if only one class exists
    try:
        out["bal_acc"] = float(balanced_accuracy_score(y_true, yhat))  # Balanced accuracy
    except Exception:
        out["bal_acc"] = float("nan")

    return out                                                  # Return dict



# Calibration error (ECE)

def expected_calibration_error(
    y: np.ndarray,                                              # True binary labels
    p: np.ndarray,                                              # Predicted probabilities (0..1)
    n_bins: int = 15                                            # Number of bins
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Expected Calibration Error (ECE).
    Returns:
      - ece scalar
      - per-bin DataFrame (counts, mean p, mean y, gap)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)                    # Bin edges
    ids = np.digitize(p, bins) - 1                              # Which bin each sample falls into

    ece = 0.0                                                   # Accumulate ECE here
    rows = []                                                   # Bin table rows

    for b in range(n_bins):                                     # Iterate over each bin
        mask = (ids == b)                                       # Samples in bin b
        if not np.any(mask):                                    # Empty bin
            rows.append({
                "bin": b,
                "count": 0,
                "p_mean": np.nan,
                "y_rate": np.nan,
                "abs_gap": np.nan
            })
            continue

        p_mean = float(np.mean(p[mask]))                        # Mean predicted probability in bin
        y_rate = float(np.mean(y[mask]))                        # Empirical positive rate in bin
        gap = abs(y_rate - p_mean)                              # Calibration gap

        ece += float(np.mean(mask)) * gap                       # Weighted contribution to ECE
        rows.append({                                           # Store per-bin info
            "bin": b,
            "count": int(np.sum(mask)),
            "p_mean": p_mean,
            "y_rate": y_rate,
            "abs_gap": gap
        })

    bins_df = pd.DataFrame(rows)                                # Convert rows -> DataFrame
    return float(ece), bins_df                                  # Return ECE + table



# Save ROC / PR / calibration curves

def save_roc_pr_calibration(out_dir: str, y: np.ndarray, s: np.ndarray, prefix: str) -> Dict[str, str]:
    """
    Save:
      - ROC curve PNG
      - PR curve PNG
      - calibration curve PNG
      - calibration bins CSV (ECE table)
    Returns dict of saved paths.
    """
    ensure_dir(out_dir)                                         # Ensure output folder exists 

    paths: Dict[str, str] = {}                                  # Collect saved file paths

    #  ROC curve 
    fpr, tpr, _ = roc_curve(y, s)                               # Compute curve points
    plt.figure()                                                # New figure
    plt.plot(fpr, tpr)                                          # Plot
    plt.xlabel("False Positive Rate")                           # X label
    plt.ylabel("True Positive Rate")                            # Y label
    plt.title("ROC Curve (OOF)")                                # Title
    p = os.path.join(out_dir, f"{prefix}_roc.png")              # Output file path
    plt.savefig(p, bbox_inches="tight", dpi=160)                # Save to disk
    plt.close()                                                 # Close to avoid memory buildup
    paths["roc"] = p                                            # Store path

    #  PR curve 
    prec, rec, _ = precision_recall_curve(y, s)                 # Compute PR curve points
    plt.figure()                                                # New figure
    plt.plot(rec, prec)                                         # Plot PR curve
    plt.xlabel("Recall")                                        # X label
    plt.ylabel("Precision")                                     # Y label
    plt.title("PR Curve (OOF)")                                 # Title
    p = os.path.join(out_dir, f"{prefix}_pr.png")               # Output file path
    plt.savefig(p, bbox_inches="tight", dpi=160)                # Save
    plt.close()                                                 # Close
    paths["pr"] = p                                             # Store

    # Calibration curve + ECE 
    prob_true, prob_pred = calibration_curve(                   # Calibration curve points
        y, s, n_bins=15, strategy="uniform"
    )

    ece, bins_df = expected_calibration_error(y, s, n_bins=15)   # Compute ECE + per-bin table

    plt.figure()                                                # New figure
    plt.plot(prob_pred, prob_true)                              # Model calibration line
    plt.plot([0, 1], [0, 1])                                    # Perfect calibration baseline
    plt.xlabel("Mean predicted probability")                    # X label
    plt.ylabel("Fraction of positives")                         # Y label
    plt.title(f"Calibration (OOF) | ECE={ece:.4f}")              # Title with ECE value
    p = os.path.join(out_dir, f"{prefix}_calibration.png")       # Output file path
    plt.savefig(p, bbox_inches="tight", dpi=160)                # Save
    plt.close()                                                 # Close
    paths["calibration"] = p                                    # Store path

    # Save ECE bin table
    p = os.path.join(out_dir, f"{prefix}_calibration_bins.csv")  # CSV path
    bins_df.to_csv(p, index=False)                               # Save CSV
    paths["calibration_bins"] = p                                # Store path

    return paths                                                # Return all saved paths

# Error analysis saving (FP/FN/Uncertain)

def save_error_analysis(
    out_dir: str,                                               # Folder for outputs
    texts: List[str],                                           # Raw texts for interpretability
    y: np.ndarray,                                              # True labels
    s: np.ndarray,                                              # Predicted scores
    thr: float,                                                 # Threshold for decisions
    prefix: str,                                                # Output prefix
    top_k: int = 200                                            # How many examples to save
) -> Dict[str, str]:
    """
    Save:
      - false positives CSV
      - false negatives CSV
      - most uncertain CSV (closest to threshold)
    """
    ensure_dir(out_dir)                                         # Ensure directory exists

    yhat = (s >= thr).astype(int)                               # Predictions
    df = pd.DataFrame({                                         # Build analysis DataFrame
        "text": texts,
        "y_true": y.astype(int),
        "score": s.astype(float),
        "y_pred": yhat.astype(int),
        "margin_to_thr": (s - thr).astype(float),               # Positive=above threshold, negative=below
    })

    # FP: predicted 1 but true 0 (sorted by highest wrong confidence)
    fp = df[(df.y_true == 0) & (df.y_pred == 1)].copy()         # Filter FP
    fp = fp.sort_values("score", ascending=False).head(top_k)   # Highest score FPs

    # FN: predicted 0 but true 1 (sorted by lowest score / most wrong)
    fn = df[(df.y_true == 1) & (df.y_pred == 0)].copy()         # Filter FN
    fn = fn.sort_values("score", ascending=True).head(top_k)    # Lowest score FNs

    # Most uncertain: score closest to thr (small abs margin)
    un = df.iloc[(df.margin_to_thr.abs().sort_values().index)]  # Sort by abs distance to threshold
    un = un.head(top_k)                                        # Take top_k closest points

    # Output paths
    p_fp = os.path.join(out_dir, f"{prefix}_false_positives.csv")   # FP CSV path
    p_fn = os.path.join(out_dir, f"{prefix}_false_negatives.csv")   # FN CSV path
    p_un = os.path.join(out_dir, f"{prefix}_most_uncertain.csv")    # uncertain CSV path

    # Save files
    fp.to_csv(p_fp, index=False)                                # Save FP
    fn.to_csv(p_fn, index=False)                                # Save FN
    un.to_csv(p_un, index=False)                                # Save uncertain

    return {"fp": p_fp, "fn": p_fn, "uncertain": p_un}          # Return all paths



# Winner selection logic


def better(a: float, b: float, eps: float) -> bool:
    """Return True if a is meaningfully larger than b (by eps)."""
    return (a is not None) and (b is not None) and (a > b + eps)

def pick_winner_row(summary: pd.DataFrame, cfg: "Config") -> pd.Series:
    """
    Select winner from df_summary.
    Primary metric: val_pr_auc_mean
    Tie-breakers: cfg.tie_metrics in order.
    """
    cols = {                                                    # Mapping to actual df_summary columns
        "val_pr_auc": "val_pr_auc_mean",
        "val_roc_auc": "val_roc_auc_mean",
        "val_f1": "val_f1_mean",
        "val_mcc": "val_mcc_mean",
        "val_bal_acc": "val_bal_acc_mean",
        "val_acc": "val_acc_mean",
    }

    best = None                                                 # Will hold the best row
    best_idx = None                                             # Best row index

    for i in range(len(summary)):                               # Iterate rows
        row = summary.iloc[i]                                   # Current candidate row

        if best is None:                                        # First row becomes best initially
            best = row
            best_idx = i
            continue

        # Compare primary metric: PR-AUC
        a = float(row[cols["val_pr_auc"]])                      # Candidate PR-AUC mean
        b = float(best[cols["val_pr_auc"]])                     # Best-so-far PR-AUC mean

        if better(a, b, cfg.tie_eps):                           # Candidate better in primary metric
            best = row
            best_idx = i
            continue

        # If tie in primary metric (within eps), apply tie-breakers
        if abs(a - b) <= cfg.tie_eps:
            for m in cfg.tie_metrics:                           # Tie metrics in priority order
                av = float(row[cols[m]])                        # Candidate metric
                bv = float(best[cols[m]])                       # Best metric

                if better(av, bv, cfg.tie_eps):                 # Candidate wins this tie-breaker
                    best = row
                    best_idx = i
                    break

                if better(bv, av, cfg.tie_eps):                 # Best remains better
                    break

    return summary.iloc[int(best_idx)]                          # Return best row as Series

# Final print so we can confirm the cell executed successfully
print("[OK] Metrics/threshold/plots/winner logic are defined.")
