# -*- coding: utf-8 -*-  # Declares file encoding; ensures UTF-8 text is read/written correctly.

from typing import Optional  # Enables optional type hints.
import numpy as np  # Numerical library used for arrays and math.
import matplotlib.pyplot as plt  # Main plotting interface.
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # Metrics for ROC and PR curves.
from .env_and_imports import (  # Import global constants for figure sizes and defaults.
    FIGSIZE_STD, FIGSIZE_CM, CMAP_HEATMAP,
    SECONDS_PER_DAY, RECENCY_TAU_DAYS_DEFAULT
)

# -------------------- Utils & plotting --------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Logistic activation; maps any real number to (0,1) range.
    return 1.0 / (1.0 + np.exp(-x))

def recency_decay(published_ts: Optional[float], now: float, tau_days: float = RECENCY_TAU_DAYS_DEFAULT) -> float:
    # Computes exponential decay to downweight old items by publication age.
    if published_ts is None:  # If timestamp is missing, treat as neutral (no decay).
        return 1.0
    delta_days = max(0.0, (now - published_ts) / SECONDS_PER_DAY)  # Convert seconds difference → days.
    return float(np.exp(-delta_days / max(tau_days, 1e-6)))  # Apply exponential decay with safe denominator.

def _save_and_show(path: Optional[str]):
    # Saves current Matplotlib figure if path provided, then closes it to free memory.
    try:
        if path is not None:
            plt.savefig(path, dpi=150, bbox_inches="tight")  # Save with reasonable resolution and compact layout.
    except Exception as e:
        print(f"[PLOT SAVE WARNING] {e}")  # Gracefully warn if saving fails.
    finally:
        plt.close()  # Always close figure to avoid memory leaks.

def plot_roc_single(y_true: np.ndarray, scores: np.ndarray, title: str, save_to: Optional[str] = None):
    # Draws a ROC curve (TPR vs FPR) and returns the AUC value.
    fpr, tpr, _ = roc_curve(y_true, scores)  # Compute false/true positive rates for all thresholds.
    roc_auc = auc(fpr, tpr)  # Integrate area under ROC curve.

    plt.figure(figsize=FIGSIZE_STD)  # Use standard figure size.
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")  # Plot curve with AUC label.
    plt.plot([0, 1], [0, 1], '--')  # Add diagonal baseline (random guess).
    plt.xlabel("FPR")  # Label X axis.
    plt.ylabel("TPR")  # Label Y axis.
    plt.title(title)  # Add plot title.
    plt.legend()  # Show legend (AUC value).
    plt.grid(True)  # Add background grid for readability.
    _save_and_show(save_to)  # Save and close figure if path given.
    return float(roc_auc)  # Return numeric AUC for logging.

def plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, title: str, save_to: Optional[str] = None):
    # Draws precision-recall curve and returns area under PR curve.
    precision, recall, _ = precision_recall_curve(y_true, scores)  # Compute precision/recall for thresholds.
    from sklearn.metrics import auc as sk_auc  # Alias for AUC integration.
    pr_auc = float(sk_auc(recall, precision))  # Compute PR-AUC numeric value.

    plt.figure(figsize=FIGSIZE_STD)  # Create figure.
    plt.plot(recall, precision)  # Plot recall (x) vs precision (y).
    plt.xlabel("Recall")  # X-axis label.
    plt.ylabel("Precision")  # Y-axis label.
    plt.title(title + f" (PR-AUC={pr_auc:.3f})")  # Add title and score.
    plt.grid(True)  # Add grid.
    _save_and_show(save_to)  # Save and close.
    return pr_auc  # Return numeric PR-AUC.

def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_scores: np.ndarray, tau: float,
                                  normalize: bool = False, title: str = "Confusion",
                                  cmap: Optional[str] = None, save_to: Optional[str] = None):
    # Creates a 2×2 confusion matrix heatmap for given threshold tau.
    y_pred = (y_scores >= tau).astype(int)  # Convert scores to binary predictions using threshold.
    TP = int(((y_pred == 1) & (y_true == 1)).sum())  # True positives count.
    FN = int(((y_pred == 0) & (y_true == 1)).sum())  # False negatives count.
    FP = int(((y_pred == 1) & (y_true == 0)).sum())  # False positives count.
    TN = int(((y_pred == 0) & (y_true == 0)).sum())  # True negatives count.

    cm = np.array([[TP, FN], [FP, TN]], dtype=float)  # Arrange counts in 2×2 matrix.
    disp = cm.copy()  # Copy for normalization step.
    if normalize:
        row_sums = disp.sum(axis=1, keepdims=True)  # Sum each actual-class row.
        disp = np.divide(disp, np.maximum(row_sums, 1e-9))  # Normalize each row safely.

    plt.figure(figsize=FIGSIZE_CM)  # Smaller figure for confusion matrix.
    im = plt.imshow(disp, cmap=(cmap or CMAP_HEATMAP))  # Draw heatmap using chosen colormap.

    try:
        plt.colorbar(im, fraction=0.046, pad=0.04)  # Add color scale bar.
    except Exception:
        pass  # Skip if backend doesn’t support colorbar.

    plt.xticks([0, 1], ["Pred +", "Pred -"])  # Label X ticks for predicted classes.
    plt.yticks([0, 1], ["Actual +", "Actual -"])  # Label Y ticks for true classes.
    plt.title(title)  # Add title.

    # Annotate each cell with its numeric value.
    for i in range(2):
        for j in range(2):
            val = disp[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"  # Show % or raw count.
            plt.text(j, i, txt, ha="center", va="center", color="black",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()  # Adjust layout to prevent clipping.
    _save_and_show(save_to)  # Save and close figure.

def _tau_for_max_accuracy(y: np.ndarray, s: np.ndarray) -> float:
    # Finds threshold τ maximizing accuracy on validation set.
    uniq = np.unique(s)  # Get all unique score values.
    if len(uniq) == 1:  # If all scores identical, threshold choice irrelevant.
        return float(uniq[0])

    best_acc, best_tau = -1.0, float(uniq[0])  # Initialize trackers.
    for t in uniq:
        y_pred = (s >= t).astype(int)  # Convert scores to binary predictions.
        acc = float((y_pred == y).mean())  # Compute accuracy for this threshold.
        if acc > best_acc:  # Keep if accuracy improved.
            best_acc, best_tau = acc, float(t)
    return best_tau  # Return threshold achieving highest accuracy.
