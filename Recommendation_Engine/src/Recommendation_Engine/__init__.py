# -*- coding: utf-8 -*-  # Source file encoding declaration for safe Unicode handling.
# Ensures this module can include non-ASCII text (e.g., names, comments) safely across OS and editors.

# Expose key entry points for convenience
# This section re-exports commonly used symbols so that users can `import` directly from the package root
# instead of needing to navigate deep submodules. It improves developer ergonomics.

from .env_and_imports import PRESET, ART_DIR, EXPORT_TOP_N, RECIPIENT_NAME
# Imports configuration constants and preset environment logic from the `env_and_imports` module.
# - PRESET: indicates whether we're in FAST or ACCURATE mode.
# - ART_DIR: path for all experiment artifacts (e.g., ROC plots, CSV exports).
# - EXPORT_TOP_N: default number of top recommendations to export.
# - RECIPIENT_NAME: user label used in export filenames.

from .synthetic_corpus import build_big_corpus, Article
# Imports the synthetic data generator and the `Article` dataclass.
# - build_big_corpus(seed): builds a reproducible pseudo-article dataset for simulation.
# - Article: container for article metadata (id, title, text, category, etc.).

from .plots_and_metrics import plot_roc_single, plot_pr_curve, _tau_for_max_accuracy
# Imports key plotting and evaluation utilities:
# - plot_roc_single(): creates and optionally saves ROC curve and returns AUC.
# - plot_pr_curve(): draws a precision-recall curve with PR-AUC label.
# - _tau_for_max_accuracy(): computes optimal threshold τ that maximizes accuracy on validation data.

from .export_utils import _write_top_k_for_recipient
# Imports helper to write top-K recommendation outputs in both TXT and CSV formats.
# Used in main driver to produce human-readable lists and structured logs for each recipient/test split.
