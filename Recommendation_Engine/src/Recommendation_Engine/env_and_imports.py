# -*- coding: utf-8 -*-  # Declares UTF-8 encoding so the file can safely contain any Unicode characters.

from __future__ import annotations  # Enables postponed evaluation of type hints (Python 3.7+).

# ---- Recommendation export knobs ----
TOP_K_RECS = 5  # How many items to include in the exported Top-N recommendation list.
RECIPIENT_NAME = "Hila Ronen"  # Display name for personalized output files (used in filenames and headers).

import os, csv, json, math, time, random, warnings  # Standard Python modules used for I/O, math, and runtime control.
from dataclasses import dataclass  # Lightweight container class for structured data records.
from typing import Dict, List, Optional, Tuple  # Type hints to improve readability and static analysis.

# keep BLAS sane on small machines
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Caps BLAS threads to 1 to avoid CPU overuse on limited systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")       # Caps OpenMP threads for same reason — improves stability.

# plotting (headless-safe)
import matplotlib  # Main plotting library (Matplotlib) imported for figure generation.
matplotlib.use("Agg")  # Forces non-interactive backend (Agg) for headless environments (Colab/CI servers).
import matplotlib.pyplot as plt  # Submodule for easy plotting commands (plt interface).

import numpy as np  # Numerical arrays and math operations.
import pandas as pd  # Tabular data management and I/O.

from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into TF-IDF weighted term matrices.
from sklearn.decomposition import TruncatedSVD  # Used for Latent Semantic Analysis (LSA) dimensionality reduction.
from sklearn.preprocessing import Normalizer, StandardScaler  # Normalization and feature scaling utilities.
from sklearn.pipeline import make_pipeline  # Builds linear transformation pipelines (e.g., SVD → Normalizer).
from sklearn.metrics.pairwise import cosine_similarity  # Computes cosine similarity for recommendation scoring.
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score  # Common evaluation metrics.
from sklearn.linear_model import LogisticRegression  # Linear model used for base reranking or meta-level classifier.
from sklearn.calibration import CalibratedClassifierCV  # Wraps classifiers for calibrated probability outputs.
from sklearn.model_selection import StratifiedKFold  # Cross-validation split that preserves class balance.
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier  # Tree-based learners.
from sklearn.svm import LinearSVC  # Optional linear Support Vector Classifier model.

"""
Recommendation Engine v5.5
=======================

Purpose:
--------
Implements a transparent recommendation pipeline designed for reproducibility and integrity.  
The system generates a synthetic corpus, simulates user interactions, and evaluates ranking  
models under a strictly leak-free protocol.

Overview:
---------
• Full separation between TRAIN, VALIDATION, and TEST.
• No information leakage across splits.
• Threshold τ chosen on validation, applied once on test.
• Compatible with Colab and low-memory environments.

Presets (set before running):
  %env RECO_PRESET=FAST
  %env RECO_PRESET=ACCURATE   # recommended
"""

import argparse  # Used for CLI preset flag parsing.

# -------------------- preset resolution (works in scripts & notebooks) --------------------
def _resolve_preset() -> str:
    parser = argparse.ArgumentParser(add_help=False)  # Minimal parser that won’t conflict with notebooks.
    parser.add_argument("--preset", choices=["FAST", "ACCURATE"], default=None)  # Optional runtime preset flag.
    try:
        args, _ = parser.parse_known_args()  # Reads known args but ignores unknown notebook flags.
    except SystemExit:
        args = argparse.Namespace(preset=None)  # Prevents parser exit behavior inside notebooks.
    val = (args.preset or os.environ.get("RECO_PRESET", "FAST")).upper()  # Chooses CLI → ENV → default order.
    return "ACCURATE" if val == "ACCURATE" else "FAST"  # Ensures valid normalized preset output.

PRESET = _resolve_preset()  # Determines active preset at runtime.
print(f"[PRESET_MODE] {PRESET}")  # Prints which preset is active (FAST/ACCURATE).

# -------------------- Global constants (no magic numbers) --------------------
# Time
SECONDS_PER_DAY = 86_400
RECENCY_TAU_DAYS_DEFAULT = 20.0
REC_FAST_HALFLIFE_DAYS = 7.0
ROUND_STEP_FRACTION_OF_DAY = 1.0 / 6.0
RANK_STEP_SECONDS = 45.0

# Title/text length normalizations
TITLE_LEN_NORM_DIV_GENERATOR = 60.0
TITLE_LEN_NORM_DIV_FEATURES = 40.0
TEXT_LEN_NORM_DIV_FEATURES = 400.0

# Generator (clicks-only) signal
CLICK_ALPHA_DEFAULT = 75.0
CLICK_BETA_POS_DEFAULT = 0.10
CLICK_NOISE_DEFAULT = 0.005
POSITION_BIAS_NOISE = 0.12

# Generator TF-IDF / simulation constants
GEN_TF_MIN_DF = 3
GEN_TF_MAX_DF = 0.95
GEN_TF_TITLE_MAXFEAT = 20_000
GEN_TF_GLOBAL_MAXFEAT = 60_000
TITLE_MASS_DIVISOR = 100.0
BASE_POP_REC_WEIGHT = 0.5
BASE_POP_TITLE_WEIGHT = 0.5
GEN_HISTORY_BACK_DAYS = 220

# W2V training knobs
W2V_WINDOW = 6
W2V_MIN_COUNT = 2
W2V_NEGATIVE = 10
W2V_EPOCHS = 6
W2V_SG = 1

# Profile/history defaults
PROFILE_K_PER_ITEM = 8

# Hard-negative mining
HNM_BASE_LR_C = 0.35
HNM_MAX_ITER = 2000

# Plotting
FIGSIZE_STD = (6, 4)
FIGSIZE_CM = (5.5, 4.5)
CMAP_HEATMAP = "YlGnBu"

# Exports
EXPORT_TOP_N = 10
VAL_SUMMARY_FILE = "val_summary.json"
INTERACTIONS_FILE = "interactions_all.csv"

# Co-visibility
COVIS_TOPK = 150
COVIS_TIME_BUCKET_SECONDS = 60

# Position debias
POSITION_DEBIAS_ALPHA = 0.60
POSITION_DEBIAS_MIN_WEIGHT = 1e-3

# Safety / feature toggles
SAFE_DISABLE_CHAR_N_THRESHOLD = 35_000

# RNG seeds
GLOBAL_RS_SVD = 42
GLOBAL_RS_HGB_BASE = 42
GLOBAL_RS_ET_BASE = 99
BAG_SEED_MULTIPLIER = 17
META_RS_HGB = 777

# Meta / base model knobs
HGB_MAX_DEPTH = 8
HGB_LR = 0.05
HGB_VALID_FRAC = 0.08
HGB_EARLYSTOP_N = 80
HGB_L2 = 1e-3

ET_MIN_SAMPLES_SPLIT = 2
ET_MIN_SAMPLES_LEAF = 2
ET_BOOTSTRAP = False

SVC_C = 0.8
SVC_MAX_ITER = 6000

META_LR_C = 2.0
META_LR_MAX_ITER = 4000
META_HGB_MAX_DEPTH = 3
META_HGB_LR = 0.08
META_HGB_MAX_ITER = 400
META_HGB_VALID_FRAC = 0.10
META_HGB_EARLYSTOP_N = 40

# Deployment
DEPLOY_MODE_MAX_ACCURACY = "max_accuracy"

# -------------------- artifacts --------------------
ART_DIR = "reco_engine_artifacts_v5_5"
os.makedirs(ART_DIR, exist_ok=True)

# -------------------- Optional gensim Word2Vec; fallback to pseudo2vec --------------------
try:
    from gensim.models import Word2Vec  # noqa: F401
    GENSIM_AVAILABLE = True
except Exception:
    GENSIM_AVAILABLE = False

# -------------------- Defaults (overridden by preset) --------------------
N_JOBS = max(1, min(4, (os.cpu_count() or 2)))
USE_CHAR_FEATS = True
USE_CAL_SVC = False

if PRESET == "FAST":
    TF_TITLE_MAXFEAT = 24_000
    TF_BODY_MAXFEAT  = 48_000
    TF_MIN_DF = 7
    TF_MAX_DF = 0.88
    NGRAM_TITLE = (1, 2)
    NGRAM_BODY  = (1, 2)

    USE_CHAR_FEATS = True
    CHAR_TITLE_MAXFEAT = 18_000
    CHAR_BODY_MAXFEAT  = 36_000
    CHAR_NGRAM_RANGE = (3, 5)
    CHAR_MIN_DF = 6
    CHAR_MAX_DF = 0.90

    LSA_DIM = 256
    W2V_DIM = 64

    N_IMPRESSIONS = 120_000
    SLATE_SIZE = 18

    K_OUT_FOLDS = 5
    HGB_ITERS = 1000
    ET_TREES = 900

    HARD_NEG_MULT = 8.0
    ADD_TRAIN_NOISE = 0.008

    N_JOBS = 1
else:
    TF_TITLE_MAXFEAT = 40_000
    TF_BODY_MAXFEAT  = 80_000
    TF_MIN_DF = 4
    TF_MAX_DF = 0.93
    NGRAM_TITLE = (1, 2)
    NGRAM_BODY  = (1, 2)

    USE_CHAR_FEATS = False

    LSA_DIM = 320
    W2V_DIM = 64

    N_IMPRESSIONS = 150_000
    SLATE_SIZE    = 24

    K_OUT_FOLDS = 5
    HGB_ITERS   = 1400
    ET_TREES    = 1200

    USE_CAL_SVC = False

    HARD_NEG_MULT   = 10.0
    ADD_TRAIN_NOISE = 0.010

    N_JOBS = 1

# -------------------- Generator signal (affects clicks only) --------------------
CLICK_ALPHA    = CLICK_ALPHA_DEFAULT
CLICK_BETA_POS = CLICK_BETA_POS_DEFAULT
CLICK_NOISE    = CLICK_NOISE_DEFAULT
POSITION_BIAS  = POSITION_BIAS_NOISE

# -------------------- Stacking / training --------------------
RERANKER_C   = 0.25
N_BAGS       = 6
BAG_SUBSAMPLE= 0.85

# τ selection & deployment
DEPLOY_MODE = DEPLOY_MODE_MAX_ACCURACY

# Seeds & histories
SEEDS = [3, 5]
USER_NAME = "Alex"
RUN_ALL_HISTORIES = False

# Chronological split
TRAIN_PCT = 0.70
VAL_PCT   = 0.85

# History profile size
HISTORY_TARGET_K = 28
HISTORY_PROFILES: List[List[Dict[str, str]]] = [
    [
        {"title": "Ferrari turbo model lands", "text": "track performance carbon-ceramic"},
        {"title": "Tesla expands Supercharger", "text": "fast charging 800V roadmap"},
        {"title": "BMW semi-autonomous suite", "text": "ADAS lane-keeping OTA"},
        {"title": "NVIDIA AI chips for cars", "text": "edge sensor fusion autopilot"},
        {"title": "Action blockbuster hits box office", "text": "stunts chases sequel"},
        {"title": "Markets wary on inflation & rates", "text": "EPS guidance bond yields ETF"},
    ],
    [
        {"title": "Vector search in RAG", "text": "FAISS ANN recall latency"},
        {"title": "Cloud scale inference", "text": "throughput autoscaling GPU TPU"},
        {"title": "FM eval & safety", "text": "privacy red-teaming benchmarks"},
        {"title": "Quantum materials", "text": "superconductivity phase diagram"},
        {"title": "Chips roadmap", "text": "packaging HBM memory bandwidth"},
    ],
]
