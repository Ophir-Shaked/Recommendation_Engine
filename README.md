# Recommendation Engine (v5.5)

A lightweight, reproducible recommendation system designed for clean experiments and strict evaluation. It simulates a multi-domain news corpus (cars, movies, tech, basketball, finance, etc.), builds user profiles from reading history, and ranks candidate items using an ensemble of classic ML models and text embeddings.

---

## Key Principles

- **Strict train / validation / test separation**  
  All models, embeddings, statistics, and signals are fit on TRAIN only.

- **Threshold selection on validation only**  
  The decision threshold \( \tau \) is chosen exclusively on VAL (maximize Accuracy or F1).

- **One-shot TEST report**  
  TEST is evaluated exactly once, with frozen models and fixed \( \tau \). No refitting, no peeking, no leakage.

- **Determinism**  
  Explicit RNG seeding and per-seed artifacts.

- **RAM-safe execution**  
  Works on modest hardware; BLAS/OpenMP threads are capped; “FAST” vs “ACCURATE” presets.

---

## Project Structure (matches ZIP)

```text
recommendation_engine/
  Recommendation Engine/
    main.py
    src/
      Recommendation Engine/
        core_model.py
        embedders.py
        env_and_imports.py
        export_utils.py
        hard_negative_mining.py
        plots_and_metrics.py
        runner.py
        show_recommendations.py
```

---

## Presets and Environment

Two execution modes, selected before running:

```bash
# Recommended for quality
export RECO_PRESET=ACCURATE

# Faster / lighter (default if not set)
export RECO_PRESET=FAST
```

Internal guards keep BLAS sane on small machines:

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
```

---

## What the System Actually Does

This system is not “a single model that recommends items.”  
It is an experiment framework that:

1. Builds a time-ordered multi-domain article corpus  
2. Simulates user reading histories and impression slates  
3. Converts text into vector spaces (TF-IDF / optional LSA / Word2Vec)  
4. Computes multiple signals per (user, candidate item)  
5. Trains supervised learners to predict click probability  
6. Optionally stacks learners (leakage-safe) for better calibration and robustness  
7. Chooses a deployment threshold \( \tau \) on VAL only  
8. Produces a final one-shot TEST report and exports artifacts and Top-K lists  

The protocol is fixed and explicit.

---

## Data and Splits

- **Synthetic multi-domain corpus** via `build_big_corpus(seed, now_anchor)`  
  Each article contains:  
  `article_id, title, text, category, published_ts, source`

- **Chronological split by publication timestamp**  
  TRAIN / VAL / TEST (no temporal leakage)

- **User history profiles**  
  Pre-baked reading histories representing different interest patterns  
  (e.g., “cars-heavy”, “mixed”).

---

## Algorithms (Modeling Pipeline)

### 1) Text Embeddings (Representation Models)

All representation models are **fit on TRAIN only** and frozen for VAL/TEST.

#### 1.1 TF-IDF (Global)

\[
\text{TF-IDF}(t,d)=\text{TF}(t,d)\cdot \log\frac{N}{\text{DF}(t)}
\]

- Global TF-IDF (unigrams + bigrams, sublinear TF)
- Sparse, high-dimensional baseline representation
- Strong performance with classical ML

---

#### 1.2 Optional LSA via Truncated SVD (Dimensionality Reduction)

\[
X \approx U_k \Sigma_k V_k^\top
\]

- TF-IDF → Truncated SVD → Normalizer  
- Reduces dimensionality and noise  
- Captures latent semantic structure  
- Improves cosine stability and efficiency  
- Optional and preset-controlled  

**Motivation:**
- Compress TF-IDF into a dense semantic space
- Reduce noise and improve generalization for some models/signals
- Enable faster cosine computations / centroids in low-dim space

---

#### 1.3 Word2Vec / Pseudo2Vec (Train-time dependent)

- Skip-gram Word2Vec if `gensim` is available  
- Deterministic pseudo-embedding fallback otherwise  

Ensures reproducibility and consistent behavior across environments.

---

#### 1.4 Embedder APIs

`GlobalEmbedder` and `W2VEmbedder` expose `.fit()` and `.transform()` for both articles and queries.

---

## Features and Constants (Reproducibility Layer)

All tunables (TF-IDF limits, SVD dimension, recency decay, K-folds, thresholds) live in centralized constants.  
No inline magic numbers.

---

## Models (Complete List)

In this project, a **model** is any component that outputs a numeric score or probability used for ranking or classification.

So you have both:

- Supervised predictive models (learn from labels)
- Unsupervised scoring models (cosine/centroid, co-visitation)
- Heuristic temporal models (recency decay)
- Ensemble combination models (stacking)

---

## 3) Base Learners (Supervised ML Models)

These are the classical supervised models trained on TRAIN:

- Logistic Regression (`liblinear`, `class_weight="balanced"`) — calibrated clicks vs non-clicks  
- HistGradientBoostingClassifier — robust to non-linearities  
- ExtraTreesClassifier — variance reduction via randomization  
- Optional sample weights (exposure priors / recency) when enabled by constants

### Logistic Regression and the Sigmoid (Probability Model)

Linear score:
\[
z = w^\top x + b
\]

Sigmoid (logistic) link:
\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Predicted probability:
\[
p(y=1\mid x)=\sigma(w^\top x+b)
\]

This matters because the pipeline is probability-first: you choose \( \tau \) on VAL, and you plot ROC/PR on scores.

---

## 4) Similarity and Scoring Models (Unsupervised / Geometric)

These are not trained on labels, but are first-class models/signals used for ranking and/or as features.

### 4.1 Centroid Profile Model (User Representation)

You compute profile centroids in the embedding space for user history (cosine).  
User vector:

\[
u=\frac{1}{|H|}\sum_{h\in H} v(h)
\]

where \(v(h)\) is the embedding of a previously read article.

---

### 4.2 Cosine Similarity Model (User–Item Matching)

Candidate articles are scored by cosine similarity to the user centroid:

\[
\cos(u,v)=\frac{u\cdot v}{\|u\|\,\|v\|}
\]

This is a strong content-based baseline and a valuable feature for supervised models.

---

### 4.3 Co-Visitation / Co-Occurrence Model (Behavioral Signal)

You build co-visitation/co-occurrence counts on TRAIN only.  
This captures “users who read X also read Y” style signals without leakage.

---

## 5) Temporal / Heuristic Models (Time-Awareness)

### 5.1 Recency Decay Model (Time Weighting)

You explicitly include recency decay tunables.  
A typical form is exponential decay:

\[
w(\Delta t)=\exp(-\Delta t/\tau)
\]

This influences either:
- ranking score directly (freshness bias)
- sample weights during training (time-aware learning)

---

## 6) Meta-Model: Stacking (Ensemble Model)

You optionally train a stacking meta-model:

- Concatenates base model scores and selected features  
- Trains a Logistic Regression meta-model on TRAIN via K-folds  
- Prevents leakage by fitting only on TRAIN folds; VAL/TEST consume frozen stackers

---

## K-Fold Cross-Validation (Where It’s Used and Why)

Yes — you use K-fold validation, but not as “CV for performance reporting.”  
You use it as a leakage-safe mechanism for stacking:

- Split TRAIN into K folds (`K_FOLDS`)  
- Train base models on K−1 folds  
- Generate out-of-fold predictions on the held-out fold  
- Train the meta Logistic Regression on the concatenated OOF predictions  

This ensures the meta-model never sees base predictions generated from a base model that trained on the same examples.

---

## Negative Sampling and Hard-Negative Mining

- Balanced mini-batches with negative sampling from non-clicked impressions  
- Optional HNM: train a warmup LR on standardized features, score TRAIN, keep top-scoring negatives (`hard_neg_multiplier`), optionally downsample positives via `keep_pos_frac`

This is a modeling trick that improves decision boundaries by focusing on confusing negatives.

---

## Threshold Selection (VAL only)

- Sweep \( \tau \in [0,1] \) to maximize Accuracy (or F1) on VAL  
- Freeze \( \tau^* \) and use it unchanged on TEST for final reporting

This is a critical part of the evaluation discipline (no peeking).

---

## Metrics (Formulas)

### Accuracy
\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision
\[
\text{Precision}=\frac{TP}{TP+FP}
\]

### Recall
\[
\text{Recall}=\frac{TP}{TP+FN}
\]

### F1-Score
\[
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\]

### ROC Curve
\[
TPR=\frac{TP}{TP+FN},\quad FPR=\frac{FP}{FP+TN}
\]

### AUC
\[
\text{AUC}=\int_0^1 TPR(FPR)\,d(FPR)
\]

### Precision–Recall Curve
\[
(\text{Recall},\text{Precision})\ \text{as threshold varies}
\]

### Average Precision (AP)
\[
\text{AP}=\sum_n (\text{Recall}_n-\text{Recall}_{n-1})\cdot\text{Precision}_n
\]

---

## Protocol: End-to-End

1. Build corpus → articles  
2. Split by time → TRAIN / VAL / TEST  
3. Fit text models on TRAIN (TF-IDF/LSA/W2V) and transform all splits using the frozen transforms  
4. Train base models on TRAIN (optionally with HNM)  
5. Select \( \tau \) on VAL (report metrics, choose deployment threshold)  
6. Report once on TEST with frozen models and fixed \( \tau^* \)  
7. Export artifacts (plots and personal Top-K lists)

---

## Artifacts

Per-seed directory contains:

```text
ROC.png
PR.png
confusion_matrix.png
interactions_all.csv
val_metrics.json
test_metrics.json
val_topK_<NAME>.txt
test_topK_<NAME>.txt
```

These are the practical “recommendation artifacts” (what a user would see).

---

## Running

### A) From main.py

```bash
# (optional) choose preset
export RECO_PRESET=ACCURATE  # or FAST

# run
python -m "Recommendation Engine.main"
```

### B) Show personal Top-K

```bash
python -m "Recommendation Engine.src.Recommendation Engine.show_recommendations"
```

---

## Reproducibility

- Multi-seed runs logged per directory; each seed stores metrics and plots  
- Deterministic TF/LSA/W2V given fixed seeds and gensim availability  
- Thread caps for BLAS/OpenMP

---

## Troubleshooting

- “No summary file found …/multi_run_summary.csv” — appears only after multi-seed aggregator finishes  
- Paths with spaces — Python modules can be invoked with `-m "package.with spaces.module"`  
- NameError for constants — ensure constants live in config/env module and are imported

---

## No-Cheating Checklist

- [x] Fit only on TRAIN (including text models, priors, centroids, co-vis)  
- [x] Choose \( \tau \) on VAL only  
- [x] Report once on TEST with frozen models/\( \tau \)  
- [x] No feature computed using VAL/TEST labels or statistics  
- [x] All randomness seeded and logged  
- [x] Constants block; no implicit defaults

---

## Configuration Highlights

- `TOP_K_RECS` — personal list length (default 5)  
- `RECIPIENT_NAME` — display name for personalized outputs  
- `SEEDS`, `HISTORY_PROFILES`, `RUN_ALL_HISTORIES` — multi-run knobs  
- `TF_MIN_DF`, `TF_MAX_DF`, `TF_BODY_MAXFEAT`, `LSA_DIM` — text features  
- `W2V_DIM`, `W2V_WINDOW`, `W2V_MIN_COUNT` — W2V training  
- `HNM_*` — hard-negative mining controls  
- `N_IMPRESSIONS`, `SLATE_SIZE`, `DECAY_*` — simulator/scoring  
- `K_FOLDS`, `BASE_LR_C`, `HGB_PARAMS`, `EXTRATREES_PARAMS` — learners/stacker

All live in a central constants/config area (see `env_and_imports.py` and related modules).

---

## Final “All Models / All Metrics” Summary (No Ambiguity)

### Models (including scoring models)
- TF-IDF representation model  
- Optional LSA (Truncated SVD + Normalizer)  
- Word2Vec (gensim) / Pseudo2Vec fallback  
- Cosine centroid profile model (user centroid + cosine scoring)  
- Co-visitation / co-occurrence model  
- Recency decay / time weighting  
- Logistic Regression  
- HistGradientBoostingClassifier  
- ExtraTreesClassifier  
- Stacking meta-model (Logistic Regression), trained via K-folds  

### Metrics
- Threshold selection on VAL to maximize Accuracy or F1  
- ROC Curve (plot)  
- Precision-Recall Curve (plot)  
- Confusion Matrix (plot)  
- Metrics JSON outputs for VAL/TEST (`val_metrics.json`, `test_metrics.json`)

