# Recommendation Engine (v5.5) 

A lightweight, reproducible recommendation system designed for clean experiments and strict evaluation. It simulates a multi-domain news corpus (cars, movies, tech, basketball, finance, etc.), builds user profiles from reading history, and ranks candidate items using an ensemble of classic ML models and text embeddings.

---

## Key Principles

- Strict train/val/test separation. Models fit only on TRAIN.  
- Threshold selection τ is chosen only on VAL (maximize accuracy or F1).  
- One-shot TEST report. No refit on VAL/TEST, no peeking, no leakage.  
- Determinism. Explicit RNG seeding and artifacts per seed.  
- RAM-safe. Works on modest hardware; BLAS threads clamped; “FAST” vs “ACCURATE” presets.

---

## Project Structure (matches ZIP)

```
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
```
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
```

---

## Data and Splits

- Synthetic multi-domain corpus via `build_big_corpus(seed, now_anchor)` (see `core_model.py` / `env_and_imports.py`).  
  Each article has: `article_id, title, text, category, published_ts, source`.
- Chronological split by publication timestamp → TRAIN / VAL / TEST.  
- User history profiles: pre-baked profiles represent a user’s past reads (e.g., Hila Ronen “cars-heavy” or mixed).

---

## Algorithms (Modeling Pipeline)

### 1) Text Embeddings
- Global TF-IDF (`TfidfVectorizer`) on article bodies (1–2 grams, sublinear_tf=True, caps via TF_BODY_MAXFEAT).
- Optional LSA (Truncated SVD to `lsa_dim`, followed by Normalizer) controlled by preset constants.
- Word2Vec / Pseudo2Vec
  - If gensim is available: train skip-gram W2V (dim, window, min_count, sg=1).
  - Otherwise: LSA-term pseudo2vec fallback (deterministic term vectors from the global space).
- GlobalEmbedder and W2VEmbedder classes expose `.fit()` and `.transform()` for articles and queries.

### 2) Features and Constants
- All tunables live in constants blocks (TF-IDF min/max DF, max features; SVD dim; recency decay; K-folds; thresholds).  
- No inline literals — improves auditability and reproducibility.

### 3) Base Learners (bagged / K-out)
- Logistic Regression (liblinear, class_weight balanced) — calibrated clicks vs non-clicks.  
- HistGradientBoostingClassifier — robust to non-linearities.  
- ExtraTreesClassifier — variance reduction via randomization.  
- Optional sample weights (exposure priors / recency) when enabled by constants.

### 4) Meta-Stacking (Optional)
- Concatenates base model scores and selected features; trains a simple Logistic Regression meta-model on TRAIN via K-folds.  
- Prevents data leakage by fitting only on TRAIN folds; VAL/TEST consume frozen stackers.

### 5) Centroid and Co-Visitation Signals
- Profile centroids in the embedding space (cosine) for user history.  
- Co-visitation / co-occurrence counts built on TRAIN only (no cross-split leakage).

### 6) Negative Sampling and Hard-Negative Mining
- Balanced mini-batches with negative sampling from non-clicked impressions.  
- Optional HNM: train a warmup LR on standardized features, score TRAIN, keep top-scoring negatives (hard_neg_multiplier), optionally downsample positives via keep_pos_frac (see `hard_negative_mining.py`).

### 7) Threshold Selection (VAL only)
- Sweep τ in [0,1] to maximize Accuracy (or F1) on VAL.  
- Freeze τ* and use it unchanged on TEST for final reporting.

---

## Protocol: End-to-End

1. Build corpus → articles, NOW  
2. Split by time → TRAIN / VAL / TEST  
3. Fit text models on TRAIN (TF-IDF/LSA/W2V) and transform all splits using the frozen transforms.  
4. Train base models on TRAIN (optionally with HNM).  
5. Select τ on VAL (report metrics, choose deployment threshold).  
6. Report once on TEST with frozen models and fixed τ*.  
7. Export artifacts (plots and personal Top-K lists).

---

## Artifacts

Per-seed directory (example: `reco_engine_artifacts_v5_5/seed_3/`):

```
ROC.png, PR.png, confusion_matrix.png
interactions_all.csv
val_metrics.json, test_metrics.json
val_topK_<NAME>.txt, test_topK_<NAME>.txt
```

Personalized output (as used in examples):
```
test_top5_Hila_Ronen.txt
val_top5_Hila_Ronen.txt
```

To print the personalized list from disk:
```python
# show_recommendations.py
import pandas as pd

print(" Hila Ronen — Personalized Recommendations (TEST split, seed=3)")
print("=" * 75)
with open(top5_path, encoding="utf-8") as f:
    print(f.read())
```

---

## Running

### A) From main.py
```bash
# (optional) choose preset
export RECO_PRESET=ACCURATE  # or FAST

# run
python -m "Recommendation Engine.main"
```

The runner logs something like:
```
[PRESET_MODE] FAST
[ENV] RECO_PRESET=FAST
[W2V] Using gensim Word2Vec   # or "Using LSA-term pseudo2vec (fallback)"
[Stage] history #0
===== RUN seed=3 =====
...
Artifacts saved under: reco_engine_artifacts_v5_5
 - seed_*/: ROC/PR/Confusion + interactions_all.csv + top10 files
 - multi_run_summary.csv
```

### B) Show personal Top-K (Hila)
```bash
python -m "Recommendation Engine.src.Recommendation Engine.show_recommendations"
```


---

## Reproducibility

- Seeds: multi-seed runs (e.g., 3, 5, …) logged per directory; each seed stores metrics and plots.  
- Deterministic TF/LSA/W2V given fixed seeds and gensim availability.  
- Thread caps for BLAS/OpenMP.

---

## Troubleshooting

- “No summary file found …/multi_run_summary.csv”  
  That file appears only after multi-seed runs finish and write the aggregator. It’s normal to be missing if a run crashed early or only a single seed executed.


- Paths with spaces  
  Python modules can be invoked with `-m "package.with spaces.module"`. Consider renaming folders to avoid quoting.

- NameError for constants (e.g., `GEN_TF_MIN_DF`)  
  Ensure all constants live in the same config/env module and are imported (no “magic numbers”).

---

## No-Cheating Checklist

- [x] Fit only on TRAIN (including text models, priors, centroids, co-vis).  
- [x] Choose τ on VAL only.  
- [x] Report once on TEST with frozen models/τ.  
- [x] No feature computed using VAL/TEST labels or statistics.  
- [x] All randomness seeded and logged.  
- [x] Constants block; no implicit defaults.

---

## Configuration Highlights

- `TOP_K_RECS` — personal list length (default 5).  
- `RECIPIENT_NAME` — display name for personalized outputs (e.g., "Hila Ronen").  
- `SEEDS`, `HISTORY_PROFILES`, `RUN_ALL_HISTORIES` — multi-run driver knobs.  
- `TF_MIN_DF`, `TF_MAX_DF`, `TF_BODY_MAXFEAT`, `LSA_DIM` — text features.  
- `W2V_DIM`, `W2V_WINDOW`, `W2V_MIN_COUNT` — W2V training (if gensim available).  
- `HNM_*` — hard-negative mining controls.  
- `N_IMPRESSIONS`, `SLATE_SIZE`, `DECAY_*` — simulator and scoring.  
- `K_FOLDS`, `BASE_LR_C`, `HGB_PARAMS`, `EXTRATREES_PARAMS` — learners/stacker.

All live in a central constants/config area (see `env_and_imports.py` and related modules).



