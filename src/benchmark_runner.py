#  Main benchmark runner (Nested CV, Winner, OOF, Saves, TopK demo)


import json                      # saving winner metadata to json
import time                      # timing folds
from typing import Dict, List, Any  # typing hints for readability

# Third party imports
import numpy as np               # numeric arrays
import pandas as pd              # tables, saving csv

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # nested CV splitting
from sklearn.base import clone   # clone estimators/pipelines safely for folds

def run_benchmark(cfg: "Config") -> None:
    """
    Runs the full pipeline end to end:
    Load data
    Make optional binary target
    Nested evaluation using outer CV with an inner validation split
    Threshold selection is done ONLY on inner validation
    Winner is selected using summary metrics
    Winner OOF diagnostics are generated
    TopK demo recommender outputs are saved
    """

    ensure_dir(cfg.out_dir)                    # ensure artifacts folder exists
    ensure_dir(cfg.cache_dir)                  # ensure cache folder exists (HF datasets may use it)

    texts, y, times_from_data = load_dataset_hf(cfg)   # load only train split texts and labels

    if bool(cfg.make_binary):                          # if configured as binary task
        y = make_binary_target(y, cfg.pos_class)       # convert multiclass labels to binary

    reps = build_representations(cfg)                  # build representation pipelines
    models = build_models(cfg)                         # build model dict

    print("[RUN] Outer folds:", cfg.outer_folds, "| Inner val size:", cfg.inner_val_size)  # CV settings
    print("[RUN] Overfit rule: train_acc - val_acc >", cfg.overfit_acc_gap_thr)           # overfit policy
    print("[RUN] Top-K: k=", cfg.topk_k, "| theta=", cfg.topk_theta)                      # recommender knobs

    skf = StratifiedKFold(                             # create stratified outer CV
        n_splits=int(cfg.outer_folds),
        shuffle=True,
        random_state=int(cfg.seed)
    )

    rows: List[Dict[str, Any]] = []                    # store fold results row by row

    for fold, (outer_tr_idx, outer_te_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        t0 = time.time()                               # fold timer start
        print(f"\n[CV] OUTER FOLD {fold}/{cfg.outer_folds}")  # fold header

        sss = StratifiedShuffleSplit(                  # inner split for threshold selection
            n_splits=1,
            test_size=float(cfg.inner_val_size),
            random_state=int(cfg.seed) + fold
        )

        inner_tr_rel, inner_val_rel = next(sss.split(np.zeros(len(outer_tr_idx)), y[outer_tr_idx]))  # split indices
        inner_tr_idx = outer_tr_idx[inner_tr_rel]      # map inner train to original indices
        inner_val_idx = outer_tr_idx[inner_val_rel]    # map inner val to original indices

        inner_tr_texts = [texts[i] for i in inner_tr_idx]   # texts for inner train
        inner_val_texts = [texts[i] for i in inner_val_idx] # texts for inner validation
        outer_tr_texts = [texts[i] for i in outer_tr_idx]   # texts for outer train refit
        outer_te_texts = [texts[i] for i in outer_te_idx]   # texts for outer test scoring

        y_inner_tr = y[inner_tr_idx]                   # labels for inner train
        y_inner_val = y[inner_val_idx]                 # labels for inner validation
        y_outer_tr = y[outer_tr_idx]                   # labels for outer train
        y_outer_te = y[outer_te_idx]                   # labels for outer test

        for rep_name, rep_obj in reps.items():         # loop representations
            for model_name, model_obj in models.items():  # loop models

                if rep_name.startswith("w2v") and ((not RUN_W2V) or (model_name != W2V_ONLY_FOR_MODEL)):
                    continue                           # skip W2V combinations except the allowed one

                fitted_inner, X_tr, X_val = featurize(rep_obj, inner_tr_texts, inner_val_texts)  # fold safe features
                est = clone(model_obj)                 # fresh estimator instance
                est.fit(X_tr, y_inner_tr)              # train on inner train only

                s_tr = model_scores(est, X_tr)         # train scores for metrics
                s_val = model_scores(est, X_val)       # validation scores for threshold selection

                thr = choose_threshold_best_f1(y_inner_val, s_val)  # threshold chosen ONLY on val

                train_m = compute_metrics(y_inner_tr, s_tr, thr)    # train metrics at thr
                val_m = compute_metrics(y_inner_val, s_val, thr)    # val metrics at thr

                acc_gap = float(train_m["acc"] - val_m["acc"])      # gap for overfitting check
                overfit_flag = int(bool(cfg.enable_overfit_flag) and (acc_gap > float(cfg.overfit_acc_gap_thr)))  # flag

                fitted_outer, X_outer_tr, X_outer_te = featurize(rep_obj, outer_tr_texts, outer_te_texts)  # fold safe features
                est2 = clone(model_obj)                # fresh estimator for outer train
                est2.fit(X_outer_tr, y_outer_tr)       # refit on full outer train

                s_te = model_scores(est2, X_outer_te)  # score outer test
                test_m = compute_metrics(y_outer_te, s_te, thr)  # test metrics using thr from inner val

                rows.append({                          # store results for this rep x model x fold
                    "fold": fold,
                    "rep": rep_name,
                    "model": model_name,
                    "thr_from_val": float(thr),

                    "overfit_acc_gap": acc_gap,
                    "overfit_flag": overfit_flag,

                    "train_pr_auc": train_m["pr_auc"],
                    "train_roc_auc": train_m["roc_auc"],
                    "train_acc": train_m["acc"],
                    "train_f1": train_m["f1"],
                    "train_mcc": train_m["mcc"],
                    "train_bal_acc": train_m["bal_acc"],

                    "val_pr_auc": val_m["pr_auc"],
                    "val_roc_auc": val_m["roc_auc"],
                    "val_acc": val_m["acc"],
                    "val_f1": val_m["f1"],
                    "val_mcc": val_m["mcc"],
                    "val_bal_acc": val_m["bal_acc"],

                    "test_pr_auc": test_m["pr_auc"],
                    "test_roc_auc": test_m["roc_auc"],
                    "test_acc": test_m["acc"],
                    "test_f1": test_m["f1"],
                    "test_mcc": test_m["mcc"],
                    "test_bal_acc": test_m["bal_acc"],
                })

        print("[CV] Fold done | rows=", len(rows), "| time_sec=", round(time.time() - t0, 1))  # fold summary

    df = pd.DataFrame(rows)                             # results table (fold level)
    folds_path = os.path.join(cfg.out_dir, "results_folds.csv")  # output path
    df.to_csv(folds_path, index=False)                  # save fold level results
    print("[SAVE] results_folds.csv ->", folds_path)     # confirm save

    group_keys = ["rep", "model"]                       # grouping dimensions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # numeric cols only

    if "fold" in numeric_cols:                          # fold is id not metric
        numeric_cols.remove("fold")                     # remove fold from aggregation

    df_summary = df.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).reset_index()  # mean/std table

    df_summary.columns = [                              # flatten multiindex column names
        c if not isinstance(c, tuple) else (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        for c in df_summary.columns
    ]

    summary_path = os.path.join(cfg.out_dir, "results_summary.csv")  # output path
    df_summary.to_csv(summary_path, index=False)          # save summary
    print("[SAVE] results_summary.csv ->", summary_path)   # confirm save

    if bool(cfg.enable_overfit_flag):                    # if overfit filtering enabled
        df_ok = df_summary[df_summary["overfit_flag_mean"] == 0].copy()  # keep only safe candidates
        if df_ok.empty:                                  # nothing left
            print("[STOP] All candidates are flagged as OVERFITTING. No winner selected.")  # abort
            return
    else:
        df_ok = df_summary                               # use all candidates

    winner_row = pick_winner_row(df_ok, cfg)             # pick best row by policy

    winner_info = {                                     # pack metadata to json
        "winner_rep": str(winner_row["rep"]),
        "winner_model": str(winner_row["model"]),
        "winner_primary": cfg.primary_metric,
        "results_folds_csv": folds_path,
        "results_summary_csv": summary_path,
        "winner_metrics_mean": {k: float(winner_row[k]) for k in winner_row.index if k.endswith("_mean")},
    }

    winner_json_path = os.path.join(cfg.out_dir, "winner.json")  # output path
    with open(winner_json_path, "w", encoding="utf-8") as f:     # write json file
        json.dump(winner_info, f, indent=2, ensure_ascii=False)  # save formatted

    print("\n[WINNER] rep:", winner_info["winner_rep"])          # winner rep
    print("[WINNER] model:", winner_info["winner_model"])        # winner model
    print("[SAVE] winner.json ->", winner_json_path)             # confirm save

    rep_obj = build_representations(cfg)[winner_info["winner_rep"]]   # rebuild rep object by name
    model_obj = build_models(cfg)[winner_info["winner_model"]]        # rebuild model object by name

    oof_y: List[int] = []                                 # OOF labels
    oof_s: List[float] = []                               # OOF scores
    oof_texts: List[str] = []                             # OOF texts (for error analysis)
    oof_thr: List[float] = []                             # thresholds selected per fold

    for fold, (outer_tr_idx, outer_te_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        sss = StratifiedShuffleSplit(                     # inner split again for threshold
            n_splits=1,
            test_size=float(cfg.inner_val_size),
            random_state=int(cfg.seed) + fold
        )

        inner_tr_rel, inner_val_rel = next(sss.split(np.zeros(len(outer_tr_idx)), y[outer_tr_idx]))  # rel indices
        inner_tr_idx = outer_tr_idx[inner_tr_rel]         # absolute indices
        inner_val_idx = outer_tr_idx[inner_val_rel]       # absolute indices

        inner_tr_texts = [texts[i] for i in inner_tr_idx] # inner train texts
        inner_val_texts = [texts[i] for i in inner_val_idx] # inner val texts
        outer_tr_texts = [texts[i] for i in outer_tr_idx] # outer train texts
        outer_te_texts = [texts[i] for i in outer_te_idx] # outer test texts

        y_inner_tr = y[inner_tr_idx]                      # inner train labels
        y_inner_val = y[inner_val_idx]                    # inner val labels
        y_outer_tr = y[outer_tr_idx]                      # outer train labels
        y_outer_te = y[outer_te_idx]                      # outer test labels

        fitted_inner, X_tr, X_val = featurize(rep_obj, inner_tr_texts, inner_val_texts)  # fold safe features
        est = clone(model_obj)                            # estimator instance
        est.fit(X_tr, y_inner_tr)                         # fit on inner train
        s_val = model_scores(est, X_val)                  # score inner val
        thr = choose_threshold_best_f1(y_inner_val, s_val)  # threshold per fold
        oof_thr.append(float(thr))                        # store threshold

        fitted_outer, X_outer_tr, X_outer_te = featurize(rep_obj, outer_tr_texts, outer_te_texts)  # outer features
        est2 = clone(model_obj)                           # new estimator
        est2.fit(X_outer_tr, y_outer_tr)                  # train on outer train
        s_te = model_scores(est2, X_outer_te)             # score outer test

        oof_y.extend(list(y_outer_te.astype(int)))        # collect true labels
        oof_s.extend(list(s_te.astype(float)))            # collect scores
        oof_texts.extend(list(outer_te_texts))            # collect texts

    oof_y_arr = np.asarray(oof_y, dtype=int)              # convert list to array
    oof_s_arr = np.asarray(oof_s, dtype=float)            # convert list to array
    thr_global = float(np.median(oof_thr)) if len(oof_thr) else 0.5  # robust global threshold

    plot_paths = save_roc_pr_calibration(cfg.out_dir, oof_y_arr, oof_s_arr, prefix="winner_oof")  # save plots
    err_paths = save_error_analysis(cfg.out_dir, oof_texts, oof_y_arr, oof_s_arr, thr_global, prefix="winner_oof")  # save csv errors

    print("[OOF] thr_global (median of fold thresholds):", round(thr_global, 6))  # print OOF thr
    print("[SAVE] OOF plots:", plot_paths)                                         # print plot files
    print("[SAVE] Error analysis CSVs:", err_paths)                                # print error csv files

    print("\n[TOPK] Building Top-K recommendations demo...")  # start recommender demo

    fitted_all = clone(rep_obj)                              # new rep fitted on full set
    X_all = fitted_all.fit_transform(texts)                  # fit on full texts for retrieval embedding
    if hasattr(X_all, "toarray"):                            # sparse to dense if needed
        X_all = X_all.toarray()
    X_all = X_all.astype(np.float32)                         # ensure float32 for speed

    norms = np.linalg.norm(X_all, axis=1, keepdims=True)      # L2 norms per doc
    norms = np.where(norms == 0, 1.0, norms)                  # avoid divide by zero
    X_norm = (X_all / norms).astype(np.float32)               # normalized document vectors

    doc_times = get_doc_times(cfg, n_docs=len(texts), times_from_data=times_from_data)  # timestamps per doc

    rng = np.random.default_rng(int(cfg.seed))                # rng for reproducibility
    history_size = int(min(cfg.topk_history_size, len(texts) - 1))  # avoid too large history
    history_idx = rng.choice(len(texts), size=history_size, replace=False).tolist()     # simulated history

    rec_df = top_k_time_aware(                                # compute TopK recommendations
        texts=texts,
        X_norm=X_norm,
        doc_times=doc_times,
        history_idx=history_idx,
        top_k=int(cfg.topk_k),
        theta=float(cfg.topk_theta),
    )

    rec_path = os.path.join(cfg.out_dir, "recommendations_topk.csv")  # save path for recs
    rec_df.to_csv(rec_path, index=False)                      # save recommendations
    print("[SAVE] recommendations_topk.csv ->", rec_path)      # confirm save

    hist_path = os.path.join(cfg.out_dir, "recommendations_history_idx.json")  # save path history
    with open(hist_path, "w", encoding="utf-8") as f:          # open json file
        json.dump({"history_idx": history_idx}, f, indent=2, ensure_ascii=False)  # write history indices
    print("[SAVE] recommendations_history_idx.json ->", hist_path)  # confirm save

    history_df = pd.DataFrame({                               # build readable history dataframe
        "doc_idx": [int(i) for i in history_idx],              # chosen doc indices
        "doc_time": [float(doc_times[int(i)]) for i in history_idx],  # their times
        "text": [texts[int(i)] for i in history_idx],          # raw text
    }).sort_values("doc_time").reset_index(drop=True)          # sort by time and reset row index

    history_csv = os.path.join(cfg.out_dir, "history_read_only.csv")  # save path history csv
    history_df.to_csv(history_csv, index=False)                # save history csv
    print("[SAVE] history_read_only.csv ->", history_csv)       # confirm save

    sort_cols = [                                              # preferred sorting metrics if available
        c for c in [
            "val_pr_auc_mean",
            "val_roc_auc_mean",
            "val_f1_mean",
            "val_mcc_mean",
            "val_bal_acc_mean",
            "val_acc_mean"
        ]
        if c in df_summary.columns
    ]

    print("\n[SUMMARY] Results summary (sorted by main/tie metrics):")  # show final summary table
    print(df_summary.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).to_string(index=False))  # print

print("[OK] run_benchmark(cfg) is defined.")  # confirmation


run_benchmark(cfg)  # execute everything
