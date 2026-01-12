# Recommendation Engine

## 1) Overview
This project implements an end-to-end **Recommendation Engine / Retrieval Benchmarking pipeline** for text items (e.g., news articles).  
It evaluates multiple **document representations** and multiple **machine learning models**, selects the best pipeline using a strict evaluation protocol, and produces **Top-K recommendations** with complete artifacts (tables, plots, error analysis).

The repository is designed to run in **Google Colab**, ensuring stable environments and sufficient compute resources.

---

## 2) Key Features
 Multiple text representations (TF-IDF, LSA/SVD, Word2Vec document embeddings)  
 Multiple supervised ML models benchmarked under the same protocol  
 Outer **Stratified K-Fold Cross-Validation** evaluation  
 Inner stratified validation split for threshold tuning  
 Threshold chosen by **maximizing F1 on validation**  
 Full metrics and plots (ROC / PR / Calibration)  
 Top-K recommendation generation + export to CSV/JSON  
 Leakage-safe protocol (“no cheating” checklist)

---

## 3) Run in Google Colab (Recommended)

### 3.1 Upload + Extract
Upload the project zip(s) into Colab and run:

```python
!unzip -q src.zip
%cd /content/src
```

---

### 3.2 Install Dependencies (Colab-safe)
```python
!python -m pip install -q datasets matplotlib gensim scikit-learn scipy pandas numpy
```



### 3.3 Run Full Benchmark
```python
from benchmark_runner import run_benchmark
from config import Config

cfg = Config(preset="quality", seed=42)
run_benchmark(cfg)
```

---

### 3.4 Generate Top-K Recommendations
```python
from topk_retrieval import run_topk
run_topk(cfg)
```

---

## 4) Repository Structure
```
src/
 ├── benchmark_runner.py            # benchmarking orchestration + CV
 ├── config.py                     # configuration dataclass + presets
 ├── setup_environment.py          # environment checks / safe installs
 ├── text_preprocessing.py         # normalization + tokenization
 ├── data_reps_models.py           # representations + model builders
 ├── metrics_threshold_plots.py    # metrics + threshold selection + plots
 ├── topk_retrieval.py             # top-k recommendation generation
 ├── inspect_artifacts.py          # artifact sanity checks
 ├── view_topk.py                  # display top-k results
 ├── view_history.py               # inspect session/user history
artifacts/
 ├── results_folds.csv
 ├── results_summary.csv
 ├── winner.json
 ├── recommendations_topk.csv
 ├── recommendations_history_idx.json
 ├── winner_oof_roc.png
 ├── winner_oof_pr.png
 ├── winner_oof_calibration.png
 ├── winner_oof_calibration_bins.csv
 ├── winner_oof_false_positives.csv
 ├── winner_oof_false_negatives.csv
 ├── winner_oof_most_uncertain.csv
```

---

## 5) Data
The dataset is loaded using HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("...")
```

Each sample includes at minimum:
- `text` (document/article)
- `label` (binary relevance / click / interest)

---

## 6) Text Preprocessing
Documents are normalized to reduce noise and increase generalization. The preprocessing pipeline typically includes:

- lowercasing  
- URL replacement  
- email replacement  
- numeric normalization  
- punctuation cleanup  
- whitespace normalization  
- tokenization  
- stopword removal (if enabled)

The output is a token list \(T_d\) for each document \(d\).

---

## 7) Document Representations (Embeddings)

Each document \(d\) is mapped to a numeric vector:

\[
d \rightarrow x_d
\]

### 7.1 TF-IDF
Let:
- \(N\) = number of documents
- \(df(t)\) = number of documents containing term \(t\)

\[
idf(t)=\log\left(\frac{N+1}{df(t)+1}\right)+1
\]

\[
tfidf(t,d)=tf(t,d)\cdot idf(t)
\]

Each document vector:

\[
x_d \in \mathbb{R}^{|V|}
\]

---

### 7.2 LSA / Truncated SVD over TF-IDF
We reduce the TF-IDF matrix \(X\) by truncated SVD:

\[
X \approx U_k \Sigma_k V_k^T
\]

Dense document embedding:

\[
z_d = U_k(d,:) \Sigma_k
\quad\Rightarrow\quad
z_d \in \mathbb{R}^{k}
\]

---

### 7.3 Word2Vec Document Embedding (Average Pooling)
Word2Vec learns a dense vector per token:

\[
v(w)\in\mathbb{R}^{D}
\]

Document embedding by mean pooling:

\[
q(d)=\frac{1}{|T_d|}\sum_{w\in T_d} v(w)
\]

Result:

\[
q(d)\in\mathbb{R}^{D}
\]

---

### 7.4 User / Query Representation by History Pooling
For a user history \(H=\{d_1,\dots,d_m\}\):

\[
u(H)=\frac{1}{m}\sum_{i=1}^{m} x_{d_i}
\]

This yields a stable profile embedding representing the user’s interest.

---

## 8) Models Evaluated

### Core Models
The main benchmarking pipeline evaluates:

- **Logistic Regression**
- **Linear SVM (LinearSVC)**
- **HistGradientBoostingClassifier (HGB)**

---

### Additional Models Tested (Not Selected as Winner)
The following models were also trained and evaluated under the same protocol.  
They produced valid results but were **not good enough** compared to the selected best configuration (especially on PR-AUC / F1 stability), therefore they were not chosen as the final winner:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**

---

## 9) Model Formulas

### 9.1 Logistic Regression
Binary probability:

\[
p(y=1|x)=\sigma(w^Tx+b)
\]

Sigmoid:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

---

### 9.2 Linear SVM (LinearSVC)
\[
\min_{w} \frac{1}{2}\|w\|^2 + C\sum_i\xi_i
\]

---

### 9.3 HistGradientBoostingClassifier (HGB)
Boosted additive model:

\[
F_M(x)=\sum_{m=1}^{M}\eta\cdot h_m(x)
\]

Where:
- \(h_m(x)\) = weak learner (tree)
- \(M\) = boosting iterations
- \(\eta\) = learning rate

---

### 9.4 Random Forest Classifier (Tested)
Random Forest averages tree predictions:

\[
\hat{p}(x)=\frac{1}{T}\sum_{t=1}^{T} p_t(x)
\]

Where:
- \(T\) = number of trees
- \(p_t(x)\) = probability output from the \(t\)-th tree

---

### 9.5 Gradient Boosting Classifier (Tested)
Additive boosting model:

\[
F_M(x) = \sum_{m=1}^{M} \eta \cdot h_m(x)
\]

Where:
- \(h_m(x)\) = weak learner (tree)
- \(\eta\) = learning rate
- \(M\) = number of iterations

---

## 10) Evaluation Protocol (Implemented in Code)

### 10.1 Outer Cross-Validation: Stratified K-Fold
The benchmark uses **Stratified K-Fold** as an outer evaluation loop.  
The dataset is split into \(K\) folds \(F_1,\dots,F_K\), each preserving label distribution.

For each fold \(i\):

\[
Train_i=\bigcup_{j\ne i}F_j
\]
\[
Test_i=F_i
\]

Final reported metric across folds:

\[
metric_{mean}=\frac{1}{K}\sum_{i=1}^{K} metric_i
\]

---

### 10.2 Inner Validation Split: StratifiedShuffleSplit
Inside each outer fold training partition, the pipeline performs an **inner stratified split**:

- inner-train  
- inner-validation  

The inner validation split is used for:
- decision threshold tuning
- stable model selection without using outer test labels

---

## 11) Threshold Selection (Validation-Based F1)

Models output predicted probabilities/scores \(p\).  
Predictions are:

\[
\hat{y}=
\begin{cases}
1 & p\ge\tau\\
0 & p<\tau
\end{cases}
\]

The threshold is selected using validation only:

\[
\tau = \arg\max_{\tau} F1(\text{VAL},\tau)
\]

 Test labels are never used to choose \(\tau\).  
The threshold is stored in artifacts per fold as `thr_from_val`.

---

## 12) Metrics Reported

### 12.1 Accuracy
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

### 12.2 Precision
\[
Precision=\frac{TP}{TP+FP}
\]

### 12.3 Recall
\[
Recall=\frac{TP}{TP+FN}
\]

### 12.4 F1 Score
\[
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
\]

### 12.5 ROC-AUC
Definitions:

\[
TPR=\frac{TP}{TP+FN}
\quad,\quad
FPR=\frac{FP}{FP+TN}
\]

ROC-AUC is the area under the ROC curve.

### 12.6 PR-AUC
PR-AUC is the area under Precision-Recall curve (very informative in class imbalance).

### 12.7 Calibration
Calibration evaluates probability quality vs observed outcomes.

### 12.8 Overfitting / Generalization Gap
\[
Gap_{metric}=metric_{train}-metric_{val}
\]

---

## 13) Top-K Recommendation Generation
After selecting the winning pipeline, Top-K recommendations are produced.

For a user embedding \(u\) and candidate set \(C\):

\[
TopK(u) = \arg\max_{d\in C} score(u,d)
\]

Output file:
- `recommendations_topk.csv`

---

## 14) Output Artifacts

### 14.1 Benchmark Results
- `results_folds.csv` — fold-level metrics per representation/model  
- `results_summary.csv` — aggregated mean/std  
- `winner.json` — final selected configuration  

### 14.2 Recommendations
- `recommendations_topk.csv`  
- `recommendations_history_idx.json`

### 14.3 Plots
- `winner_oof_roc.png`
- `winner_oof_pr.png`
- `winner_oof_calibration.png`
- `winner_oof_calibration_bins.csv`

### 14.4 Error Analysis
- `winner_oof_false_positives.csv`
- `winner_oof_false_negatives.csv`
- `winner_oof_most_uncertain.csv`

---

 

## 15) Summary
This project delivers a complete, leakage-safe recommendation benchmarking framework:
- multiple representations + models
- strict **outer Stratified K-Fold** evaluation
- inner validation threshold selection using **F1**
- comprehensive metrics, plots, artifacts
- Top-K recommendation output ready for report/submission
