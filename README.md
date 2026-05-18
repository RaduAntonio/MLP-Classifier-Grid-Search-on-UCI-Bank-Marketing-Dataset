# MLP Classifier Grid Search on UCI Bank Marketing Dataset

> Systematic hyperparameter exploration of Multi-Layer Perceptron architectures for binary classification on the UCI Bank Marketing dataset (`bank-additional`).

---

## Overview

This project trains and evaluates **12 MLP configurations** by exhaustively combining:

- **Number of hidden layers:** 1 or 2
- **Neurons per layer:** equal to the previous layer or half of it (starting from `n_features = 20`)
- **Learning rate:** `0.1` or `0.01`

Each model is evaluated on Accuracy, Precision, Recall, F1-Score, and AUC-ROC, with a final 5-fold stratified cross-validation on the best configuration.

---

## Dataset

**Bank Marketing Data Set** — UCI Machine Learning Repository  
Source: [https://archive.ics.uci.edu/ml/datasets/Bank+Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

File used: `bank-additional.csv` (10% sample, 4,119 instances, 20 features)

**Task:** Predict whether a client will subscribe to a term deposit (`yes` / `no`).  
**Class imbalance:** ~89% `no`, ~11% `yes`

### Features

| Type | Features |
|------|----------|
| Numeric | `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome` |

---

## Architecture

### Hidden Layer Configurations (6 total)

| Config | Layers | Neurons |
|--------|--------|---------|
| 1 | 1 | (20,) |
| 2 | 1 | (10,) |
| 3 | 2 | (20, 20) |
| 4 | 2 | (20, 10) |
| 5 | 2 | (10, 10) |
| 6 | 2 | (10, 5) |

**Rule:** First layer = `n_features` or `n_features // 2`; second layer (if present) = first layer or `first // 2`.

Combined with 2 learning rates → **12 total combinations**.

### Fixed hyperparameters

| Parameter | Value |
|-----------|-------|
| Activation | ReLU |
| Solver | Adam |
| Max iterations | 500 |
| Early stopping | Yes (patience = 20) |
| Validation fraction | 10% |
| Random state | 42 |

---

## Results Summary

| # | Hidden Layers | LR | Accuracy % | Precision % | Recall % | F1 % | AUC % |
|---|---|---|---|---|---|---|---|
| 1 | (20,) | 0.1 | 91.87 | 76.19 | 35.96 | 48.85 | 87.84 |
| **2** | **(20,)** | **0.01** | **92.84** | **81.25** | **43.82** | **56.93** | **87.62** |
| 3 | (10,) | 0.1 | 91.50 | 69.39 | 38.20 | 49.28 | 86.44 |
| 4 | (10,) | 0.01 | 91.75 | 74.42 | 35.96 | 48.48 | 86.17 |
| 5 | (20, 20) | 0.1 | 90.17 | 56.25 | 40.45 | 47.06 | 80.91 |
| 6 | (20, 20) | 0.01 | 91.87 | 73.91 | 38.20 | 50.37 | 88.11 |
| 7 | (20, 10) | 0.1 | 91.99 | 74.47 | 39.33 | 51.47 | 87.89 |
| 8 | (20, 10) | 0.01 | 92.60 | 85.00 | 38.20 | 52.71 | 88.25 |
| 9 | (10, 10) | 0.1 | 91.63 | 68.52 | 41.57 | 51.75 | 86.76 |
| **10** | **(10, 10)** | **0.01** | **92.35** | **74.07** | **44.94** | **55.94** | **88.51** |
| 11 | (10, 5) | 0.1 | 91.14 | 69.05 | 32.58 | 44.27 | 85.12 |
| 12 | (10, 5) | 0.01 | 92.23 | 76.60 | 40.45 | 52.94 | 88.33 |

**Best F1:** Model #2 — `(20,)`, LR=`0.01` → F1 = **56.93%**  
**Best AUC:** Model #10 — `(10, 10)`, LR=`0.01` → AUC = **88.51%**

**5-fold CV AUC (best model):** `0.8605 ± 0.0222`

### Key observations

- LR = `0.01` consistently outperforms LR = `0.1` — slower learning rate leads to more stable convergence on imbalanced data.
- Shallow networks (1 hidden layer) achieve competitive or superior F1 compared to deeper ones, suggesting the task does not require complex feature hierarchies.
- The widest bottleneck configuration `(10, 5)` with LR = `0.1` performs worst overall (F1 = 44.27%).

---

## Visualizations

The script generates a comprehensive 9-panel figure (`mlp_bank_results.png`) including:

- Grouped bar chart comparing all 5 metrics across all 12 models
- F1 heatmap (architecture × learning rate)
- Confusion matrix for the best model
- ROC curves overlay for all models
- Training loss / validation score curve
- Mean AUC per architecture
- F1 vs Accuracy scatter (colored by LR)
- Convergence iterations per model
- Full results summary table

---

## Project Structure

```
.
├── bank-additional.csv     # Dataset (UCI bank-additional, semicolon-separated)
├── mlp_bank.py             # Main script — preprocessing, grid search, evaluation, plots
├── mlp_bank_results.png    # Output visualization (auto-generated)
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Download the dataset

Get `bank-additional.csv` from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it in the project root.

### 3. Run

```bash
python mlp_bank.py
```

The script will print a full results table to stdout and save `mlp_bank_results.png`.

---

## Requirements

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
```

---

## Academic Context

Developed as part of the *Neural Networks* coursework at **University Politehnica of Bucharest (UPB)**, Master's program in Artificial Intelligence / Computer Science.

---

## License

MIT
