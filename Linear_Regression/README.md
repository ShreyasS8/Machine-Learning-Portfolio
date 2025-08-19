# Linear Regression for Hospital Patient Cost Prediction

This repository implements multiple linear-regression-based models to predict **Total Costs** for hospital patients using a pre-processed subset of the SPARCS dataset. All parts (A, B and C) are implemented from scratch using **NumPy** and **pandas** (no scikit-learn model APIs are used). The project contains three main scripts for modelling tasks plus an exploratory feature-selection script.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Setup & Installation](#setup--installation)
* [Usage](#usage)

  * [Part A — Weighted Linear Regression (linear.py)](#part-a--weighted-linear-regression-linearpy)
  * [Part B — Ridge Regression with CV (linear.py)](#part-b--ridge-regression-with-cv-linearpy)
  * [Part C — Competitive Model (linear\_competitive.py)](#part-c--competitive-model-linear_competitivepy)
  * [Evaluation Script & Objective Function (how Part C is scored)](#evaluation-script--objective-function-how-part-c-is-scored)
* [Implementation Notes](#implementation-notes)
* [Evaluation & Metrics](#evaluation--metrics)
* [Troubleshooting](#troubleshooting)
* [License & Contact](#license--contact)

---

## Project Overview

The goal of this project is to explore linear-model solutions for predicting hospital patient costs. It includes:

* A weighted analytic Ordinary Least Squares estimator (Part A)
* A Ridge Regression pipeline with 10-fold cross-validation for selecting lambda (Part B)
* A competitive feature-engineered pipeline and Ridge model (Part C)
* An exploratory feature selection script used during development

All scripts are command-line driven and expect CSV input files in the `Data/` folder by default.

---

## Repository Structure (actual)

```
<project-root>/
├── Data/                      # expected: train.csv, test.csv (capital D in this repo)
├── feature_selection.py       # exploratory utilities used during feature engineering
├── linear.py                  # implements Part A (weighted) and Part B (ridge+CV)
├── linear_competitive.py      # Part C: feature pipeline + final model
└── README.md                  # this file
```

> Note: Older examples and assignment text sometimes reference `part_a_inputs/` and `part_b_inputs/`. In this copy of the repository those folders are not shown — if you have `weights.txt` or `lambdas.txt`, place them in `Data/` or pass their paths explicitly when invoking the scripts.

---

## Setup & Installation

1. Clone the repo and enter the directory:

```bash
git clone <your-repo-url>
cd linear-regression-sparcs
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
run the activation script in venv/Scripts (for example: venv/Scripts/activate)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

All scripts are run from the command line. Paths to `train.csv`, `test.csv`, and any auxiliary files (weights, lambdas) can be absolute or relative — place them in `Data/` or pass explicit paths.

### Part A — Weighted Linear Regression (linear.py)

**Objective**: Solve the weighted normal-equation analytically.

**Example** (weights file `Data/weights.txt` — one weight per training row):

```bash
python linear.py a Data/train.csv Data/test.csv Data/weights.txt predictions_a.csv weights_a.csv
```

**Arguments** (order may vary by script implementation — check the `linear.py` header comments if in doubt):

* `a` — mode for Part A
* `Data/train.csv` — training data
* `Data/test.csv` — test data
* `Data/weights.txt` — per-sample weights file (one value per line)
* `predictions_a.csv` — where test predictions will be saved
* `weights_a.csv` — learned weight vector (including bias if used)

### Part B — Ridge Regression with 10-Fold CV (linear.py)

**Objective**: Select the best regularization parameter lambda via 10-fold cross-validation and train the final ridge model.

**Example** (lambdas in `Data/lambdas.txt` — one lambda per line):

```bash
python linear.py b Data/train.csv Data/test.csv Data/lambdas.txt predictions_b.csv weights_b.csv best_lambda.txt
```

* `b` — mode for Part B
* `Data/lambdas.txt` — newline-separated candidate lambda values
* `best_lambda.txt` — file to which the chosen lambda will be written

Notes:

* The cross-validation implementation follows the assignment rule to exclude a small tail of rows from CV if required by the provided scripts. Consult `linear.py` comments for exact behavior.

### Part C — Competitive Model (Feature Engineering) (linear\_competitive.py)

**Objective**: Build a strong feature set (kept below \~300 features) and train a final Ridge model tuned for robustness.

**Run**:

```bash
python linear_competitive.py --train Data/train.csv --test Data/test.csv --out_test predictions_c.txt
```

This script applies the transformation pipeline (outlier filtering, OHE/encoding, polynomial terms, log transforms, feature limit) and fits a Ridge model. It saves test predictions to the path supplied with `--out_test`.

---

## Evaluation Script & Objective Function (how Part C is scored)

The Part C prediction file is evaluated with the following procedure (only the logic is summarized here):

1. Load the predictions (one value per test row) and the gold `Total Costs` values from a provided CSV.
2. Compute squared errors for each sample: `(y_true - y_pred)**2`.
3. Sort the squared errors and keep only the smallest 90% (i.e., drop the worst 10% errors).
4. Compute the mean of the retained squared errors and take the square root to obtain the RMSE on the best 90%.

Example output from the supplied evaluator:

```
Comment :=>> Objective Function obtained on the test set = 9675.008627275018
```

This is the metric reported for Part C and is implemented in the grader script included with the assignment.

---

## Implementation Notes

* **From-scratch implementation**: All parts (A, B, C) are implemented using core numerical routines written with **NumPy** and data handling via **pandas** — no `sklearn` model training functions are used. Closed-form analytic solutions (normal equations) are used where applicable.
* **Bias handling**: Most implementations prepend a column of ones to include the intercept term in the analytic solution.
* **Weighted OLS**: Part A performs weighted OLS using efficient broadcasting instead of forming large diagonal matrices explicitly to reduce memory pressure.
* **Ridge closed-form**: Ridge solutions are computed using the matrix expression `W = (X^T X + lambda I)^{-1} X^T Y`. Tiny numerical stabilizers or pseudo-inverse are used if the matrix is near-singular.
* **Feature pipeline**: Part C uses IQR-based outlier filtering on the target, selective one-hot encoding for low-to-medium-cardinality categoricals, polynomial features for important numeric columns, and transformations (e.g., log for skewed features). The final feature set is limited to keep it under \~300 dimensions.

---

## Evaluation & Metrics

* **Primary (Part C)**: RMSE computed on the **best 90%** of predictions (drop the 10% largest squared errors, compute RMSE on the remainder) — this is the objective used by the grader.
* **Part B CV**: Sum of MSEs across folds is used to select lambda.

---

## Troubleshooting

* If a required file (for example, `weights.txt`, `lambdas.txt`) is not present, create it and place it in `Data/` or pass its full path to the script.
* If you encounter a linear-algebra warning about a singular matrix, the code typically falls back to a small ridge term or `np.linalg.pinv`.
* Double-check the argument order when invoking `linear.py` — some helper scripts parse positional args in a fixed order.

---

## License & Contact

This project is released under the MIT License. For questions or feature requests, open an issue or contact the repository owner.

---

*Happy modeling!* 🎯
