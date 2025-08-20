# Logistic Regression for Hospital Discharge Classification

A NumPy-from-scratch implementation of logistic regression for hospital inpatient discharge records. This repository contains three parts:

* **Part A**: Weighted multiclass logistic regression (4 classes — predicting patient Race) with multiple learning-rate strategies.
* **Part B**: Hyperparameter tuning (competitive) to minimize the weighted loss on a test set within a 10-minute time limit.
* **Part C**: Competitive binary classification (predicting patient Gender) with feature engineering and selection (15-minute time limit).

---

## Repository structure

```
.
├── logistic.py                 # Main script for Part A and Part B
├── logistic_competitive.py     # Main script for the competitive Part C
├── Data/
│   ├── train1.csv              # Training data for Part A/B
│   ├── test1.csv               # Test data for Part A/B
│   ├── train2.csv              # Training data for Part C
│   ├── test2.csv               # Test data for Part C
│   └── mapping.json            # JSON file with categorical feature mappings
└── README.md                   # This README file
```

---

## Highlights

* Fully implemented logistic regression using **NumPy** (no scikit-learn/TensorFlow for the core model).
* Supports **multiclass** (softmax) and **binary** (sigmoid) classification.
* **Weighted cross-entropy** loss to handle class imbalance.
* **Mini-batch gradient descent** with three learning-rate strategies:

  * Constant learning rate
  * Adaptive (decaying) learning rate: `eta_t = eta_0 / (1 + k * t)`
  * Exact line search per-step learning rate (implemented with both Ternary Search and Golden-Section Search)
* Competitive part (Part C) includes one-hot encoding, standardization, ANOVA F-score feature selection, and a pipeline for fast experimentation.

---

## Requirements

* Python 3.8+
* numpy
* pandas
* scikit-learn

Install with pip:

```bash
pip install numpy pandas scikit-learn
```

---

## Core model & loss

The multiclass weighted loss used for Parts A/B is the weighted negative log-likelihood (cross-entropy):

$L(w) = -\frac{2}{n} \sum_{i=1}^{n} \sum_{j=1}^{k} \mathbb{I}(y^{(i)}=j) \frac{1}{\text{freq}_j} \log p_j^{(i)}$

* `n` is the number of samples in a batch.
* `k` is the number of classes (4 for Part A/B).
* `freq_j` is the frequency of class `j` computed from the entire training set (used to weight samples inversely to class frequency).
* `p_j^{(i)}` is the softmax probability for class `j` of sample `i`.

The implementation uses vectorized NumPy operations for efficiency. Gradients are computed analytically from the weighted loss and applied in a mini-batch gradient descent loop.

---

## Scripts & Usage

### Part A — Train weighted multiclass logistic regression

This runs the training routine for Part A and saves final weights.

```bash
python3 logistic.py a train1.csv params.txt modelweights.txt
```

* `a` — run Part A.
* `train1.csv` — training CSV for Parts A/B.
* `params.txt` — text file containing hyperparameters (format described below).
* `modelweights.txt` — file path to save learned model weights (NumPy text or binary format as implemented).

**Example `params.txt` format (one key=value per line):**

```
learning_strategy=constant
eta=0.01
epochs=100
batch_size=64
adaptive_k=0.001         # only used for adaptive strategy
line_search_method=golden # only for exact line search: 'ternary' or 'golden'
seed=42
```

### Part B — Competitive prediction (multiclass)

Train and produce predictions on the provided test set.

```bash
python3 logistic.py b train1.csv test1.csv modelweights.txt modelpredictions.csv
```

* `b` — run Part B.
* `test1.csv` — test CSV.
* `modelpredictions.csv` — output CSV containing predicted class probabilities (softmax) or predicted labels depending on script mode.

**Notes:** Part B is intended for fast hyperparameter tuning; the script includes utilities to report loss on hold-out folds and to export per-class confusion statistics.

### Part C — Competitive binary classification (feature engineering + training)

This script runs the full binary-class pipeline: encoding, scaling, feature selection, training, and prediction.

```bash
python3 logistic_competitive.py train2.csv test2.csv output.txt
```

* `train2.csv` — training CSV for the binary task.
* `test2.csv` — test CSV for the binary task.
* `output.txt` — predictions written as `-1` or `1` (one per line) as required by the competition.

**Pipeline steps in `logistic_competitive.py`:**

1. Load data and mapping file (if present) for categorical encoding.
2. One-hot encode categorical features; drop the first category to avoid multicollinearity.
3. Standardize numeric features using `StandardScaler` (scikit-learn).
4. Score features with `f_classif` (ANOVA F-test) and select top-`k` features. The script accepts `k` as a parameter or uses a default.
5. Train binary logistic regression (NumPy implementation) and write test predictions.

---

## Hyperparameter tuning tips

* For the **adaptive learning rate**, tuning `eta_0` and decay factor `k` is crucial. A typical strategy is to start with `eta_0` in `[0.01, 0.1]` and `k` small (e.g., `1e-3`).
* For **exact line search**, Golden-Section is generally more robust; Ternary Search can be faster when the loss along the line is strictly unimodal.
* Use **mini-batches** (e.g., 32–256) for stable convergence and to speed up computation.
* Set a fixed random seed for reproducibility (training script reads `seed` from `params.txt`).

---

## Feature engineering notes (Part C)

* **One-hot encoding**: Categorical variables are one-hot encoded and the first level is dropped (dummy encoding) to prevent singularities.
* **Scaling**: Standardization is important — gradient descent converges faster and more stably when features are centered and have similar scales.
* **Feature selection**: `f_classif` gives per-feature F-scores and p-values; select features with the highest scores and statistically significant p-values to reduce dimensionality and improve generalization.

---

## Evaluation & Outputs

* For multiclass tasks (A/B), the scripts can produce:

  * Per-class predicted probabilities (softmax) in CSV format.
  * Confusion matrix and per-class accuracy (printed to stdout or saved to a file).
* For the binary task (C), the output is a plain text file with `-1`/`1` labels (one per line) suitable for submission to the competition.

### Evaluation script used for Part B

We include the grader used for Part B which expects a CSV of predicted **probabilities** with 4 columns (no header). Each row corresponds to a test sample and the four columns are the predicted softmax probabilities for classes `1,2,3,4` respectively. The evaluator checks the shape and computes the weighted loss using class frequencies from the test set.

A simplified version of the evaluation logic (used in the project) is shown below:

```python
import sys
import numpy as np
import os
import pandas as pd

predicted_csv = sys.argv[1]
gold_csv = sys.argv[2]

def comment(s):
    print('Comment :=>> ' + s)

if os.path.exists(predicted_csv) == False:
    comment("Prediction csv not created for part (b)")
    exit()

# Load predictions (n x 4) and gold labels
y_pred = np.genfromtxt(predicted_csv, delimiter=',', dtype=None)
df_gold = pd.read_csv(gold_csv)
y_true = df_gold['Race'].to_numpy()

# Basic shape checks
if (y_true.shape[0] != y_pred.shape[0] or y_pred.shape[1] != 4):
    comment("Prediction file of wrong dimensions for part (b)")
    exit()

# Computing frequencies of each class in the test set
freq = [np.count_nonzero(y_true == k) for k in [1,2,3,4]]

# Compute weighted loss
loss = 0.0
eps = 1e-12
for i in range(y_true.shape[0]):
    true_label = int(y_true[i])
    probability = y_pred[i][true_label-1]
    l = np.log(probability + eps) / freq[true_label-1]
    loss += l

loss = -loss / (2 * y_true.shape[0])

comment("Loss obtained on the test set for part (b): " + str(loss))
```

**Notes on format & checks:**

* `predicted_csv` must exist and contain `n` rows and **4** columns (no header); each row should represent softmax probabilities for classes 1–4.
* The evaluator checks that the number of prediction rows matches the number of gold samples and that there are exactly 4 columns.
* The loss is computed using the per-class frequencies in the test set as shown in the formula above.

**Reported result (example):**

```
Comment :=>> Loss obtained on the test set for part (b): 3.469833358584122e-05
```

### Evaluation script used for Part C

We include an evaluation helper (used by the grader) which compares the predicted labels file against the gold CSV. The expected prediction file is a plain text CSV with a single column (no header) containing one integer label per line. The gold CSV contains a `Gender` column with integer labels.

A simplified version of the evaluation logic (used in the project) is described below:

```python
import sys
import numpy as np
import pandas as pd

predicted_txt = sys.argv[1]
gold_csv = sys.argv[2]

# Load predictions and gold labels
y_pred = np.genfromtxt(predicted_txt, delimiter=',', dtype=None)
df_gold = pd.read_csv(gold_csv)
y_true = df_gold['Gender'].to_numpy()

# Basic shape checks
if (y_true.shape[0] != y_pred.shape[0] or y_pred.ndim != 1):
    print('Comment :=>> Prediction file of wrong dimensions for part (c)')
    sys.exit()

# Compute accuracy
correct = int(np.sum(y_pred.astype(int) == y_true.astype(int)))
total_samples = y_true.shape[0]
accuracy = correct / total_samples
print('Comment :=>> Accuracy obtained on the test set for part (c): ' + str(accuracy))
```

**Notes on format & checks:**

* `predicted_txt` must exist and contain exactly one integer prediction per line (no header).
* The evaluator checks that the number of predictions matches the number of gold samples and that predictions are a 1-D array.
* If the checks fail, the evaluator prints a `Comment :=>>` message and exits.

### Reported result (example)

The evaluation for Part C produced the following accuracy on the provided test set (printed by the evaluation script):

```
Comment :=>> Accuracy obtained on the test set for part (c): 65.69%
```

This project (including all training and prediction code) is implemented from scratch using **NumPy** and **pandas** for data processing. Scikit-learn is used only in Part C for optional preprocessing utilities (e.g. `StandardScaler` and `f_classif` for feature scoring) and is **not** used for the core logistic regression training.

---

## Implementation details & internals

* All core matrix math and optimization loops are implemented using NumPy for transparency and learning purposes.
* The weighted loss uses class frequencies computed from the entire training set to form inverse-frequency weights.
* Gradients are derived analytically and implemented in a fully vectorized manner.
* The exact line search treats the learning rate as a single scalar along the negative gradient direction and numerically minimizes the loss along that line using the selected 1-D search method.

---

## Reproducibility

* Use the `seed` parameter in `params.txt` to fix NumPy's RNG for deterministic minibatch shuffling and weight initialization.
* Save trained weights with `modelweights.txt` to reproduce predictions later.

---

## Troubleshooting & common pitfalls

* If training loss does not decrease:

  * Try reducing the initial learning rate.
  * Increase regularization (if implemented) or reduce batch size.
  * Verify features are standardized (especially when using adaptive or exact line search).
* If numerical instability occurs in softmax:

  * Ensure the implementation uses the standard `max-trick` (subtract max logit per-row before exponentiating).

---

## Extending the project

* Add `L2` regularization to the loss and gradient to control overfitting.
* Implement stochastic (single-sample) gradient descent variants or momentum-based optimizers (e.g., RMSProp, Adam) if you want faster convergence.
* Create a small `sklearn` wrapper to compare the NumPy implementation against `sklearn.linear_model.LogisticRegression` for correctness testing.

---

## Author & License

* Author: (your name or team)
* License: MIT (or choose an appropriate license)

---

If you'd like, I can also generate a simple example `params.txt`, a minimal `train1.csv` mock, or convert this README to a `README.md` file you can download. Just tell me which one you want.
