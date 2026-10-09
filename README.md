# Bank Customer Churn Prediction

Predicting which bank customers will close their accounts. A binary classification project built for a Kaggle competition, covering EDA, leakage-free preprocessing, model comparison with cross-validation, and a probability calibration check.

## Highlights

- **0.933 ROC-AUC** (5-fold stratified CV) with Gradient Boosting
- **Well-calibrated probabilities**: in every risk decile, predicted and actual churn rates differ by at most 1.2 percentage points
- **Business value**: contacting the riskiest 20% of customers would reach **≈74% of all churners**

| Metric | Score |
|---|---|
| CV ROC-AUC (Gradient Boosting) | 0.9329 ± 0.0033 |
| Brier score (baseline 0.1591) | 0.0749 |
| Log loss (baseline 0.4983) | 0.2489 |
| Kaggle public leaderboard | _TBD_ |

## Problem

Each row is a bank customer. The target `Exited` is **1 if the customer left the bank** (churn) and 0 if they stayed.

The task wording referred to loan repayment. However, the dataset has no loan amount, payment history or default flag. The label describes account closure, so the model predicts **churn**. `CreditScore` is an input feature, not the outcome.

Submissions are churn **probabilities** (the sample submission uses 0.5 placeholders), not 0/1 labels.

## Data

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 15,000 | 14 |
| `test.csv` | 10,000 | 13 (no `Exited`) |

**Features:** `CreditScore`, `Geography` (France / Germany / Spain), `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
**Identifiers:** `id`, `CustomerId`, `Surname`
**Class balance:** 80.2% stayed, 19.8% churned

### Data quality checks

- No missing values in train or test, and category values are identical in both
- No duplicate rows; integer columns are stored as float but contain no fractional values
- `CustomerId` has only 6,306 unique values across 15,000 rows. The data is synthetic, so identifiers carry no meaning and are dropped
- `NumOfProducts`: one train row with 5 and two test rows with 6 (the original dataset has 1–4)
- `EstimatedSalary`: one train row at 885k; every other value in train and test is ≤ 200k
- `CreditScore` ranges from 431 to 850; train and test distributions match closely

## Key EDA findings

Churn rate by group (train set):

| Feature | Groups | Churn rate |
|---|---|---|
| `NumOfProducts` | 1 / 2 / 3+ | 38.0% / **4.2%** / ≈94% |
| `Age` | 18–30 / 31–40 / 41–50 | 4.1% / 9.7% / **40.6%** |
| `Geography` | Germany vs France / Spain | **40.6%** vs 15.4% / 15.2% |
| `Gender` | Female vs Male | 27.6% vs 13.9% |
| `IsActiveMember` | No vs Yes | 27.3% vs 12.3% |
| `HasCrCard` | No vs Yes | 20.6% vs 19.6% (no signal) |

- **`NumOfProducts` is the strongest and non-monotonic signal.** Customers with 2 products almost never leave, while those with 3+ almost always do. It is one-hot encoded so that linear models can capture this shape.
- **`Balance` is confounded with `Geography`.** Customers with a non-zero balance churn more (28.8% vs 15.1%). However, 100% of German customers have a non-zero balance, compared with about 20% in France and Spain. Much of the raw `Balance` effect reflects Germany's higher churn rate, not the balance itself.

## Preprocessing

A single `prepare()` function is applied identically to train and test. It applies fixed rules and learns nothing from the data, so there is no leakage.

1. Drop `id`, `CustomerId`, `Surname`
2. Cast integer columns from float to int
3. Clip `NumOfProducts` at 3, which merges 3, 4, 5 and 6 into one "3+" group. A rule rather than a value-specific fix also handles the 6s that appear only in test
4. Clip `EstimatedSalary` at 200,000
5. Add a `Balance_zero` flag
6. Encode `Gender` as binary; one-hot encode `Geography` and `NumOfProducts`

Result: **13 features**. No rows are dropped, because every test customer needs a prediction. Scaling is applied only for Logistic Regression, inside a `Pipeline`.

## Modeling

Models are compared with `StratifiedKFold` (5 folds, shuffled, `random_state=42`) on ROC-AUC:

| Model | ROC-AUC |
|---|---|
| Logistic Regression (scaled) | 0.9206 ± 0.0039 |
| Random Forest | 0.9310 ± 0.0028 |
| **Gradient Boosting** | **0.9329 ± 0.0033** |
| HistGradientBoosting | 0.9305 ± 0.0033 |

The three tree models are within one standard deviation of each other, so the differences are mostly noise. Gradient Boosting was selected and retrained on the full training set.

`class_weight` is intentionally **not** used. The submission consists of probabilities, and reweighting classes would inflate them and break calibration.

## Probability calibration

ROC-AUC measures ranking only, so calibration was checked separately. Out-of-fold probabilities (`cross_val_predict`) were split into 10 equal-size groups:

| Predicted probability range | Mean predicted | Actual churn | Customers |
|---|---|---|---|
| 0.006 – 0.011 | 0.009 | 0.002 | 1,503 |
| 0.011 – 0.015 | 0.013 | 0.005 | 1,497 |
| 0.015 – 0.020 | 0.017 | 0.008 | 1,502 |
| 0.020 – 0.029 | 0.024 | 0.025 | 1,498 |
| 0.029 – 0.046 | 0.036 | 0.033 | 1,500 |
| 0.046 – 0.085 | 0.063 | 0.051 | 1,500 |
| 0.085 – 0.168 | 0.122 | 0.133 | 1,500 |
| 0.168 – 0.370 | 0.255 | 0.265 | 1,500 |
| 0.370 – 0.751 | 0.553 | 0.559 | 1,500 |
| 0.751 – 0.993 | 0.891 | 0.903 | 1,500 |

- The calibration curve lies almost on the diagonal: a predicted 0.25 really means about 25 out of 100 similar customers leave.
- The model slightly overestimates risk in the three lowest-risk groups (predicts 0.9–1.7%, actual 0.2–0.8%). The gap is about one percentage point and does not change the ranking, so no recalibration was applied.
- Brier score and log loss are both about half of a naive baseline that predicts 19.8% for every customer.

## Business view

Ranking customers by predicted risk (approximate, from the calibration table):

| Customers contacted | Share of all churners reached |
|---|---|
| Riskiest 10% | ≈45% |
| Riskiest 20% | ≈74% |
| Riskiest 30% | ≈87% |
| Lowest-risk 50% | contain only ≈4% of churners |

A retention team could focus on one customer in five and still reach roughly three quarters of the customers who are about to leave.

## How to run

**On Kaggle:** open the notebook with the competition data attached and choose *Run All*. The notebook writes `/kaggle/working/submission.csv`.

**Locally:**

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page, set `input_path` in the first cell to their folder, and run the notebook top to bottom.

## Repository structure

```
├── bank-churn-prediction.ipynb   # EDA, preprocessing, modeling, calibration, submission
└── README.md
```

## Next steps

- Tune Gradient Boosting hyperparameters with `GridSearchCV`
- Try a soft-voting ensemble of the tree models and Logistic Regression
- Compare with LightGBM and CatBoost
- Add SHAP values to explain individual predictions

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · Kaggle Notebooks

## Author

**Abdurahmon Qodirov**
[GitHub](https://github.com/abdurrohmanq) · [LinkedIn](https://linkedin.com/in/abdurrohman-qodirov)
