# Automated ML Data Preprocessor

- **Author:** Animireddy Srinu
- **Program:** GetIntern.in 8-Week Practical Data Science & ML Internship
- **Phase 2 / Week 5 Milestone Project:** Feature Encoding & Train-Test Splitting

## Project Summary
An automated preprocessing pipeline built with **Pandas and scikit-learn** that transforms the mixed text and numeric **Telco Customer Churn** dataset (7,043 customers, 21 columns) into clean, encoded, scaled and model-ready train and test sets.

The pipeline covers three objectives:
1. **Categorical feature encoding** with `LabelEncoder` and `pd.get_dummies()`
2. **Z-score feature scaling** with `StandardScaler`
3. **Train-test partitioning** (80/20) with `train_test_split(test_size=0.2, random_state=42)`

The processed partitions are exported as `X_train.csv`, `X_test.csv`, `y_train.csv` and `y_test.csv` for model training in Week 6.

## Repository Contents
| File | Description |
|------|-------------|
| `automated_ml_data_preprocessor.ipynb` | Fully executed notebook with outputs and markdown explanations |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Raw dataset (7,043 rows, 21 columns) |
| `X_train.csv` | Training features (5,634 rows, 30 columns) |
| `X_test.csv` | Test features (1,409 rows, 30 columns) |
| `y_train.csv` | Training target (5,634 rows) |
| `y_test.csv` | Test target (1,409 rows) |
| `README.md` | Project documentation |

## Dataset
Source: *Telco Customer Churn* (Kaggle) - https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The target is `Churn` (5,174 customers stayed, 1,869 churned).

## Data Cleaning
- No duplicate rows were found.
- `TotalCharges` was stored as text and had 11 blank values. These belong to new customers with tenure 0 who have not been billed yet, so they were converted to numbers and filled with 0.
- Text columns were stripped of extra spaces.
- `customerID` was dropped because it is a unique label with no predictive value.
- After cleaning: 7,043 rows, 20 columns, 0 nulls.

## Encoding Choices
| Column type | Columns | Method | Reason |
|-------------|---------|--------|--------|
| Target | `Churn` | `LabelEncoder` (No = 0, Yes = 1) | Converts the label to numbers |
| Binary text (5) | `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `LabelEncoder` | Only two values, so 0/1 is a natural fit |
| Multi-category text (10) | `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod` | `pd.get_dummies(drop_first=True)` | One-hot flags avoid a false order (for example Month-to-month < One year < Two year) |
| Already numeric | `SeniorCitizen` | Kept as is | Already 0/1 |

`drop_first=True` removes one redundant flag per feature, because it can be inferred from the others. Encoding the 10 multi-category columns turned 20 columns into **30 numeric features**.

**Binary mappings:** gender (Female = 0, Male = 1); Partner, Dependents, PhoneService and PaperlessBilling (No = 0, Yes = 1).

## Train-Test Split
- `train_test_split(test_size=0.2, random_state=42)` gives **5,634 training** rows and **1,409 test** rows.
- The churn rate is almost identical in both parts (26.55% in train and 26.47% in test), so the split is representative.
- The data was split **before** scaling, so the scaler learns its mean and standard deviation from the training set only and then applies them to the test set. No test information leaks into training.

## Scaling Summary
`StandardScaler` converts `tenure`, `MonthlyCharges` and `TotalCharges` to z-scores (mean 0, standard deviation 1). It was fitted on the training set only.

| Column | Original mean (train) | Original std (train) |
|--------|----------------------:|---------------------:|
| tenure | 32.37 | 24.42 |
| MonthlyCharges | 64.86 | 30.09 |
| TotalCharges | 2287.09 | 2263.00 |

**Check after scaling**

| Column | Train mean | Train std | Test mean | Test std |
|--------|-----------:|----------:|----------:|---------:|
| tenure | 0.0 | 1.0 | -0.0004 | 1.0274 |
| MonthlyCharges | 0.0 | 1.0 | -0.0170 | 1.0001 |
| TotalCharges | 0.0 | 1.0 | -0.0162 | 1.0079 |

The training columns have a mean of 0 and a standard deviation of 1. The test columns are close but not exact, which is expected because they use the training statistics.

## Before / After Shape Comparison
| Stage | Rows | Columns |
|:------|-----:|--------:|
| Raw dataset | 7043 | 21 |
| After cleaning (ID dropped) | 7043 | 20 |
| Encoded features (X) | 7043 | 30 |
| X_train | 5634 | 30 |
| X_test | 1409 | 30 |
| y_train | 5634 | 1 |
| y_test | 1409 | 1 |

## Feature Table
| Original column | Type | Method | New columns |
|:----------------|:-----|:-------|------------:|
| gender | binary | LabelEncoder | 1 |
| Partner | binary | LabelEncoder | 1 |
| Dependents | binary | LabelEncoder | 1 |
| PhoneService | binary | LabelEncoder | 1 |
| PaperlessBilling | binary | LabelEncoder | 1 |
| MultipleLines | multi-category | get_dummies (drop_first) | 2 |
| InternetService | multi-category | get_dummies (drop_first) | 2 |
| OnlineSecurity | multi-category | get_dummies (drop_first) | 2 |
| OnlineBackup | multi-category | get_dummies (drop_first) | 2 |
| DeviceProtection | multi-category | get_dummies (drop_first) | 2 |
| TechSupport | multi-category | get_dummies (drop_first) | 2 |
| StreamingTV | multi-category | get_dummies (drop_first) | 2 |
| StreamingMovies | multi-category | get_dummies (drop_first) | 2 |
| Contract | multi-category | get_dummies (drop_first) | 2 |
| PaymentMethod | multi-category | get_dummies (drop_first) | 3 |
| tenure | numeric | StandardScaler (z-score) | 1 |
| MonthlyCharges | numeric | StandardScaler (z-score) | 1 |
| TotalCharges | numeric | StandardScaler (z-score) | 1 |
| SeniorCitizen | already 0/1 | kept as is | 1 |
| **Total** | | | **30** |

## Key Points
- All 30 features are numeric with no missing values, so the data is ready for model training.
- One-hot encoding was used for multi-category columns to avoid implying a false order.
- Splitting before scaling prevents data leakage from the test set.
- The exported files are saved without an index column, and the rows of `X_train` and `y_train` (and of `X_test` and `y_test`) are in matching order.

## How to Run
```bash
pip install pandas numpy scikit-learn jupyter tabulate
jupyter notebook automated_ml_data_preprocessor.ipynb
```
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder and run all cells from top to bottom.

## Tech Stack
Python 3, Pandas, NumPy, scikit-learn, Jupyter Notebook
