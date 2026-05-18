# 💳 Financial Fraud Detection — ML Project

## Overview

This project builds a machine learning pipeline to detect fraudulent financial transactions using the **AIML Dataset**. It covers end-to-end steps: exploratory data analysis (EDA), feature engineering, preprocessing, model training, and evaluation.

---

## Dataset

**File:** `AIML Dataset.csv`

| Feature | Description |
|---|---|
| `step` | Time step of the transaction |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.) |
| `amount` | Transaction amount |
| `nameOrig` | Originating account ID |
| `oldbalanceOrg` | Balance before transaction (sender) |
| `newbalanceOrig` | Balance after transaction (sender) |
| `nameDest` | Destination account ID |
| `oldbalanceDest` | Balance before transaction (receiver) |
| `newbalanceDest` | Balance after transaction (receiver) |
| `isFraud` | Target label — 1 if fraudulent, 0 otherwise |
| `isFlaggedFraud` | System-flagged fraud indicator |

**Shape:** 6,362,620 rows × 11 columns | No missing values

---

## Key Findings from EDA

- **Fraud rate:** Only ~0.13% of all transactions are fraudulent — highly imbalanced dataset.
- **Fraud types:** Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transactions.
- **Amount distribution:** Transaction amounts are right-skewed; log transformation used for visualization.
- **Balance pattern:** Many fraud cases show the originating account balance going to exactly 0 after the transaction (~1.18M such cases in TRANSFER/CASH_OUT).
- **Top receivers:** Some destination accounts appear in 97–113 transactions, suggesting potential mule accounts.
- **Correlation:** `amount` has weak positive correlation with `isFraud` (0.077); most balance columns are highly correlated with each other but not strongly with fraud.

---

## Feature Engineering

Two new features derived from balance columns:

```python
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
```

Columns dropped before modeling: `step`, `nameOrig`, `nameDest`, `isFlaggedFraud`

---

## Modeling

### Pipeline

```
ColumnTransformer
├── StandardScaler       → numeric features
└── OneHotEncoder        → categorical feature (type)
        ↓
LogisticRegression (class_weight="balanced", max_iter=1000)
```

### Train/Test Split

- Test size: 30%
- Stratified on `isFraud` to preserve class ratio

### Numeric Features Used

`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`

### Categorical Feature

`type` (one-hot encoded, first category dropped)

---

## Results

| Metric | Non-Fraud (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.02 |
| Recall | 0.95 | 0.94 |
| F1-Score | 0.97 | 0.04 |

**Overall Accuracy:** ~94.5%

> ⚠️ **Note:** Despite high accuracy, precision for fraud is very low (2%) — meaning many false positives. This is a known trade-off when using `class_weight="balanced"` with extreme class imbalance. Consider advanced methods (Random Forest, XGBoost, SMOTE) for better precision-recall balance.

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 1,802,260 | 104,062 |
| Actual 1 | 148 | 2,316 |

---

## Saved Model

The trained pipeline is saved using `joblib`:

```python
import joblib
joblib.dump(pipeline, "fraud_detection_pipeline.pkl")

# To load and predict:
pipeline = joblib.load("fraud_detection_pipeline.pkl")
pipeline.predict(X_new)
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## How to Run

1. Place `AIML Dataset.csv` in the project directory.
2. Run the notebook or script top to bottom.
3. The trained model will be saved as `fraud_detection_pipeline.pkl`.

---

## Possible Improvements

- Use ensemble models (Random Forest, XGBoost, LightGBM) for better fraud precision.
- Apply SMOTE or other oversampling techniques to handle class imbalance.
- Add the engineered balance features (`balanceDiffOrig`, `balanceDiffDest`) to the model input.
- Tune decision threshold to optimize for recall vs. precision based on business needs.
- Incorporate `nameOrig`/`nameDest` frequency encoding as features.
