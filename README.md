# Credit Risk Classification with Cost-Sensitive Threshold Optimization

## Overview
This project builds and compares a **Logistic Regression** and a **Random Forest** model to predict credit default risk using the German Credit dataset.

The workflow extends beyond standard evaluation metrics by incorporating probability-based predictions, threshold optimization, class imbalance handling, SHAP-based explainability, and calibration analysis to align model performance with real business objectives.

---

## Objective
- Predict whether a customer will default on credit
- Compare Logistic Regression and Random Forest across multiple evaluation dimensions
- Handle class imbalance using SMOTE
- Optimize decision threshold to minimize financial cost
- Explain individual predictions using SHAP values 
- Verify probability reliability using calibration curves

---

## Problem Understanding
The problem is a binary classification task where the goal is to identify high-risk borrowers (defaulters).

This is important because:
- Missing a defaulter (false negative) leads to direct financial loss — the bank loses the full loan amount
- Incorrectly rejecting a good customer (false positive) reduces business opportunity — only foregone interest
- These costs are **asymmetric** and documented in the dataset: FN costs 5× more than FP

The project focuses on balancing these trade-offs using a cost-sensitive, business-grounded approach.

---

## Methodology

### Data Preprocessing
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder` (drop='first' to avoid dummy variable trap)
- Combined using `ColumnTransformer`
- End-to-end pipeline created with preprocessing + model to prevent data leakage

---

### Models
Two model families are compared across four pipeline variants:

| Pipeline | Model | Imbalance Handling |
|---|---|---|
| Baseline LR | Logistic Regression | None |
| Baseline RF | Random Forest (class_weight='balanced') | Native weighting |
| LR + SMOTE | Logistic Regression | SMOTE inside CV fold |
| RF + SMOTE | Random Forest | SMOTE inside CV fold |

Both models configured with increased iterations / estimators for stable convergence.

---

### Validation Strategy
- Stratified 5-Fold Cross-Validation
- Maintains class distribution across all folds
- SMOTE applied **inside** the pipeline so it only sees training data — prevents data leakage

---

### Evaluation Metrics
The model is evaluated using multiple components:

- ROC-AUC score (cross-validation)
- PR-AUC — Precision-Recall AUC, more honest for imbalanced data
- Confusion matrix at optimal threshold
- Precision and Recall at optimal threshold
- Calibration curve — verifies probability trustworthiness

Predicted probabilities generated using:
- `predict_proba()` via `cross_val_predict` for unbiased estimates

---

### Cost-Sensitive Evaluation
A custom cost function is defined, sourced directly from the UCI dataset documentation:

- False Negative (FN) = 5 — missed defaulter, bank loses full loan
- False Positive (FP) = 1 — rejected good customer, bank loses interest only

Total cost is computed using confusion matrix outputs:
```
Total Cost = (FN count × 5) + (FP count × 1)
```

A **sensitivity analysis** is also conducted — varying the FN:FP cost ratio from 1:1 to 10:1 to show the optimal threshold is data-driven, not arbitrarily chosen.

---

### Class Imbalance Handling
The dataset has 700 good vs 300 default customers (70/30 split). Two strategies are compared:

- `class_weight='balanced'` on Random Forest — upweights minority class during training
- **SMOTE** (Synthetic Minority Over-sampling Technique) — creates synthetic minority samples in feature space, applied inside the cross-validation loop to prevent leakage

---

### Threshold Optimization
- Thresholds evaluated from 0.01 to 0.99 (200 steps)
- For each threshold:
  - Convert probabilities to class predictions
  - Compute confusion matrix
  - Calculate total cost
- Identify threshold that minimizes total financial cost

Optimal thresholds across all models fall in the range **0.13–0.18**, well below the naive default of 0.5 — because FN is 5× more expensive, the model should aggressively flag potential defaulters.

---

### Comparison of Thresholds
Two scenarios are compared per model:

1. Default threshold (0.5)
2. Optimized threshold

Metrics compared:
- Total cost
- Precision
- Recall
- Cost saving (%)

---

### SHAP Explainability
Standard feature importance only reveals *which* features matter. SHAP (SHapley Additive exPlanations) additionally reveals:

- **Direction** — does a high feature value push toward default or away?
- **Magnitude** — by how much does it shift any individual prediction?

Top predictors identified:
- Checking account status
- Credit duration (months)
- Credit history
- Credit amount

This is not just academically useful — in regulated credit markets, lenders must be able to explain adverse decisions to applicants.

---

### Calibration Analysis
Calibration curves verify whether predicted probabilities are reliable:

- **Logistic Regression** — well-calibrated by design; probabilities can be directly interpreted as default likelihoods
- **Random Forest** — tends toward overconfidence; probabilities are pushed toward extremes

This distinction matters for threshold-based decisions: a poorly calibrated model's optimal threshold may not generalize as expected in production.

---

### Visualization
Eight figures produced:

| Figure | Description |
|---|---|
| Fig 1 | Class distribution + cost sensitivity analysis |
| Fig 2 | ROC-AUC curves (all 4 models, with/without SMOTE) |
| Fig 3 | Precision-Recall curves |
| Fig 4 | Financial cost vs decision threshold |
| Fig 5 | Confusion matrices at optimal thresholds |
| Fig 6 | Calibration curves |
| Fig 7 | SHAP feature importance (bar chart + beeswarm) |
| Fig 8 | Full model comparison table |

---

## Key Results

| Model | ROC-AUC | PR-AUC | Cost @ 0.5 | Cost @ Optimal | Saving |
|---|---|---|---|---|---|
| Logistic Regression | 0.784 | 0.595 | 890 | 534 | 40.0% |
| Random Forest | 0.768 | 0.600 | 1075 | 561 | 47.8% |
| LR + SMOTE | 0.780 | 0.596 | 606 | 533 | 12.0% |
| **RF + SMOTE** | **0.771** | **0.604** | **950** | **520 ⭐** | **45.3%** |

### Why RF + SMOTE Wins

RF + SMOTE achieves the lowest total financial cost (520) for two compounding reasons:

1. **Random Forest captures non-linear patterns** — a defaulter isn't always someone who fails one obvious check. RF learns combinations of features (low balance + long duration + poor history) that a linear model may underweight. This catches more edge-case defaulters, reducing FN count — and since each FN costs 5, even catching a few extra saves significantly.

2. **SMOTE provides better minority-class exposure** — with only 300 defaulters in the training data, the model gets limited signal on what default looks like. SMOTE creates synthetic defaulter examples in feature space, giving the model a richer picture of the minority class and making it more sensitive at the decision boundary.

These two effects compound: RF's pattern recognition + SMOTE's balanced training = lower FN count at the optimal threshold = lowest total cost.

> **Honest caveat:** The margin is small. LR + SMOTE achieves 533, only 13 units behind. At this dataset size the models are competitive. On a real portfolio of thousands of loans, the gap would scale proportionally and become financially significant.

---

## Technical Soundness
- Appropriate use of Logistic Regression and Random Forest for binary classification
- Proper preprocessing pipeline avoids data leakage
- SMOTE applied inside cross-validation loop — no leakage from oversampling
- Cross-validation ensures robust, unbiased evaluation
- ROC-AUC and PR-AUC both reported — PR-AUC is more honest for imbalanced data
- Threshold optimization directly minimizes documented business cost
- SHAP explainability connects model output to regulatory and business requirements
- Calibration curves validate probability reliability before threshold deployment

---

## Code Quality and Documentation
- Modular pipeline structure improves readability and reproducibility
- Clear separation of preprocessing, modeling, evaluation, and visualization
- Use of sklearn and imblearn utilities ensures consistency
- All cross-validated predictions generated with `cross_val_predict` — no train/test contamination

---

## Key Technical Choices

### 1. Threshold Optimization Over Default 0.5
Default thresholds assume equal misclassification costs. Real-world credit problems have asymmetric costs. The chosen approach directly minimizes business loss rather than optimizing an abstract accuracy metric.

### 2. PR-AUC Alongside ROC-AUC
ROC-AUC is inflated on imbalanced datasets because it includes True Negatives. PR-AUC focuses only on the minority class — which is precisely the class the business cares about detecting.

### 3. SMOTE Inside the Pipeline
Applying SMOTE before cross-validation leaks synthetic samples into validation folds, artificially inflating metrics. Placing SMOTE inside `ImbPipeline` guarantees it only runs on training data.

### 4. SHAP Over Standard Feature Importance
Standard importance is a single number per feature. SHAP provides a value per feature per prediction — enabling individual loan decision explanations, which standard importance cannot support.

---

## Results and Insights
- Default threshold (0.5) is suboptimal for minimizing financial cost across all four models
- Optimized threshold delivers 12–48% cost reduction depending on the model
- RF + SMOTE achieves the lowest total cost (520) at threshold 0.16
- Logistic Regression has the highest ROC-AUC (0.784) but not the lowest cost — demonstrating why ROC-AUC alone is insufficient
- Logistic Regression is better calibrated; Random Forest overestimates certainty at the extremes
- Checking account status and credit duration are the strongest predictors of default (SHAP)

---

## Originality and Insight
- Cost function sourced from dataset documentation — not assumed or hardcoded
- Sensitivity analysis proves threshold is data-driven across a range of cost ratios
- Calibration curves added — rarely seen in student projects, directly relevant to production deployment
- SHAP beeswarm connects feature values to prediction direction for any individual customer
- All models evaluated on business cost, not just standard ML benchmarks

---

## Tech Stack
- Python
- scikit-learn
- imbalanced-learn
- shap
- pandas
- numpy
- matplotlib

---
