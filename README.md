# Credit Risk Classification with Cost-Sensitive Threshold Optimization

## Overview
This project builds a Logistic Regression model to predict credit default risk using the German Credit dataset.  

The workflow extends beyond standard evaluation metrics by incorporating probability-based predictions and threshold optimization to align model performance with business objectives.

---

## Objective
- Predict whether a customer will default on credit  
- Evaluate model performance using appropriate classification metrics  
- Optimize decision threshold to minimize financial cost  

---

## Problem Understanding
The problem is a binary classification task where the goal is to identify high-risk borrowers (defaulters).  

This is important because:
- Missing a defaulter (false negative) leads to financial loss  
- Incorrectly rejecting a good customer (false positive) reduces business opportunity  

The project focuses on balancing these trade-offs using a cost-sensitive approach.

---

## Methodology

### Data Preprocessing
- Numerical features scaled using `StandardScaler`  
- Categorical features encoded using `OneHotEncoder`  
- Combined using `ColumnTransformer`  
- End-to-end pipeline created with preprocessing + model  

---

### Model
- Algorithm: Logistic Regression  
- Solver configured with increased iterations (`max_iter=1000`)  
- Pipeline ensures reproducibility and prevents data leakage  

---

### Validation Strategy
- Stratified 5-Fold Cross-Validation  
- Maintains class distribution across folds  

---

### Evaluation Metrics
The model is evaluated using multiple components:

- ROC-AUC score (cross-validation)  
- Confusion matrix  
- Precision  
- Recall  

Predicted probabilities are generated using:
- `predict_proba()`  

---

### Cost-Sensitive Evaluation
A custom cost function is defined:

- False Negative (FN) = 5  
- False Positive (FP) = 1  

Total cost is computed using confusion matrix outputs.

---

### Threshold Optimization
- Thresholds evaluated from 0 to 1  
- For each threshold:
  - Convert probabilities to class predictions  
  - Compute confusion matrix  
  - Calculate total cost  

- Identify threshold that minimizes total cost  

---

### Comparison of Thresholds
Two scenarios are compared:

1. Default threshold (0.5)  
2. Optimized threshold  

Metrics compared:
- Total cost  
- Precision  
- Recall  

---

### Visualization
- Cost vs Threshold plot used to identify optimal decision boundary  

---

## Technical Soundness
- Appropriate use of Logistic Regression for binary classification  
- Proper preprocessing pipeline avoids data leakage  
- Cross-validation ensures robust evaluation  
- Use of ROC-AUC and probability predictions strengthens analysis  
- Threshold optimization aligns model with real-world decision-making  

---

## Code Quality and Documentation
- Modular pipeline structure improves readability and reproducibility  
- Clear separation of preprocessing, modeling, and evaluation  
- Use of sklearn utilities ensures consistency  

---

## Key Technical Choice
A key design decision is optimizing the classification threshold instead of using the default value (0.5).  

This is appropriate because:
- Default thresholds assume equal misclassification costs  
- Real-world problems often have asymmetric costs  
- The chosen approach directly minimizes business loss  

---

## Results and Insights
- Default threshold is not optimal for minimizing financial cost  
- Optimized threshold improves detection of defaulters  
- Trade-off observed between precision and recall  
- Model performance is better aligned with business impact  

---

## Originality and Insight
- Incorporates cost-sensitive learning instead of relying only on standard metrics  
- Demonstrates practical understanding of decision thresholds  
- Connects machine learning outputs with business implications  

---


## Tech Stack
- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

---
