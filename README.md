# DA5401 – Assignment 06  
### Imputation via Regression

## Overview
This repository contains the implementation for **DA5401: Data Analytics Laboratory – Assignment 6**, focusing on the impact of **missing value imputation techniques** on downstream model performance.

The experiment investigates how different imputation strategies influence a classifier’s accuracy and reliability when data is **Missing At Random (MAR)**.  
The analysis is based on the **UCI Credit Card Default Dataset**, where the feature **`LIMIT_BAL`** (credit limit) was deliberately perturbed with controlled MAR missingness.

---

## Objective
To compare multiple imputation strategies—simple, linear, and non-linear—against listwise deletion, and evaluate their effect on predictive model performance.

Specifically, the assignment addresses:
1. Injecting MAR missingness into a chosen numeric column (`LIMIT_BAL`).
2. Performing imputation via:
   - Median replacement  
   - Linear regression model  
   - Non-linear regression model (KNN)
3. Comparing against listwise deletion.
4. Measuring the impact on a downstream **Logistic Regression** classifier using standard metrics.

---

## Dataset
**Source:** UCI Machine Learning Repository – Default of Credit Card Clients Dataset  
**File:** `UCI_Credit_Card.csv`

- **Observations:** 30,000  
- **Target variable:** `default.payment.next.month`  
- **Injected MAR feature:** `LIMIT_BAL`  
- **MAR condition:** Probability of missingness increases with `BILL_AMT1` percentile (5–10% missing rate).

---

## Methodology

### 1. MAR Injection
- Introduced **Missing At Random** missingness only in `LIMIT_BAL`.  
- Missing probability increases proportionally with the percentile of `BILL_AMT1`.  
- Visualized using scatter plots and NaN count distributions.

### 2. Datasets and Imputation Pipelines

| Dataset | Description | Imputation Method | Key Detail |
|----------|--------------|-------------------|-------------|
| **A** | Median Imputation | SimpleImputer (Median) | Baseline, robust to outliers |
| **B** | Linear Regression Imputation | LinearRegression | Predicts missing `LIMIT_BAL` from other features |
| **C** | Non-Linear Imputation | KNeighborsRegressor | Captures local non-linear relationships |
| **D** | Listwise Deletion | dropna() on training rows only | Removes rows with missing `LIMIT_BAL` |

All pipelines share the same **train/test split** and **identical test cohort** (rows where `LIMIT_BAL` is observed), ensuring fair comparison.

### 3. Model and Evaluation
- **Downstream Model:** Logistic Regression (`scikit-learn`)
- **Preprocessing:** StandardScaler (fit on training data only)
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

### 4. Visual Analysis
- Comparison bar plots (F1 and ROC-AUC)
- ROC and Precision–Recall curves
- Confusion matrices for each pipeline
- Distribution plots (density and log-histograms) of imputed `LIMIT_BAL`
- Scatter visualization of imputed vs. observed relationships

---

## Results Summary

| Model | Method | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:------|:--------|:---------:|:----------:|:-------:|:----------:|:---------:|
| **A** | Median Imputation | 0.8097 | 0.6779 | 0.2664 | 0.3825 | 0.7187 |
| **B** | Linear Regression Imputation | 0.8092 | 0.6754 | 0.2646 | 0.3803 | 0.7189 |
| **C** | Non-Linear Regression Imputation | 0.8091 | 0.6743 | 0.2646 | 0.3801 | 0.7185 |
| **D** | Listwise Deletion (train-only) | 0.8164 | 0.7349 | 0.2685 | 0.3933 | 0.7270 |

**Key Observations:**
- All models perform similarly because of the small MAR rate (≈5–10%) and the weak correlation of `LIMIT_BAL` with the target.
- Listwise Deletion appears slightly higher numerically but benefits from removing difficult samples, not from improved modeling.
- Model-based imputations (B and C) retain full data, maintaining generalization and statistical consistency.
- ROC-AUC values (~0.72) show comparable ranking ability across all pipelines.

---

## Discussion

### Trade-off: Listwise Deletion vs Imputation
Listwise Deletion simplifies preprocessing but causes **data loss** and potential **bias** when missingness is MAR.  
Imputation retains all records, preserving population structure. The minor numeric edge of deletion stems from cleaner (easier) subsets, not genuine predictive improvement.

### Linear vs Non-Linear Regression Imputation
Linear and non-linear methods yield almost identical results, implying that `LIMIT_BAL` has a mostly linear relationship with other variables and limited predictive contribution.  
In more complex scenarios, non-linear imputers like KNN would typically outperform linear models.

### Recommended Strategy
- **Preferred:** Model-based imputation (Linear or Non-Linear) for MAR data.  
- **Acceptable Baseline:** Median imputation for simplicity and speed.  
- **Avoid:** Listwise deletion, unless missingness is truly MCAR and minimal.

---

## Repository Structure

