# secom-quality-failure-prediction
High-dimensional manufacturing quality analytics on SECOM (1567×590): dimensionality reduction, clustering interpretation, and SMOTE-based failure prediction.

# SECOM Failure Detection (Clustering + Autoencoder + Imbalanced Learning)

End-to-end machine learning pipeline on the **SECOM semiconductor manufacturing dataset** to (1) discover structure via **unsupervised clustering** and (2) predict rare **product failures** under severe class imbalance.

## Highlights
- Cleaned a high-dimensional sensor dataset (**1567 samples, ~590 sensors**) with substantial missingness.
- Built preprocessing + modeling pipelines with reproducible splits and scaling.
- Compared representations:
  - cleaned features
  - RFECV-selected features
  - PCA components
  - deep autoencoder latent vectors
- Evaluated failure prediction using **Precision–Recall / AUPRC** (more informative than accuracy for rare failures).

## Dataset
- Source: SECOM dataset (UCI / Kaggle mirror)
- Label: Pass/Fail (rare Fail class)

> Note: The dataset is not stored in this repo. See **Quickstart** to download.

## Methods
### 1) Preprocessing
- Drop features with high missingness
- Row filtering for near-complete columns
- Median/mean imputation (distribution-aware)
- Standardization (z-score)

### 2) Dimensionality Reduction / Feature Engineering
- Low-variance feature removal
- Correlation pruning (Pearson/Spearman)
- RFECV (class-weighted logistic regression)
- PCA for compact linear representation
- Deep Autoencoder for non-linear latent representation

### 3) Unsupervised Learning (Structure Discovery)
- KMeans baseline
- DBSCAN for density-based clusters + noise handling
- t-SNE for 2D/3D visualization of cluster geometry

### 4) Supervised Learning (Failure Prediction under Imbalance)
- Logistic Regression baseline
- SMOTE oversampling on training split
- Hybrid samplers: SMOTE-ENN, SMOTE-Tomek
- Metrics: F1 (by class) + AUPRC, confusion matrices, PR curves
