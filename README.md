# DL Final Project: Breast Cancer Classification

End-to-end machine learning and deep learning workflow on the **Wisconsin Breast Cancer** dataset, implemented in Jupyter notebooks.

## Project Goals
- Build strong baseline classifiers for benign vs malignant diagnosis.
- Compare **full feature space (30 features)** vs **selected feature subsets (10 features)**.
- Evaluate impact of feature engineering, feature selection, and hyperparameter tuning.
- Add model explainability (feature importance + SHAP) and an ANN benchmark.

## Dataset
- Source: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numeric features
- Target: binary classification (`0=malignant`, `1=benign` in sklearn encoding)
- Split strategy: stratified train/test split with stratified 5-fold CV in experiments

## Tech Stack
- Python 3.12
- NumPy, pandas, scikit-learn
- PyTorch (ANN experiments)
- SHAP + matplotlib (interpretability visuals)
- Jupyter notebooks
- Dependency management: `uv`

## Repository Structure
- `notebooks/01_eda.ipynb` — data exploration and inspection
- `notebooks/02_required_ml_models.ipynb` — baseline ML model comparison
- `notebooks/03_feature_engineering_selection.ipynb` — feature engineering + L1/RFE selection
- `notebooks/04_tuning_experiments.ipynb` — hyperparameter tuning (LogReg, SVM, RF)
- `notebooks/05_model_explainability.ipynb` — RF importances + SHAP analysis
- `notebooks/06_deep_learning_ann.ipynb` — ANN on full vs selected features
- `outputs/` — CSV/JSON experiment results and selected features
- `figures/` — confusion matrix, SHAP plots, ANN learning curves
- `models/` — saved ANN checkpoints and scalers

## Reproducibility
Most notebooks use a fixed random seed (`SEED = 42`) and stratified cross-validation.

## Main Results (from `outputs/`)

### Baseline ML (CV, full features)
- Logistic Regression: F1 = **0.9825**, AUC = **0.9959**
- SVM (RBF): F1 = **0.9756**, AUC = **0.9956**
- Random Forest: F1 = **0.9699**, AUC = **0.9896**

### Baseline ML (test, full features)
- Logistic Regression: Accuracy = **0.9825**, F1 = **0.9861**, AUC = **0.9954**
- SVM (RBF): Accuracy = **0.9825**, F1 = **0.9861**, AUC = **0.9950**

### Feature Selection Impact
Using selected features improved several models in CV (notably Decision Tree and Random Forest), while Logistic Regression stayed very close to full-feature performance.

L1-selected feature list is saved in:
- `outputs/03_selected_features_l1.json`

RFE-selected feature list is saved in:
- `outputs/03_selected_features_rfe.json`

### Hyperparameter Tuning
Best parameters are stored in:
- `outputs/04_best_params.json`

Tuning generally improved CV F1 for LogReg/SVM/RF, with strong test metrics maintained for tuned SVM and tuned LogReg.

### ANN (PyTorch)
From `outputs/06_ann_full_vs_selected_results.csv`:
- Best configuration in this run: **selected features (10), baseline ANN**
  - Accuracy = **0.9649**, F1 (malignant) = **0.9524**, AUC = **0.9944**

## Setup
### 1) Create environment and install dependencies
```bash
uv sync
```

### 2) Start Jupyter
```bash
uv run jupyter notebook
```

Then run notebooks in this order:
1. `01_eda.ipynb`
2. `02_required_ml_models.ipynb`
3. `03_feature_engineering_selection.ipynb`
4. `04_tuning_experiments.ipynb`
5. `05_model_explainability.ipynb`
6. `06_deep_learning_ann.ipynb`

## Notes
- Existing result files in `outputs/` and `figures/` let you inspect outcomes without rerunning everything.
- If you change random seeds or library versions, exact metrics may vary slightly.
