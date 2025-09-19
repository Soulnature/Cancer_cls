# Cancer_cls

Python utilities for evaluating machine-learning classifiers on cytotoxicity data from the study
"Dual engagement of T cells to tumor and myeloid cells augments anti-tumor immunity". The
`model_class.py` script bundles feature selection pipelines (Boruta, SHAP, mutual information) with
model comparison across several classical classifiers and automated report export.

## Features
- End-to-end stratified k-fold evaluation for Naive Bayes, Logistic Regression (L1/L2/Elastic Net),
  Decision Tree, SVM, and Random Forest.
- Automatic handling of class imbalance via SMOTE inside each CV fold (using `imblearn.Pipeline`).
- Mean ROC curve plotting with per-fold AUC tracking and PDF/CSV output.
- Boxplot visualization of F1/Precision/Recall/Accuracy distributions across folds.
- Feature selection helpers powered by Boruta, SHAP (TreeExplainer), and mutual information scores.
- Optional evaluation of curated gene panels for external validation sets.

## Requirements
- Python 3.9+
- Packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `boruta`,
  `shap`, `joblib`

Create an isolated environment (conda or venv) and install dependencies, for example:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn boruta shap joblib
```

## Data Preparation
- Place the main dataset at `./ecotyper_cellstate.csv` with columns `_ID`, `cytotoxicity(0-1)`, and
  the feature columns (genes or other measurements).
- Optionally add a curated gene set at
  `./validation_set/cytotoxicity_select_genes/Cytotoxicity_select genes.csv` for the external
  validation step. Columns should overlap with the main feature matrix; `sample_id` will be dropped
  automatically if present.

## Usage
Run the example pipeline (feature selection + model evaluation) directly:

```bash
python model_class.py
```

This will:
1. Load `ecotyper_cellstate.csv` and binarize the `cytotoxicity(0-1)` label using a 0.5 threshold.
2. Perform Boruta feature selection, compute SHAP importances, and keep the top `fea_num` features
   (default 10).
3. Evaluate all supported models on the selected feature subset, exporting plots and metrics.
4. Optionally repeat evaluation on a curated gene panel if the validation CSV is available.

Key outputs are saved in the project root unless you edit the paths:
- `*_roc.pdf` / `*_box.pdf`: ROC curves and metric boxplots.
- `*_roc.csv` / `*_auc.csv` / `*_metrics.csv`: Raw data behind the figures.
- `*_best_model.joblib`: Serialized best-performing pipeline (chosen by mean precision).
- `*_summary.csv`: Lightweight summary containing the best model name per evaluation run.

## Customization
- Adjust `threshold`, `n_splits`, or output paths by passing arguments to `evaluate_models`.
- Change `fea_num` or swap to mutual information selection by using the respective helper methods in
  `FeatureSelector`.
- Modify the classifier dictionary inside `evaluate_models` to experiment with additional models or
  hyperparameters.

## Reproducibility Notes
- `random_state=42` is used consistently for deterministic CV splits, SMOTE sampling, and model
  training.
- The script uses multiprocessing where available (e.g., RandomForest, cross-validation) but is
  configured to run on typical workstation hardware.
