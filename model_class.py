# -*- coding: utf-8 -*-
"""
End-to-end classification evaluation with:
- Model comparison (NB, Logistic Regression variants, Decision Tree, SVM, RF)
- Stratified K-fold mean ROC curves
- Per-fold metrics for boxplots (F1/Precision/Recall/Accuracy)
- Save plots (PDF) and raw data used to plot (CSV)
- Feature selection via Boruta and SHAP (TreeExplainer), then evaluate with selected features

Notes:
- We use imblearn.Pipeline so SMOTE is applied only on training folds.
- 'Best model' is chosen by mean precision (to match your original logic).
"""

import os
import joblib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # use imblearn Pipeline to include SMOTE

from boruta import BorutaPy
import shap
from sklearn.feature_selection import mutual_info_classif


# ---------------------------
# Model evaluation function
# ---------------------------
def evaluate_models(
    X, Y,
    threshold=0.5,
    save_path='best_model.joblib',
    roc_pdf_path='./roc_mean.pdf',
    box_pdf_path='./metrics_boxplot.pdf',
    roc_csv_path='./roc_mean.csv',
    auc_csv_path='./auc_folds.csv',
    metrics_csv_path='./metrics_folds.csv',
    n_splits=5,
    random_state=42
):
    """
    Evaluate multiple models using stratified K-fold CV, plot & save mean ROC curves and
    boxplots for F1/Precision/Recall/Accuracy, and save the best model (by mean precision).

    Besides figures, also save:
      - roc_mean.csv: per-model mean ROC points on a shared FPR grid
      - auc_folds.csv: per-model per-fold AUC values
      - metrics_folds.csv: per-model per-fold metrics used for the boxplot

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    Y : pd.Series or 1D array
        Target values (will be binarized by threshold).
    threshold : float
        Threshold to binarize target into {0,1}.
    save_path : str
        File path to save the best model via joblib.
    roc_pdf_path : str
        Output PDF path for mean ROC curves.
    box_pdf_path : str
        Output PDF path for the metrics boxplot.
    roc_csv_path : str
        CSV path to save per-model mean ROC points.
    auc_csv_path : str
        CSV path to save per-model per-fold AUC.
    metrics_csv_path : str
        CSV path to save per-model per-fold metrics.
    n_splits : int
        Number of folds for StratifiedKFold.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        {
          'results': dict(model -> {metric -> np.ndarray of per-fold scores}),
          'roc_mean': pd.DataFrame(columns=['model','fpr','tpr']),
          'auc_folds': pd.DataFrame(columns=['model','fold','auc']),
          'metrics_folds': pd.DataFrame(columns=['model','metric','fold','value']),
          'best_model_name': str or None
        }
    """
    # --- 1) Binarize labels
    binarizer = Binarizer(threshold=threshold)
    y = binarizer.fit_transform(np.asarray(Y).reshape(-1, 1)).ravel().astype(int)
    print("Class counts (before SMOTE; SMOTE is used within pipelines only):", Counter(y))

    # --- 2) Define models (SMOTE only within training folds)
    models = {
        'Naive Bayes': GaussianNB(),

        'Logistic Regression (L1)': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('lr', LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state))
        ]),

        'Logistic Regression (L2)': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('lr', LogisticRegression(penalty='l2', random_state=random_state))
        ]),

        'Logistic Regression (Elastic Net)': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('lr', LogisticRegression(penalty='elasticnet', solver='saga',
                                     l1_ratio=0.5, random_state=random_state, max_iter=10000))
        ]),

        'Decision Tree': Pipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('decision_tree', DecisionTreeClassifier(random_state=random_state))
        ]),

        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('svm', SVC(kernel='rbf', class_weight='balanced',
                        random_state=random_state, probability=True))
        ]),

        'Random Forest': Pipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('RF', RandomForestClassifier(n_estimators=200, random_state=random_state))
        ]),
    }

    # --- 3) CV config and containers
    scoring = ['f1', 'precision', 'recall', 'accuracy']
    results = {}
    best_model = None
    best_model_name = None
    best_mean_precision = -np.inf

    mean_fpr = np.linspace(0, 1, 100)
    roc_rows = []   # long-form rows: {model, fpr, tpr}
    auc_rows = []   # per fold rows: {model, fold, auc}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    plt.figure(figsize=(10, 8))

    # --- 4) For each model: compute mean ROC and cross-validated metrics
    for name, model in models.items():
        tprs = []
        fold_idx = 0
        for train_idx, test_idx in cv.split(X, y):
            # Handle DataFrame vs ndarray indexing
            X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_test  = X.iloc[test_idx]  if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train = y[train_idx]
            y_test  = y[test_idx]

            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probas_)

            # Interpolate TPR onto a shared FPR grid for averaging
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

            fold_auc = auc(fpr, tpr)
            auc_rows.append({'model': name, 'fold': fold_idx, 'auc': fold_auc})
            fold_idx += 1

        # Mean ROC across folds
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        # Store long-form ROC rows (100 points per model)
        for fpr_v, tpr_v in zip(mean_fpr, mean_tpr):
            roc_rows.append({'model': name, 'fpr': fpr_v, 'tpr': tpr_v})

        plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f})')

        # Cross-validated metrics (per fold) to drive boxplots
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False)
        results[name] = {metric: scores[f'test_{metric}'] for metric in scoring}

        # Choose best model by mean precision (kept for compatibility with your original code)
        mean_precision = np.mean(results[name]['precision'])
        if mean_precision >= best_mean_precision:
            best_mean_precision = mean_precision
            best_model = model
            best_model_name = name

    # --- 5) Save mean ROC figure
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves (Stratified {n_splits}-Fold)')
    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    os.makedirs(os.path.dirname(roc_pdf_path) or '.', exist_ok=True)
    plt.savefig(roc_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    # --- 6) Build long-form metrics DataFrame for boxplot and save figure
    metrics_long = []
    for model_name, metric_dict in results.items():
        for metric, values in metric_dict.items():
            for fold_i, v in enumerate(values):
                metrics_long.append({'model': model_name, 'metric': metric, 'fold': fold_i, 'value': v})
    metrics_df = pd.DataFrame(metrics_long)

    # Boxplot across models for each metric (grouped horizontally)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, metric in enumerate(['f1', 'precision', 'recall', 'accuracy']):
        subset = metrics_df[metrics_df['metric'] == metric]
        order = list(models.keys())
        data_to_plot = [subset[subset['model'] == m]['value'].values for m in order]

        pos = np.arange(len(order)) + i * 0.22
        bp = ax.boxplot(data_to_plot, positions=pos, widths=0.18, patch_artist=True, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_alpha(0.5)

    ax.set_xticks(np.arange(len(models)) + 0.33)
    ax.set_xticklabels(list(models.keys()), rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance (Stratified {n_splits}-Fold)')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(box_pdf_path) or '.', exist_ok=True)
    plt.savefig(box_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    # --- 7) Save raw data used for plotting
    roc_df = pd.DataFrame(roc_rows)   # columns: model, fpr, tpr
    auc_df = pd.DataFrame(auc_rows)   # columns: model, fold, auc

    os.makedirs(os.path.dirname(roc_csv_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(auc_csv_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(metrics_csv_path) or '.', exist_ok=True)

    roc_df.to_csv(roc_csv_path, index=False)
    auc_df.to_csv(auc_csv_path, index=False)
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(f"Saved mean ROC to: {roc_csv_path}")
    print(f"Saved per-fold AUC to: {auc_csv_path}")
    print(f"Saved per-fold metrics (for boxplot) to: {metrics_csv_path}")

    # --- 8) Fit and save the best model on full data
    if best_model is not None:
        best_model.fit(X, y)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"Best model ({best_model_name}, mean precision={best_mean_precision:.4f}) saved to: {save_path}")
    else:
        print("No model saved.")

    return {
        'results': results,
        'roc_mean': roc_df,
        'auc_folds': auc_df,
        'metrics_folds': metrics_df,
        'best_model_name': best_model_name
    }


# ---------------------------
# Feature selection utilities
# ---------------------------
class FeatureSelector:
    """
    Provide Boruta, Mutual Information, and SHAP-based feature importance utilities.
    """
    def __init__(self, random_state=42, fea_num=20):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.fea_num = fea_num

    def boruta_selection(self, X, y, n_estimators=100, max_iter=200):
        """
        Run Boruta on scaled features with a RandomForest base estimator.
        Returns selected feature names, X_selected (original scale), fitted RF, and X_scaled.
        """
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=4,
            class_weight='balanced',
            random_state=self.random_state
        )
        selector = BorutaPy(rf, n_estimators='auto', verbose=2,
                            random_state=self.random_state, max_iter=max_iter)
        selector.fit(X_scaled.values, np.asarray(y))

        print('\nNumber of selected features:', selector.n_features_)
        selected = X.columns[selector.support_].tolist()

        # Inspect ranked list (1 = top rank)
        feature_df = pd.DataFrame({'features': X_scaled.columns, 'rank': selector.ranking_})
        feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
        print(f'\nTop {selector.n_features_} features by Boruta ranking:')
        print(feature_df.head(selector.n_features_))

        print(f"Boruta selected {len(selected)} features: {selected}")
        X_sel = X[selected]
        return selected, X_sel, rf, X_scaled

    def mutual_info_selection(self, X, y):
        """
        Mutual Information scores on scaled features. Returns top-k names, X_selected, and full score series.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if isinstance(y, pd.Series) and y.ndim > 1:
            y = y.squeeze()

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        mi_scores = mutual_info_classif(X_scaled, y, random_state=self.random_state)
        mi_scores = pd.Series(mi_scores, index=X_scaled.columns).sort_values(ascending=False)
        print(mi_scores)

        selected = mi_scores.head(self.fea_num).index.tolist()
        X_sel = X[selected]
        print(f"Mutual Information selected {len(selected)} features: {selected}")
        return selected, X_sel, mi_scores

    def shap_importance(self, fitted_tree_model, X_scaled, pdf_path='boruta_importance.pdf', csv_path='shap_value.csv'):
        """
        Compute SHAP values for a fitted tree model on X_scaled, save a horizontal bar plot and CSV.
        Returns a DataFrame with non-zero importance: columns ['feature','importance'] sorted desc.
        """
        explainer = shap.TreeExplainer(fitted_tree_model)
        shap_values = explainer.shap_values(X_scaled)

        # Unify to (n_samples, n_features)
        if isinstance(shap_values, list):  # multi-class or binary returns list
            sv = np.mean([np.abs(sv_i) for sv_i in shap_values], axis=0)  # average over classes
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)  # mean |SHAP| over samples
        feature_importance = pd.DataFrame({'feature': X_scaled.columns, 'importance': mean_abs})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Save CSV of all features
        feature_importance.to_csv(csv_path, index=False)

        # Plot top-N or all (here: all, but you can slice head(k))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance based on SHAP values')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.yticks(fontsize=6)
        plt.xticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP importance to: {csv_path} and {pdf_path}")

        # Return non-zero important features (useful to pick top-k later)
        return feature_importance[feature_importance['importance'] > 0].reset_index(drop=True)


# ---------------------------
# Example: feature selection + evaluation
# ---------------------------
if __name__ == "__main__":
    # --- Load data
    file_path = './ecotyper_cellstate.csv'  # adjust to your local path
    data = pd.read_csv(file_path)

    # --- Preprocess labels
    X_full = data.drop(columns=['_ID', 'cytotoxicity(0-1)'])
    y_cont = data['cytotoxicity(0-1)'].values
    y_bin = (y_cont > 0.5).astype(int)
    y_ser = pd.Series(y_bin, name='target')

    # --- Feature selection via Boruta and SHAP
    selector = FeatureSelector(random_state=42, fea_num=10)
    boruta_features, X_boruta, rf_model, X_scaled = selector.boruta_selection(X_full, y_ser)

    # Compute SHAP importances using the Boruta-trained RandomForest on scaled X
    shap_df = selector.shap_importance(
        fitted_tree_model=rf_model,
        X_scaled=X_scaled,
        pdf_path='boruta_shap_importance.pdf',
        csv_path='shap_value.csv'
    )

    # Pick top-k features by SHAP (fallback to min(len(shap_df), fea_num))
    k = min(selector.fea_num, shap_df.shape[0])
    select_fea = shap_df['feature'].head(k).tolist()
    print(f"[SHAP] Top-{k} features:", select_fea)

    # --- Evaluate models with SHAP-selected features
    # Build DataFrame subset to keep column names with DataFrame behavior
    X_shap = X_full[select_fea]

    # Define a helper to run evaluation and save artifacts for a given feature set
    def test_feature_result(fea_df, label, threshold, tag):
        """
        fea_df : pd.DataFrame
            Feature subset (as DataFrame).
        label : array-like
            Target (continuous or already binary; will be binarized by threshold inside evaluate_models).
        threshold : float
            Binarization threshold for label.
        tag : str
            Prefix for output files.
        """
        model_path = f'./{tag}_best_model.joblib'
        roc_pdf = f'./{tag}_roc.pdf'
        box_pdf = f'./{tag}_box.pdf'
        roc_csv = f'./{tag}_roc.csv'
        auc_csv = f'./{tag}_auc.csv'
        metrics_csv = f'./{tag}_metrics.csv'

        results = evaluate_models(
            fea_df, label, threshold,
            save_path=model_path,
            roc_pdf_path=roc_pdf,
            box_pdf_path=box_pdf,
            roc_csv_path=roc_csv,
            auc_csv_path=auc_csv,
            metrics_csv_path=metrics_csv,
            n_splits=5,
            random_state=42
        )

        # Save a compact JSON-like CSV with summary (optional)
        # Here we just save best model name and ignore the rest (you can extend if needed)
        pd.DataFrame({'best_model': [results['best_model_name']]}).to_csv(f'./{tag}_summary.csv', index=False)
        print(f"[{tag}] evaluation completed.")

    # Run evaluation on SHAP-selected features
    test_feature_result(X_shap, y_cont, 0.5, 'shap')

    # --- (Optional) Evaluate a curated gene set if available
    # Path contains a space; ensure exact file name is used
    gene_path = './validation_set/cytotoxicity_select_genes/Cytotoxicity_select genes.csv'
    if os.path.exists(gene_path):
        gene_df = pd.read_csv(gene_path).drop(columns=['sample_id'], errors='ignore')
        # Align columns to X_full if necessary
        gene_cols = [c for c in gene_df.columns if c in X_full.columns]
        if len(gene_cols) > 0:
            X_gene = X_full[gene_cols]
            test_feature_result(X_gene, y_cont, 0.5, 'gene')
        else:
            print("[gene] No overlapping gene columns found in the main matrix; skipped.")
    else:
        print(f"[gene] File not found: {gene_path} (skipped)")

    # --- (Optional) Evaluate without feature selection (baseline slice)
    # baseline_X = X_full.iloc[:, 20:30]
    # test_feature_result(baseline_X, y_cont, 0.5, 'no_feature_selection')