from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.inspection import permutation_importance
import pandas as pd

from data_processing import load_data
from model import get_LogisticRegression, get_RandomForest, get_GradientBoosting, get_StackingEnsemble
from eval import eval_model, find_optimal_threshold
from visualizations import feature_importance, create_confusion_matrix, create_top_feature_plot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_features(X_train, X_test, y_train, high_card, low_card):
    """
    Apply target encoding to high-cardinality columns and ordinal encoding to
    low-cardinality columns.  Both encoders are fitted ONLY on the training
    fold to prevent data leakage.
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    te = TargetEncoder(target_type='binary', smooth='auto')
    X_train_enc[high_card] = te.fit_transform(X_train[high_card].astype(str), y_train)
    X_test_enc[high_card] = te.transform(X_test[high_card].astype(str))

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_enc[low_card] = oe.fit_transform(X_train[low_card].astype(str))
    X_test_enc[low_card] = oe.transform(X_test[low_card].astype(str))

    return X_train_enc, X_test_enc


def print_results(name, acc, roc_auc, pr_auc, f1, precision, recall, threshold):
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Threshold  : {threshold:.2f}  (F1-optimal on training set)")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  ROC AUC    : {roc_auc:.4f}")
    print(f"  PR AUC     : {pr_auc:.4f}  ← key metric for imbalanced data")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Starting data ingestion...")
    analysis_df, selected_features, high_card, low_card = load_data()
    print("Data loaded. Shape:", analysis_df.shape)

    X = analysis_df[selected_features]
    y = analysis_df['target']

    print(f"\nClass balance  →  0 (no readmit): {(y==0).sum()}  |  1 (<30 days): {(y==1).sum()}")
    print(f"Positive rate  →  {y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_enc, X_test_enc = encode_features(X_train, X_test, y_train, high_card, low_card)

    # Integer indices of low-cardinality columns (required by HistGBM's native cat support)
    cat_indices = [list(X_train_enc.columns).index(c) for c in low_card]

    # ------------------------------------------------------------------
    # 1. Logistic Regression — interpretable baseline
    # ------------------------------------------------------------------
    print("Training Logistic Regression (baseline)...")
    lr_model = get_LogisticRegression()
    lr_model.fit(X_train_enc, y_train)
    lr_thresh = find_optimal_threshold(lr_model, X_train_enc, y_train)
    lr_results = eval_model(lr_model, X_test_enc, y_test, threshold=lr_thresh)
    acc_lr, _, cm_lr, roc_lr, pr_lr, f1_lr, prec_lr, rec_lr = lr_results
    print_results("Logistic Regression (Baseline)", acc_lr, roc_lr, pr_lr, f1_lr, prec_lr, rec_lr, lr_thresh)

    # ------------------------------------------------------------------
    # 2. Random Forest
    # ------------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf_model = get_RandomForest()
    rf_model.fit(X_train_enc, y_train)
    rf_thresh = find_optimal_threshold(rf_model, X_train_enc, y_train)
    rf_results = eval_model(rf_model, X_test_enc, y_test, threshold=rf_thresh)
    acc_rf, _, cm_rf, roc_rf, pr_rf, f1_rf, prec_rf, rec_rf = rf_results
    print_results("Random Forest", acc_rf, roc_rf, pr_rf, f1_rf, prec_rf, rec_rf, rf_thresh)

    # ------------------------------------------------------------------
    # 3. Gradient Boosting (primary model)
    # ------------------------------------------------------------------
    print("\nTraining Gradient Boosting...")
    gb_model = get_GradientBoosting(cat_indices=cat_indices)
    gb_model.fit(X_train_enc, y_train)
    gb_thresh = find_optimal_threshold(gb_model, X_train_enc, y_train)
    gb_results = eval_model(gb_model, X_test_enc, y_test, threshold=gb_thresh)
    acc_gb, _, cm_gb, roc_gb, pr_gb, f1_gb, prec_gb, rec_gb = gb_results
    print_results("Gradient Boosting (HGBC)", acc_gb, roc_gb, pr_gb, f1_gb, prec_gb, rec_gb, gb_thresh)

    # ------------------------------------------------------------------
    # 4. Stacking Ensemble (RF + GB → LR meta-learner)
    # ------------------------------------------------------------------
    print("\nTraining Stacking Ensemble (RF + GB → LR)  — this takes a few minutes...")
    stack_model = get_StackingEnsemble(cat_indices=cat_indices)
    stack_model.fit(X_train_enc, y_train)
    st_thresh = find_optimal_threshold(stack_model, X_train_enc, y_train)
    st_results = eval_model(stack_model, X_test_enc, y_test, threshold=st_thresh)
    acc_st, _, cm_st, roc_st, pr_st, f1_st, prec_st, rec_st = st_results
    print_results("Stacking Ensemble (RF + GB → LR)", acc_st, roc_st, pr_st, f1_st, prec_st, rec_st, st_thresh)

    # ------------------------------------------------------------------
    # 5-Fold Stratified Cross-Validation on Gradient Boosting
    # A single 80/20 split can be lucky or unlucky; CV gives a reliable estimate.
    # ------------------------------------------------------------------
    print("\n--- 5-Fold Stratified CV  (Gradient Boosting, ROC AUC) ---")
    gb_cv = get_GradientBoosting(cat_indices=cat_indices)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_cv, X_train_enc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  CV ROC AUC: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")
    print(f"  Per-fold:   {[f'{s:.4f}' for s in cv_scores]}")

    # ------------------------------------------------------------------
    # Feature importances (permutation — model-agnostic, on test set)
    # ------------------------------------------------------------------
    print("\nExtracting permutation feature importances from Gradient Boosting...")
    perm_imp = permutation_importance(
        gb_model, X_test_enc, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    sorted_imp = feature_importance(perm_imp.importances_mean, selected_features)
    create_confusion_matrix(cm_gb)

    top_feature = sorted_imp[0][0]
    create_top_feature_plot(analysis_df, top_feature)

    print("\nTop 15 Features (permutation importance on Gradient Boosting):")
    for feat, imp in sorted_imp[:15]:
        print(f"  {feat:<40} {imp:.4f}")


if __name__ == "__main__":
    main()