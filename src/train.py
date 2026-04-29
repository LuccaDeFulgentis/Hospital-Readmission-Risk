from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from data_processing import load_data
from model import get_LogisticRegression, get_RandomForest, get_GradientBoosting
from eval import eval_model
from visualizations import *
import pandas as pd

def main():

    print("Starting data ingestion...")

    analysis_df, selected_features, high_card, low_card = load_data()

    print("Data loaded. Shape:", analysis_df.shape)
    print(f"\nTraining Models on all {len(selected_features)} features to achieve absolute SOTA...")

    X = analysis_df[selected_features]
    y = analysis_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Encode High Cardinality Features using Target Encoding (ONLY fit on train to prevent data leakage)
    te = TargetEncoder(target_type='binary', smooth='auto')
    X_train_encoded[high_card] = te.fit_transform(X_train[high_card].astype(str), y_train)
    X_test_encoded[high_card] = te.transform(X_test[high_card].astype(str))

    # Encode Low Cardinality Features using Ordinal Encoding
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded[low_card] = oe.fit_transform(X_train[low_card].astype(str))
    X_test_encoded[low_card] = oe.transform(X_test[low_card].astype(str))

    # Get the integer indices of the low_card categorical columns to pass to Gradient Boosting
    cat_indices = [list(X_train_encoded.columns).index(c) for c in low_card]

    rf_model = get_RandomForest()
    gb_model = get_GradientBoosting(cat_indices=cat_indices)

    rf_model.fit(X_train_encoded, y_train)
    acc_rf, y_pred_rf, cm_rf, roc_rf = eval_model(rf_model, X_test_encoded, y_test)

    gb_model.fit(X_train_encoded, y_train)
    acc_gb, y_pred_gb, cm_gb, roc_gb = eval_model(gb_model, X_test_encoded, y_test)

    from sklearn.inspection import permutation_importance
    print("Extracting SOTA feature importances (this may take a moment)...")
    perm_importance = permutation_importance(gb_model, X_test_encoded, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    # Create the sorted_imp structure expected by visualizations.py
    imp_dict = dict(zip(selected_features, perm_importance.importances_mean))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Use Gradient Boosting true feature importances for visualization
    sorted_imp = feature_importance(perm_importance.importances_mean, selected_features)
    create_confusion_matrix(cm_gb)
    
    top_feature = sorted_imp[0][0]
    create_top_feature_plot(analysis_df, top_feature)

    #Display Statistics 
    print("")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"Random Forest ROC: {roc_rf:.4f}")
    print(f"Absolute SOTA Gradient Boosting Accuracy: {acc_gb:.4f}")
    print(f"Absolute SOTA Gradient Boosting ROC: {roc_gb:.4f}")

    print("\nTop 15 Features:")
    for feature, importance in sorted_imp[:15]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()