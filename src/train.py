from sklearn.model_selection import train_test_split
from data_processing import load_data
from model import get_LogisticRegression, get_RandomForest
from eval import eval_model
from visualizations import *

def main():

    print("Starting data ingestion...")

    analysis_df, selected_features = load_data()

    print("Data loaded. Shape:", analysis_df.shape)
    print("\nTraining Models...")

    X = analysis_df[selected_features]
    y = analysis_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr_model = get_LogisticRegression()
    rf_model = get_RandomForest()

    lr_model.fit(X_train, y_train)
    acc_lr, y_pred_lr, cm_lr, roc_lr = eval_model(lr_model, X_test, y_test)

    rf_model.fit(X_train, y_train)
    acc_rf, y_pred_rf, cm_rf, roc_rf = eval_model(rf_model, X_test, y_test)

    sorted_imp = feature_importance(rf_model, selected_features)
    create_confusion_matrix(cm_rf)

    #Display Statistics 
    print("")
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Logistic Regression ROC: {roc_lr:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"Random Forest ROC: {roc_rf:.4f}")

    print("\nTop Features:")
    for feature, importance in sorted_imp:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()