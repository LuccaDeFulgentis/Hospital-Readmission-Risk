import os
import matplotlib.pyplot as plt
import seaborn as sns

def feature_importance(rf_model, selected_features):
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style="whitegrid")
    importances = rf_model.feature_importances_
    imp_dict = dict(zip(selected_features, importances))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[x[1] for x in sorted_imp], y=[x[0] for x in sorted_imp], hue=[x[0] for x in sorted_imp], palette="viridis", legend=False)
    plt.title('Random Forest Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

    return sorted_imp

def create_confusion_matrix(cm):
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()