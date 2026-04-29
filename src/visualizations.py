import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def feature_importance(importances, selected_features):
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style="whitegrid")
    imp_dict = dict(zip(selected_features, importances))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[x[1] for x in sorted_imp[:20]], y=[x[0] for x in sorted_imp[:20]], hue=[x[0] for x in sorted_imp[:20]], palette="viridis", legend=False)
    plt.title('Gradient Boosting Feature Importances (Top 20)')
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
    plt.title('Gradient Boosting Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

def create_top_feature_plot(df, top_feature):
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    # Check cardinality to decide plot type
    unique_vals = df[top_feature].dropna().unique()
    
    if len(unique_vals) <= 10:
        # Categorical or low cardinality
        sns.barplot(x=top_feature, y='target', data=df, errorbar=None, palette="viridis", hue=top_feature, legend=False)
        plt.title(f'Readmission Rate by {top_feature}')
        plt.ylabel('Average Readmission Rate')
        plt.xlabel(top_feature)
    else:
        # Continuous: bin it into quintiles for plotting
        df_plot = df.copy()
        df_plot['feature_binned'] = pd.qcut(df_plot[top_feature], q=5, duplicates='drop')
        sns.barplot(x='feature_binned', y='target', data=df_plot, errorbar=None, palette="viridis", hue='feature_binned', legend=False)
        plt.title(f'Readmission Rate by {top_feature} (Binned)')
        plt.ylabel('Average Readmission Rate')
        plt.xlabel(f'{top_feature} Quintiles')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.savefig('visualizations/top_feature_trend.png')
    plt.close()