import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo

def main():
    print("Starting data ingestion...")
    # 1. Fetch data
    # fetch dataset 296: Diabetes 130-US hospitals for years 1999-2008
    diabetes = fetch_ucirepo(id=296)

    # data (as pandas dataframes)
    X = diabetes.data.features
    y = diabetes.data.targets

    df = pd.concat([X, y], axis=1)
    
    # Filter out missing readmitted (if any)
    df = df.dropna(subset=['readmitted'])
    
    print("Data loaded. Shape:", df.shape)

    # 2. Data Processing
    print("Processing data...")
    # Convert 'readmitted' to a binary target: 1 if '<30', 0 otherwise
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # Select features based on the proposal
    features = ['age', 'time_in_hospital', 'num_procedures', 'num_medications']
    
    # Process 'age' from categorical '[0-10)' to numerical midpoint
    def parse_age(age_str):
        if pd.isna(age_str):
            return np.nan
        age_str = str(age_str).replace('[', '').replace(')', '')
        parts = age_str.split('-')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        return np.nan

    df['age_num'] = df['age'].apply(parse_age)
    
    # Final feature set
    selected_features = ['age_num', 'time_in_hospital', 'num_procedures', 'num_medications']
    analysis_df = df[selected_features + ['target']].copy()
    
    # Drop rows with NaN in these specific features
    analysis_df = analysis_df.dropna()
    print("Processed data shape:", analysis_df.shape)

    # 3. Visualizations
    print("Generating visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    # Setting up the visual style
    sns.set_theme(style="whitegrid")

    # Plot 1: Readmission Rates by Age
    plt.figure(figsize=(10, 6))
    sns.histplot(data=analysis_df, x='age_num', hue='target', multiple="stack", bins=10)
    plt.title('Patient Age Distribution by 30-Day Readmission')
    plt.xlabel('Age (Midpoint of 10-year bucket)')
    plt.ylabel('Count')
    plt.legend(title='Readmitted < 30 Days', labels=['Yes (1)', 'No (0)'])
    plt.savefig('visualizations/age_distribution.png')
    plt.close()

    # Plot 2: Time in Hospital vs Readmission
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='time_in_hospital', data=analysis_df)
    plt.title('Time in Hospital vs 30-Day Readmission Risk')
    plt.xlabel('Readmitted < 30 Days (0 = No, 1 = Yes)')
    plt.ylabel('Days in Hospital')
    plt.savefig('visualizations/time_in_hospital_boxplot.png')
    plt.close()

    # 4. Modeling
    print("Training models...")
    X_train = analysis_df[selected_features]
    y_train = analysis_df['target']

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Fit Logistic Regression Model
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    # Fit Random Forest (mentioned in proposal)
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Feature importances
    importances = rf_model.feature_importances_
    imp_dict = dict(zip(selected_features, importances))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

    # Plot 3: Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[x[1] for x in sorted_imp], y=[x[0] for x in sorted_imp], hue=[x[0] for x in sorted_imp], palette="viridis", legend=False)
    plt.title('Random Forest Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    
    # Plot 4: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    # 5. Generate Report
    print("Generating Rough Draft Report...")
    report = f"""# Project Check-In (March) - Rough Draft

## 1. Preliminary Data Visualizations
We created a few initial charts to see how some of our main features relate to our target variable (whether a patient gets readmitted within 30 days). 

1. **Age Distribution by Readmission**: 
   ![Age Distribution by Readmission](visualizations/age_distribution.png)
   We created a stacked histogram showing the age distribution of the patients in our dataset. We converted their original age brackets into numeric midpoints, so the x-axis represents age and the y-axis is the total patient count. Looking at the graph, you can see our data skews heavily older, specifically in the 60-80 range. The colors inside each bar split the patients by whether they were readmitted within 30 days or not. We used a stacked chart here so we could look at the actual *proportion* of readmissions for each age group, not just the raw counts. This helps us see if age itself really drives the readmission risk.
2. **Time in Hospital vs. Readmission Risk**:
   ![Time in Hospital vs. Readmission Risk](visualizations/time_in_hospital_boxplot.png)
   We made a boxplot to see if the length of a patient's initial hospital stay relates to their chance of coming back within 30 days. We split the patients into two groups on the x-axis: 0 for not readmitted, and 1 for readmitted. The y-axis shows how many days they were in the hospital. We chose a boxplot so we could compare the medians and the spread of those two groups right next to each other. Our initial hypothesis was that a longer initial stay usually means the patient had a more severe or complex case, which should increase the risk of a bounce-back. However, looking at the graph, the two boxes are essentially identical—both groups have a median of exactly 4 days, and the exact same spread. This is actually a very important finding: it shows us that in this dataset, the initial time spent in the hospital does *not* correlate with 30-day readmission risk, which disproves our initial theory.

## 2. Data Processing Progress
Yes, we successfully downloaded the [Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository. We actually used their official Python library (`ucimlrepo`) to pull the data directly into our pandas dataframes, which made the data ingestion super easy.

Here's exactly what we did to clean and prep the data so far:
- **Filtering features:** We narrowed down the 50ish columns in the raw dataset to just the ones we proposed in our initial plan: `age`, `time_in_hospital`, `num_procedures`, and `num_medications`.
- **Handling 'age':** The age data came in weird string buckets like `[70-80)`. To make this usable for machine learning, we wrote a function to strip the brackets and calculate the numeric midpoint (so `[70-80)` becomes `75.0`).
- **Formatting the target variable ('readmitted'):** The original labels were `NO`, `>30`, and `<30`. Since our specific goal is predicting 30-day readmissions, we mapped `<30` to `1` (positive class) and clumped `NO` and `>30` together as `0` (negative class). 
- **Handling missing data:** After selecting our features, we dropped any rows that had missing values in those specific columns to keep things clean for our baseline models. After doing all this, we were left with a solid dataset of about 101,766 patient records.

## 3. Modeling Methods
- **What we're predicting:** A binary classification model to predict if a diabetic patient will bounce back to the hospital within 30 days of leaving.
- **Features used:** The cleaned numeric versions of `age`, `time_in_hospital`, `num_procedures`, and `num_medications`.
- **Why we chose these:** We picked these starting features because logically, older age and longer time spent in the hospital usually mean someone is sicker. Combining that with the number of procedures and medications gives us a rough proxy for how complicated their case is. 
- **Our Process:** We used `scikit-learn` to randomly split the data into two sets: 80% for training the models and 20% reserved for testing. 
  - **Lines 1 - {len(X_train)}**: Training Data ({len(X_train)} records)
  - **Lines {len(X_train) + 1} - {len(X_train) + len(X_test)}**: Testing Data ({len(X_test)} records)
  
  We built two baseline models to start: a **Logistic Regression** and a **Random Forest Classifier**. Since "readmitted within 30 days" is actually the minority class (most people don't come back that fast), we made sure to balance the class weights during training so the models wouldn't just default to guessing "0" for everybody.

## 4. Preliminary Results & Interpretation
We ran both of our baseline models against our 20% test partition. Here are the initial test accuracy scores:
- **Logistic Regression Test Accuracy**: {acc_lr:.2%}
- **Random Forest Test Accuracy**: {acc_rf:.2%}

**Model Performance Visualization:**
To better interpret our model, we plotted a Confusion Matrix for the Random Forest evaluating against the test partition:
![Confusion Matrix](visualizations/confusion_matrix.png)
As visualized above, the model has a notable amount of false positives (predicting readmission when it doesn't happen). This is expected behavior given that we balanced the class weights to force the simple model to better identify the minority positive class. 

**Top Characteristics Increasing Risk:**
Based on our Random Forest model, we extracted the feature importances to analyze the top characteristics that increase readmission risk. 
![Feature Importances](visualizations/feature_importance.png)
In our current dataset, they are:
1. **`{sorted_imp[0][0]}`** (Importance Score: {sorted_imp[0][1]:.3f})
2. **`{sorted_imp[1][0]}`** (Importance Score: {sorted_imp[1][1]:.3f})

**Thoughts on the results:**
To be honest, the accuracy right now isn't amazing, but that makes sense for a rough draft baseline. Predicting readmission is a notoriously hard problem, and right now we're only feeding the models four very high-level features. 

Since we told the models to treat both classes equally (by balancing the weights), they are trying really hard to correctly guess the minority class ("Yes, they will be readmitted"), which drags down the overall broad accuracy score. For the rest of the project, our plan is to start feeding the models more detailed clinical features like `HbA1c` test results or specific diabetes medications. We're also going to start looking at better evaluation metrics for imbalanced datasets—like Precision, Recall, and ROC AUC—instead of just staring at the raw accuracy number.
"""

    with open('CheckIn_Draft.md', 'w') as f:
        f.write(report)
        
    print("Check-in draft generated at CheckIn_Draft.md")

if __name__ == "__main__":
    main()
