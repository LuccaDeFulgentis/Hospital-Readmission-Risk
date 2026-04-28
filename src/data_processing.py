import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def parse_age(age_str):
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).replace('[', '').replace(')', '')
    parts = age_str.split('-')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return np.nan

def load_data():
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
    df['age_num'] = df['age'].apply(parse_age)
    
    # Select features based on the proposal
    selected_features = ['age_num', 'time_in_hospital', 'num_procedures', 'num_medications']

    markers_features = [
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
    'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed'
    ]

    #Creates features for each marker. ie change_yes, change_no
    df_marker = df[markers_features]
    df_markers = pd.get_dummies(df_marker, drop_first=True)

    analysis_df = pd.concat([df[selected_features], df_markers, df[['target']]], axis=1).dropna()
    
    selected_features = selected_features + list(df_markers.columns)

    return analysis_df, selected_features
