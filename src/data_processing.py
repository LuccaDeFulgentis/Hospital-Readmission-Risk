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
    diabetes = fetch_ucirepo(id=296)

    X = diabetes.data.features
    y = diabetes.data.targets

    df = pd.concat([X, y], axis=1)
    
    # Filter out missing readmitted (if any)
    df = df.dropna(subset=['readmitted'])
    
    # Filter out hospice/dead patients (cannot be readmitted)
    invalid_discharge = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(invalid_discharge)]

    print("Data loaded. Shape:", df.shape)

    # 2. Data Processing
    print("Processing data...")
    # Convert 'readmitted' to a binary target: 1 if '<30', 0 otherwise
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df['age_num'] = df['age'].apply(parse_age)

    # Drop non-predictive columns. Keep medical_specialty and diag_1/2/3
    cols_to_drop = ['readmitted', 'encounter_id', 'patient_nbr', 'weight', 'payer_code']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Force some IDs to be treated as categorical instead of continuous
    id_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    for c in id_cols:
        if c not in cat_cols and c in df.columns:
            cat_cols.append(c)

    # Fill NaNs in categorical columns to 'Missing'
    df[cat_cols] = df[cat_cols].fillna('Missing')

    # Separate high-cardinality features for TargetEncoding, and low-cardinality for OrdinalEncoding
    high_card = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty']
    low_card = [c for c in cat_cols if c not in high_card]

    selected_features = [c for c in df.columns if c != 'target']
    analysis_df = df # HistGradientBoosting handles continuous NaNs natively!

    return analysis_df, selected_features, high_card, low_card
