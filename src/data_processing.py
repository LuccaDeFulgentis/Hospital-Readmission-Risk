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


def engineer_features(df):
    """
    total_procedure_burden: sum of lab and non-lab procedures.
    Captures overall clinical intensity in a single feature.

    on_insulin removed — confirmed redundant with the raw 'insulin' column
    which the model already uses. Permutation importance was 0.0000.
    """
    if 'num_lab_procedures' in df.columns and 'num_procedures' in df.columns:
        df['total_procedure_burden'] = (
            df['num_lab_procedures'].fillna(0) + df['num_procedures'].fillna(0)
        )
    return df


def load_data():
    diabetes = fetch_ucirepo(id=296)

    X = diabetes.data.features
    y = diabetes.data.targets

    df = pd.concat([X, y], axis=1)
    df = df.dropna(subset=['readmitted'])

    # Remove patients who died or went to hospice — they cannot be readmitted
    invalid_discharge = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(invalid_discharge)]

    print("Data loaded. Shape:", df.shape)
    print("Processing data...")

    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df['age_num'] = df['age'].apply(parse_age)

    # Columns confirmed to carry no usable signal, grouped by reason:

    # Administrative / identifier columns
    admin_cols = ['readmitted', 'encounter_id', 'patient_nbr', 'weight', 'payer_code', 'age']

    # Medication columns that are 100% or near-100% "No" — zero variance
    # Counts verified with value_counts() on the full 101,766-row dataset
    constant_meds = [
        'examide',                   # 100.0% No
        'citoglipton',               # 100.0% No
        'acetohexamide',             # 100.0% No (1 non-No row)
        'tolbutamide',               # 100.0% No (23 non-No rows)
        'miglitol',                  # 100.0% No (38 non-No rows)
        'troglitazone',              # 100.0% No (3 non-No rows)
        'tolazamide',                # 100.0% No (39 non-No rows)
        'glipizide-metformin',       #  99.9% No
        'glimepiride-pioglitazone',  #  99.9% No
        'metformin-rosiglitazone',   #  99.9% No
        'metformin-pioglitazone',    #  99.9% No
        'chlorpropamide',            #  99.9% No (86 non-No rows)
        'nateglinide',               #  99.3% No (703 non-No rows)
        'glyburide-metformin',       #  99.3% No (706 non-No rows)
        'repaglinide',               #  98.5% No — also showed negative permutation importance
    ]

    # Features with confirmed negative permutation importance — the model
    # learned spurious patterns from these; shuffling them improves predictions
    negative_importance = [
        'medical_specialty',   # high cardinality + many missing values
    ]

    cols_to_drop = admin_cols + constant_meds + negative_importance
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = engineer_features(df)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    id_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    for c in id_cols:
        if c not in cat_cols and c in df.columns:
            cat_cols.append(c)

    df[cat_cols] = df[cat_cols].fillna('Missing')

    high_card = ['diag_1', 'diag_2', 'diag_3']
    low_card = [c for c in cat_cols if c not in high_card]

    selected_features = [c for c in df.columns if c != 'target']

    return df, selected_features, high_card, low_card