from ucimlrepo import fetch_ucirepo
import pandas as pd

d = fetch_ucirepo(id=296)
X = d.data.features

med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed'
]

print(f"{'Column':<30} {'No':>8} {'Steady':>8} {'Up':>8} {'Down':>8}  {'% No':>8}")
print("-" * 75)
for col in med_cols:
    if col not in X.columns:
        continue
    vc = X[col].value_counts()
    no = vc.get('No', 0)
    steady = vc.get('Steady', 0)
    up = vc.get('Up', 0)
    down = vc.get('Down', 0)
    pct_no = no / len(X) * 100
    print(f"{col:<30} {no:>8} {steady:>8} {up:>8} {down:>8}  {pct_no:>7.1f}%")
