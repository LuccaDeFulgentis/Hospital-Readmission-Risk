from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_LogisticRegression():
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    return lr_model

def get_RandomForest():
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=5)
    return rf_model