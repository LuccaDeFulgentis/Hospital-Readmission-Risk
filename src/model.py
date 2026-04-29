from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def get_LogisticRegression():
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    return lr_model

def get_RandomForest():
    rf_model = RandomForestClassifier(n_estimators=150, class_weight={0: 1.0, 1: 5.0}, random_state=42, max_depth=15, min_samples_leaf=5)
    return rf_model

def get_GradientBoosting(cat_indices=None):
    gb_model = HistGradientBoostingClassifier(
        random_state=42, 
        max_iter=800, 
        learning_rate=0.03, 
        max_depth=10, 
        min_samples_leaf=20, 
        l2_regularization=0.1,
        class_weight={0: 1.0, 1: 5.0}, 
        categorical_features=cat_indices
    )
    return gb_model