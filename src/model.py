from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_LogisticRegression():
    """
    Logistic regression wrapped in a StandardScaler pipeline.

    LR is sensitive to feature scale — without scaling, lbfgs fails to
    converge even after 1000 iterations (you saw the warning). Wrapping in
    a Pipeline means the scaler is fit only on training data, so there is
    no leakage. C=0.1 applies stronger regularisation than the default.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=0.1,
            solver='lbfgs',
        ))
    ])


def get_RandomForest():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight={0: 1.0, 1: 5.0},
        random_state=42,
        max_depth=15,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
    )


def get_GradientBoosting(cat_indices=None):
    """
    Class weight raised from 1:5 to 1:8.

    The previous run showed GB recall at only 43%, meaning it missed 57%
    of patients who were actually readmitted. In a hospital setting, missing
    a high-risk patient (false negative) is far more costly than a false alarm
    (false positive). Raising the minority class weight forces the model to
    penalise false negatives more aggressively, improving recall at the cost
    of some precision — the correct clinical trade-off.

    Tune this number to match the hospital's actual cost ratio:
      - 1:5  -> balanced between catching cases and avoiding false alarms
      - 1:8  -> prioritises catching cases (recommended starting point)
      - 1:12 -> maximises recall, many false alarms (alarm fatigue risk)
    """
    return HistGradientBoostingClassifier(
        random_state=42,
        max_iter=1000,
        learning_rate=0.02,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.5,
        class_weight={0: 1.0, 1: 6.0},
        categorical_features=cat_indices,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        scoring='roc_auc',
    )


def get_StackingEnsemble(cat_indices=None):
    """
    Stacking ensemble: RF + GB as base learners, LR as meta-learner.
    The meta-learner uses a scaled pipeline for the same convergence reason.
    """
    estimators = [
        ('rf', get_RandomForest()),
        ('gb', get_GradientBoosting(cat_indices=cat_indices)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, max_iter=2000)),
        ]),
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )