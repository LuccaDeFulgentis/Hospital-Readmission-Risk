import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def find_optimal_threshold(model, X, y):
    """
    Find the classification threshold that maximises F1 score.

    We search over the training set (not the test set) to avoid leakage.
    The default 0.5 threshold is almost never optimal for imbalanced data —
    this typically recovers several points of F1 for free.
    """
    proba = model.predict_proba(X)[:, 1]
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def eval_model(model, x_test, y_test, threshold=0.5):
    """
    Evaluate a fitted classifier and return a comprehensive set of metrics.

    Returns
    -------
    acc       : overall accuracy
    y_pred    : predicted labels at the given threshold
    cm        : confusion matrix
    roc_auc   : ROC AUC (discrimination ability across all thresholds)
    pr_auc    : Precision-Recall AUC (better metric for imbalanced classes)
    f1        : F1 score at the given threshold
    precision : Precision at the given threshold
    recall    : Recall (sensitivity) at the given threshold
    """
    proba = model.predict_proba(x_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return acc, y_pred, cm, roc_auc, pr_auc, f1, precision, recall