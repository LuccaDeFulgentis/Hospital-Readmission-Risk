from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    return acc, y_pred, cm, roc_auc