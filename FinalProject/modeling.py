from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"{model_name} - Precision: {precision_score(y_test, y_pred, zero_division=1)}")
    print(f"{model_name} - Recall: {recall_score(y_test, y_pred, zero_division=1)}")
    print(f"{model_name} - F1: {f1_score(y_test, y_pred)}")
    print(f"{model_name} - ROC AUC: {roc_auc_score(y_test, y_pred)}")
    return y_pred
