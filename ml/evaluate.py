from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  roc_auc_score,
  precision_score,
  recall_score,
  f1_score
)


class Evaluator:
  def evaluate(self, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n🎯 Key Metrics:")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))