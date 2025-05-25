from sklearn.metrics import classification_report, roc_auc_score
import logging
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f'AUC: {auc}')
    print(f'Classification Report: /n/n{report}')

    return auc
