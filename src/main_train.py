import logging
import sys
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)
from mlflow.models import infer_signature

from src.tuner import tune_hyperparameters
from src.trainer import train_model

sys.path.append('../')
from config.paths import CONFIG_PATH, PROCESSED_DATA_PATH, MODELS_PATH, ARTIFACTS_PATH, PLOTS_PATH
from utils.file_management import load_configs, load_data, save_pickle, load_pickle

# Setup logging con formato detallado y nivel INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("======== Starting training pipeline ========")

    configs = load_configs(
        CONFIG_PATH / "settings.yaml",
        CONFIG_PATH / "features.yaml"
        )

    settings = configs["settings"]
    features = configs["features"]

    training_parameters = settings["training_parameters"]
    use_optuna = training_parameters["use_optuna"]
    target = features["target"]

    logger.info(f"Target variable: '{target}'")

    best_params_path = ARTIFACTS_PATH / "best_params.yaml"
    dataset_path = PROCESSED_DATA_PATH / "data.csv"

    df = load_data(dataset_path)
    df.sort_values(by="Date", ascending=True, inplace=True)
    df.drop(["Date"], axis=1, inplace=True)
    logger.info(f"Dataset loaded with shape {df.shape}")

    X = df.drop(target, axis=1)
    y = df[target]

    logger.info(f"Splitting data into train and test sets (test_size={training_parameters['test_size']}, shuffle=False)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=training_parameters['test_size'], random_state=42, shuffle=False
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    if use_optuna:
        try:
            logger.info("Starting hyperparameter tuning with Optuna...")
            best_params = tune_hyperparameters(X_train, y_train, n_trials=training_parameters['n_trials'])
            save_pickle(best_params, best_params_path, 'Optuna best hyper parameters')
            logger.info(f"Hyperparameter tuning finished. Best params saved to {best_params_path}")
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise
    else:
        logger.warning("Optuna not used. Using default parameters.")
        best_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "random_state": 42
        }

    # Configurar experimento MLflow
    mlflow.set_experiment("RainPrediction")

    run_name = "LGBM_with_Optuna" if use_optuna else "LGBM_default"
    logger.info(f"Starting MLflow run with name '{run_name}'")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model": "LightGBM",
            "optuna_tuned": use_optuna,
            "developer": "robertg",
            "stage": "development"
        })

        logger.info("Logging hyperparameters to MLflow...")
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        logger.info("Training the model...")
        model_path = MODELS_PATH / "lgbm.joblib"
        model = train_model(X_train, y_train, best_params, model_path=model_path)
        logger.info(f"Model trained and saved at {model_path}")

        logger.info("Evaluating the model...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= training_parameters['pred_threshold']).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "log_loss": log_loss(y_test, y_pred_proba)
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            logger.info(f"Metric - {name}: {value:.4f}")

        report_str = classification_report(y_test, y_pred)
        mlflow.log_text(report_str, "classification_report.txt")

        # Save and log confusion matrix
        logger.info("Generating confusion matrix plot...")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
        cm_path = PLOTS_PATH / "confusion_matrix.png"
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        logger.info(f"Confusion matrix plot saved and logged at {cm_path}")

        # Save and log ROC curve plot
        logger.info("Generating ROC curve plot...")
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
        roc_path = PLOTS_PATH / "roc_curve.png"
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
        mlflow.log_artifact(str(roc_path), artifact_path="plots")
        logger.info(f"ROC curve plot saved and logged at {roc_path}")

        # Log model with signature
        logger.info("Logging model to MLflow with signature...")
        signature = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="LightGBM_RainPredictor"
        )
        logger.info("Model logged successfully.")

        # Log config artifacts
        mlflow.log_artifact(best_params_path, artifact_path="configs")
        logger.info(f"Best params artifact logged from {best_params_path}")

    logger.info("=== Training pipeline finished successfully ===")


if __name__ == "__main__":
    main()
