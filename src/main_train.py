import logging
import sys
import pandas as pd
sys.path.append('../')
import mlflow
from config.paths import CONFIG_PATH, PROCESSED_DATA_PATH, MODELS_PATH
from utils.config_loader import load_config
from tuner import tune_hyperparameters
from trainer import train_model
from evaluator import evaluate_model

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting training pipeline...")

    features = load_config(CONFIG_PATH / "features.yaml")
    target = features["target"]

    dataset_path = PROCESSED_DATA_PATH / "data.csv"
    df = pd.read_csv(dataset_path)
    df.sort_values(by='Date', ascending=True)
    df.drop(['Date'], axis=1, inplace=True)
    print(df.head(3))

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


    best_params, study = tune_hyperparameters(X_train, y_train, n_trials=5)

    with mlflow.start_run():
        mlflow.log_params(best_params)
        model_path = MODELS_PATH / 'lightGBM.joblib'
        print(best_params)
        model = train_model(X_train, y_train, best_params, model_path=model_path)
        test_auc = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("test_auc", test_auc)

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
