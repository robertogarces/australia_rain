from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import classification_report
import xgboost as xgb
import os
import joblib
import pandas as pd
import yaml

def get_data(data_path: str) -> pd.DataFrame:
    return(pd.read_csv(data_path))

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def train_evaluate_save_model(X, y, model_path, test_size=0.2):
    """
    Splits the data, trains an XGBoost classifier, evaluates it, and saves the model.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        model_path (str): Path to save the trained model (e.g., 'models/xgb_model.joblib').
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        model: The trained XGBoost model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize and train the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Ensure the directory exists and save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model


if __name__ == "__main__":
    config = load_config('../config/config.yaml')
    X = get_data(f"{config['processed_data_path']}/X_train.csv")
    y = get_data(f"{config['processed_data_path']}/y_train.csv")

    model = train_evaluate_save_model(
        X, 
        y.values.ravel(),  # Asegura que y sea un vector 1D si viene como DataFrame
        model_path=f"{config['models_path']}/model.joblib", 
        test_size=0.2
    )

    