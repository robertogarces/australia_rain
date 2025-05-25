import lightgbm as lgb
import joblib
from lightgbm import LGBMClassifier
import logging

logger = logging.getLogger(__name__)


def train_model(X, y, params=None, model_path=None):
    """
    Trains a LightGBM model using the provided parameters.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable.
        params (dict): LightGBM parameters.
        model_path (str, optional): Path to save the model.

    Returns:
        model: Trained LightGBM model.
    """
    
    if params==None:
        logger.info("Training LightGBM without optimized hyperparameters...")
        model = LGBMClassifier()
        model.fit(X, y)
    else:
        logger.info("Training LightGBM model with optimized hyperparameters...")
        model = LGBMClassifier(**params)
        model.fit(X, y)

    logger.info("LightGBM model trained successfully")
    if model_path:
        joblib.dump(model, model_path)
        logger.info(f"LightGBM model saved at {model_path}")

    return model
