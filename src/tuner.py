import optuna
import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

def tune_hyperparameters(X, y, n_trials=50, random_state=42):

    logger.info("Running hyperparameter tuning...")

    def objective(trial):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 20, 150),
            'max_depth': trial.suggest_int("max_depth", 4, 12),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 60),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.6, 1.0),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.6, 1.0),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 10),
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0001, 10.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0001, 10.0, log=True),
            'min_split_gain': trial.suggest_float("min_split_gain", 0.0, 1.0),
            'random_state': random_state,
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        auc_scores = []

        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = LGBMClassifier(**params, early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='auc',
            )

            y_pred = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_pred)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best trial AUC: {study.best_value:.4f}")

    return study.best_params, study.best_value

