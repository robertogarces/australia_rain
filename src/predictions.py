import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append('../')

from config.paths import PROCESSED_DATA_PATH, CONFIG_PATH, ARTIFACTS_PATH, MODELS_PATH
from utils.file_management import load_configs, load_data, load_json, load_pickle
from utils.preprocessors import (
    binary_mapper,
    assign_season,
    process_dates,
    apply_group_median_imputation,
    map_wind_direction,
    apply_label_encoders
)

# Ensure all columns are visible when printing DataFrames
pd.set_option('display.max_columns', None)

def run_inference_pipeline():
    """Run the full inference pipeline on a sample dataset."""
    print("Loading inference dataset...")
    df = load_data(PROCESSED_DATA_PATH / "inference_sample.csv")

    print("Loading configuration files...")
    configs = load_configs(CONFIG_PATH / "settings.yaml", CONFIG_PATH / "features.yaml")
    config = configs["settings"]
    features = configs["features"]

    # Extract configuration values
    target = features['target']
    features_to_map = features['features_to_map']
    num_features = features['numeric_features']
    cat_features = features['categorical_features']
    model_features = features['model_features']
    wind_mapping = config['wind_mapping']

    # Drop rows with missing target values
    df.dropna(subset=[target], inplace=True)

    # Apply preprocessing steps
    print("Applying preprocessing steps...")
    df = binary_mapper(df, features_to_map)
    df = process_dates(df)

    dropped_features_path = ARTIFACTS_PATH / 'dropped_correlated_features.json'
    dropped_features = load_json(dropped_features_path)
    df.drop(dropped_features, axis=1, inplace=True)

    for col in cat_features:
        if col in df.columns:
            df[col].fillna('Missing', inplace=True)

    medians_path = ARTIFACTS_PATH / 'imputation_group_medians.json'
    medians = load_json(medians_path)
    df = apply_group_median_imputation(df, medians, num_features)

    df = map_wind_direction(df, wind_mapping)
    df["Season"] = df["Month"].apply(assign_season)

    encoder_path = ARTIFACTS_PATH / 'label_encoders.pkl'
    encoder = load_pickle(encoder_path)
    df = apply_label_encoders(df, encoder)

    # Prepare input features and target variable
    X = df[model_features].drop(target, axis=1)
    y = df[target]

    # Load trained model and make predictions
    print("Loading trained model and generating predictions...")
    model_path = MODELS_PATH / 'lgbm.joblib'
    model = joblib.load(model_path)
    preds = model.predict(X)

    # Evaluate model performance
    auc = roc_auc_score(y, preds)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, preds))

if __name__ == "__main__":
    run_inference_pipeline()
