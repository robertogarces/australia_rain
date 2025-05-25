import pandas as pd
import json
import pickle
import logging
import sys
sys.path.append('../')

from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH, CONFIG_PATH, ARTIFACTS_PATH
from utils.config_loader import load_config
from utils.preprocessors import (
    binary_mapper,
    filter_outliers_by_percentile,
    remove_highly_correlated_features,
    impute_median_by_group,
    fill_missing_categories,
    label_encoder,
    assign_season
)

# Setup logging con formato profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

def main():
    logger.info("=== Starting preprocessing pipeline ===")
    try:
        dataset_path = RAW_DATA_PATH / "weatherAUS.csv"
        logger.info(f"Loading raw dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"Raw dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    try:
        logger.info("Loading configuration files...")
        config_path = CONFIG_PATH / "settings.yaml"
        config = load_config(config_path)
        features_path = CONFIG_PATH / "features.yaml"
        features = load_config(features_path)
        logger.info("Configuration files loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        raise

    # Extract configs
    pp_params = config['preprocessing_parameters']
    wind_mapping = config['wind_mapping']

    target = features['target']
    features_to_map = features['features_to_map']
    num_features = features['numeric_features']
    cat_features = features['categorical_features']
    location_nulls = features['location_nulls']
    features_with_outliers = features['features_with_outliers']

    logger.info(f"Target variable: '{target}'")
    logger.info(f"Initial dataset columns: {list(df.columns)}")

    # Drop null targets
    logger.info(f"Dropping rows with null values in target '{target}'")
    before_drop = df.shape[0]
    df.dropna(subset=[target], inplace=True)
    after_drop = df.shape[0]
    logger.info(f"Dropped {before_drop - after_drop} rows with null target values. New shape: {df.shape}")

    # Binary mapping of target and features
    logger.info(f"Mapping binary features: {features_to_map}")
    df = binary_mapper(df, features_to_map)

    # Date processing
    logger.info("Converting 'Date' to datetime and extracting 'Month'")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().any():
        n_null_dates = df['Date'].isnull().sum()
        logger.warning(f"Found {n_null_dates} invalid date entries, dropping them.")
        df = df.dropna(subset=['Date'])
    df['Month'] = df['Date'].dt.month

    # Remove outliers
    logger.info(f"Filtering outliers for features {features_with_outliers} using percentile {pp_params['outliers_percentile']}")
    df = filter_outliers_by_percentile(df, features_with_outliers, pp_params['outliers_percentile'])
    logger.info(f"Dataset shape after outlier removal: {df.shape}")

    # Remove highly correlated features
    logger.info(f"Removing highly correlated features with threshold {pp_params['correlation_threshold']} and strategy '{pp_params['correlation_strategy']}'")
    df, dropped_features = remove_highly_correlated_features(
        df,
        target=target,
        threshold=pp_params['correlation_threshold'],
        strategy=pp_params['correlation_strategy']
    )
    logger.info(f"Dropped correlated features: {dropped_features}")

    # Save dropped features list
    dropped_features_path = ARTIFACTS_PATH / "dropped_correlated_features.json"
    try:
        with open(dropped_features_path, "w") as f:
            json.dump(dropped_features, f, indent=2)
        logger.info(f"Dropped correlated features saved to {dropped_features_path}")
    except Exception as e:
        logger.error(f"Error saving dropped features: {e}")
        raise

    # Fill missing categories
    logger.info("Filling missing categorical values")
    df = fill_missing_categories(df, cat_features)

    # Impute numeric features by group median
    logger.info("Imputing numeric features by group median")
    df, group_medians = impute_median_by_group(df, num_features)
    group_medians_path = ARTIFACTS_PATH / 'imputation_group_medians.json'
    try:
        with open(group_medians_path, "w") as f:
            json.dump(group_medians, f, indent=2)
        logger.info(f"Group median imputation values saved to {group_medians_path}")
    except Exception as e:
        logger.error(f"Error saving group median imputations: {e}")
        raise

    # Wind direction mapping
    logger.info("Mapping wind direction categorical features to degrees")
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if col in df.columns:
            mapped_col = f"{col}_deg"
            df[mapped_col] = df[col].map(wind_mapping)
            n_missing = df[mapped_col].isnull().sum()
            if n_missing > 0:
                logger.warning(f"{n_missing} missing mappings in '{col}', assigned NaN in '{mapped_col}'")
        else:
            logger.warning(f"Expected column '{col}' not found in dataframe")

    # Assign season based on month
    logger.info("Assigning seasons based on 'Month' feature")
    df['Season'] = df['Month'].apply(assign_season)

    # Label encode categorical features
    logger.info(f"Label encoding categorical features: {cat_features}")
    df, label_encoders = label_encoder(df, cat_features)

    # Save label encoders
    label_encoders_path = ARTIFACTS_PATH / 'label_encoders.pkl'
    try:
        with open(label_encoders_path, "wb") as f:
            pickle.dump(label_encoders, f)
        logger.info(f"Label encoders saved to {label_encoders_path}")
    except Exception as e:
        logger.error(f"Error saving label encoders: {e}")
        raise

    # Save processed dataframe
    processed_path = PROCESSED_DATA_PATH / "data.csv"
    try:
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

    logger.info("=== Preprocessing pipeline completed successfully ===")

if __name__ == "__main__":
    main()
