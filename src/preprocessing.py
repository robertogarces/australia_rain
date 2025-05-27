import logging
import sys
sys.path.append('../')

from config.paths import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    CONFIG_PATH, 
    ARTIFACTS_PATH
    )
from utils.file_management import (
    load_data, 
    load_configs, 
    save_json, 
    save_pickle, 
    save_dataframe
    )
from utils.preprocessors import (
    binary_mapper,
    filter_outliers_by_percentile,
    remove_highly_correlated_features,
    impute_median_by_group,
    fill_missing_categories,
    label_encoder,
    assign_season,
    drop_null_target,
    process_dates,
    map_wind_direction
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_preprocessing_pipeline() -> None:
    logger.info("=== Starting preprocessing pipeline ===")

    # === Load raw data and configuration files ===
    df = load_data(RAW_DATA_PATH / "training_data.csv")
    config, features = load_configs(
        CONFIG_PATH / "settings.yaml", 
        CONFIG_PATH / "features.yaml"
    )

    # === Extract relevant parameters and feature groups ===
    pp_params = config["preprocessing_parameters"]
    wind_mapping = config["wind_mapping"]

    target = features["target"]
    features_to_map = features["features_to_map"]
    num_features = features["numeric_features"]
    cat_features = features["categorical_features"]
    features_with_outliers = features["features_with_outliers"]

    logger.info(f"Target variable: '{target}'")
    logger.info(f"Initial dataset columns: {list(df.columns)}")

    # === Preprocessing steps ===

    # Drop rows with missing target
    df = drop_null_target(df, target)

    # Map binary categorical features
    df = binary_mapper(df, features_to_map)

    # Convert date, extract month, and remove invalid dates
    df = process_dates(df)

    # Filter outliers using percentile threshold
    df = filter_outliers_by_percentile(
        df, features_with_outliers, pp_params["outliers_percentile"]
    )
    logger.info(f"Dataset shape after outlier removal: {df.shape}")

    # Remove highly correlated features
    df, dropped_features = remove_highly_correlated_features(
        df,
        target=target,
        threshold=pp_params["correlation_threshold"],
        strategy=pp_params["correlation_strategy"]
    )
    logger.info(f"Dropped correlated features: {dropped_features}")
    save_json(dropped_features, ARTIFACTS_PATH / "dropped_correlated_features.json", "Dropped correlated features")

    # Fill missing values in categorical columns
    df = fill_missing_categories(df, cat_features)

    # Impute missing numeric values using group medians
    df, group_medians = impute_median_by_group(df, num_features)
    save_json(group_medians, ARTIFACTS_PATH / "imputation_group_medians.json", "Group median imputation values")

    # Map wind direction categories to degrees
    df = map_wind_direction(df, wind_mapping)

    # Assign season based on extracted month
    df["Season"] = df["Month"].apply(assign_season)

    # Encode categorical features
    df, label_encoders = label_encoder(df, cat_features)
    save_pickle(label_encoders, ARTIFACTS_PATH / "label_encoders.pkl", "Label encoders")

    # Save the final processed dataset
    save_dataframe(df, PROCESSED_DATA_PATH / "data.csv")

    logger.info("=== Preprocessing pipeline completed successfully ===")

if __name__ == "__main__":
    run_preprocessing_pipeline()
