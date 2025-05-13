import preprocessors as pp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
import joblib
import os


def split_data_and_save(data_path, test_size):
    """
    Splits a DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test split (between 0 and 1).

    Returns:
        tuple: (train_df, test_df) Training and testing DataFrames.
    """
    df = pd.read_csv(f"{data_path}/weatherAUS.csv")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(f"{data_path}/train_weatherAUS.csv", index=False)
    test_df.to_csv(f"{data_path}/test_weatherAUS.csv", index=False)


def load_features(features_path: str):
    with open(features_path, 'r') as file:
        return yaml.safe_load(file)

def get_data(data_path: str) -> pd.DataFrame:
    return(pd.read_csv(data_path))

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

def preprocessing(config):
    """
    Executes the data preprocessing pipeline and returns X, y, and the fitted pipeline.

    Parameters:
        config (dict): Configuration dictionary with paths and preprocessing parameters.

    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
        preprocessing_pipeline (Pipeline): Fitted preprocessing pipeline.
    """

    split_data_and_save(config['raw_data_path'], test_size=0.2)

    # Load data
    df = get_data(f"{config['raw_data_path']}/train_weatherAUS.csv")

    # Drop unnecessary features
    df = df.drop(columns=config.get('features_to_drop', []))

    # Define the preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('missing', pp.MissingValueHandler(threshold=config.get('missing_data_threshold', 0.2))),
        ('binary', pp.BinaryMapper(columns=config.get('features_to_map', []))),
        ('outliers', pp.OutlierRemover(features=config.get('features_with_outliers', []))),
        ('scaling', pp.NumericalScaler(exclude=config['target'])),
        ('encoding', pp.CategoricalEncoder()),
    ])

    # Fit and transform the pipeline
    df_processed = preprocessing_pipeline.fit_transform(df)

    joblib.dump(preprocessing_pipeline, f"{config['artifacts_path']}/preprocessing_pipeline.joblib")

    # Separate features and target
    X = df_processed.drop(columns=[config['target']])
    y = df_processed[config['target']]

    X.to_csv(f"{config['processed_data_path']}/X_train.csv", index=False)
    y.to_csv(f"{config['processed_data_path']}/y_train.csv", index=False)


if __name__ == "__main__":
    config = load_config('../config/config.yaml')
    preprocessing(config)
