import pandas as pd

import sys
sys.path.append('../')

from config.paths import RAW_DATA_PATH, CONFIG_PATH

from utils.config_loader import load_config
from utils.preprocessors import binary_mapper, filter_outliers_by_percentile, remove_highly_correlated_features, impute_median_by_group, impute_with_minus_one

pd.set_option('display.max_columns', None)

# Load dataset
dataset_path = RAW_DATA_PATH / "weatherAUS.csv"
df = pd.read_csv(dataset_path)

# Load config
config_path = CONFIG_PATH / "settings.yaml"
config = load_config(config_path)
target = config['target']
pp_params = config['preprocessing_parameters']
features_to_map = config['features_to_map']
num_features = config['numeric_features']
cat_features = config['categorical_features']
location_nulls = config['location_nulls']
wind_mapping = config['wind_mapping']

# Drop target labels that are null
df.dropna(subset=[target], inplace=True)

# Convert both target and RainToday feature into boolean
df = binary_mapper(df, features_to_map)

# Convert 'date' into datetime
df['Date'] = pd.to_datetime(df['Date'])
# Create 'Month' feature
df['Month'] = df['Date'].dt.month

# Remove outliers
features_with_outliers = config['features_with_outliers']
df = filter_outliers_by_percentile(df, features_with_outliers, pp_params['outliers_percentile'])

# Remove highly correlated features
df, dropped = remove_highly_correlated_features(df, target=target, threshold=pp_params['correlation_threshold'], strategy=pp_params['correlation_strategy'])
print("Dropped features:", dropped)

# Imputation
for col in cat_features:
    df[col] = df[col].fillna('Missing')

df = impute_median_by_group(df, num_features)
df = impute_with_minus_one(df, location_nulls)

# Mapping
df['WindGustDir_deg'] = df['WindGustDir'].map(wind_mapping)
df['WindDir9am_deg'] = df['WindDir9am'].map(wind_mapping)
df['WindDir3pm_deg'] = df['WindDir3pm'].map(wind_mapping)












