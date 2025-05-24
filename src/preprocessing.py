import pandas as pd
import json
import sys
import pickle
sys.path.append('../')

from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH, CONFIG_PATH, ARTIFACTS_PATH

from utils.config_loader import load_config
from utils.preprocessors import binary_mapper, filter_outliers_by_percentile, remove_highly_correlated_features, impute_median_by_group, impute_with_minus_one, assign_season, label_encoder, fill_missing_categories

pd.set_option('display.max_columns', None)

# Load dataset
dataset_path = RAW_DATA_PATH / "weatherAUS.csv"
df = pd.read_csv(dataset_path)

# Load config and features
config_path = CONFIG_PATH / "settings.yaml"
config = load_config(config_path)
features_path = CONFIG_PATH / "features.yaml"
features = load_config(features_path)

pp_params = config['preprocessing_parameters']
wind_mapping = config['wind_mapping']

target = features['target']
features_to_map = features['features_to_map']
num_features = features['numeric_features']
cat_features = features['categorical_features']
location_nulls = features['location_nulls']
features_with_outliers = features['features_with_outliers']

# Drop target labels that are null
df.dropna(subset=[target], inplace=True)

# Convert both target and RainToday feature into boolean
df = binary_mapper(df, features_to_map)

# Convert 'date' into datetime
df['Date'] = pd.to_datetime(df['Date'])
# Create 'Month' feature
df['Month'] = df['Date'].dt.month

# Remove outliers
df = filter_outliers_by_percentile(df, features_with_outliers, pp_params['outliers_percentile'])


# Remove highly correlated features
df, dropped_features = remove_highly_correlated_features(df, target=target, threshold=pp_params['correlation_threshold'], strategy=pp_params['correlation_strategy'])
print("Dropped features:", dropped_features)
dropped_features_path = ARTIFACTS_PATH / "dropped_correlated_features.json"
with open(dropped_features_path, "w") as f:
    json.dump(dropped_features, f, indent=2)


# Imputation
df = fill_missing_categories(df, cat_features)
df, group_medians = impute_median_by_group(df, num_features)
group_medians_path = ARTIFACTS_PATH / 'imputation_group_medians.json'
with open(group_medians_path, "w") as f:
    json.dump(group_medians, f, indent=2)
# READING OF THE JSON:
#month, location = key.split('|')
#month = int(month)


#df = impute_with_minus_one(df, location_nulls)

# Mapping
df['WindGustDir_deg'] = df['WindGustDir'].map(wind_mapping)
df['WindDir9am_deg'] = df['WindDir9am'].map(wind_mapping)
df['WindDir3pm_deg'] = df['WindDir3pm'].map(wind_mapping)



df['Season'] = df['Month'].apply(assign_season)
df, label_encoders = label_encoder(df, cat_features)
label_encoders_path = ARTIFACTS_PATH / 'label_encoders.pkl'
with open(label_encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)

df.to_csv(PROCESSED_DATA_PATH / "data.csv", index=False)



