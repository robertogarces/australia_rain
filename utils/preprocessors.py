import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def binary_mapper(df, columns, mapping={'No': 0, 'Yes': 1}):
    """
    Maps specified binary categorical columns using a given mapping dictionary.

    Args:
        df (pd.DataFrame): DataFrame containing the columns.
        columns (list): List of column names to map.
        mapping (dict): Dictionary specifying the mapping.

    Returns:
        pd.DataFrame: DataFrame with transformed columns.
    """
    logger.info(f"Mapping columns {columns}. Map applied: {mapping}")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def filter_outliers_by_percentile(df, features, percentile=0.99):
    """
    Remove rows where any value in specified features exceeds the given percentile threshold.
    Only filters the upper tail (high values) as outliers.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of columns to check for outliers.
        percentile (float): Percentile threshold (between 0 and 1). Rows with values above this percentile are removed.

    Returns:
        pd.DataFrame: DataFrame without rows exceeding the specified percentile in any of the features.
    """

    df_clean = df.copy()
    outlier_indices = set()
    
    for col in features:
        threshold = df_clean[col].quantile(percentile)
        col_outliers = df_clean[df_clean[col] > threshold].index
        outlier_indices.update(col_outliers)
        
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop(index=outlier_indices)
    final_rows = df_clean.shape[0]
    logger.info(f"Removing outliers of features: {features}. Rows removed due to upper {100*(1-percentile):.2f}% outliers: {initial_rows - final_rows}")

    return df_clean


def remove_highly_correlated_features(df, target, threshold=0.9, strategy='nulls'):
    """
    Removes one of each pair of highly correlated features based on a selection strategy.

    Parameters:
        df (pd.DataFrame): The dataset.
        target (str): Name of the target variable.
        threshold (float): Correlation threshold.
        strategy (str): Strategy to drop one variable from each correlated pair. 
                        Options: 'nulls', 'target_corr'.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with reduced features.
            - list: Names of dropped features due to high correlation.
    """
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    corr_matrix = df_numeric.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()

    for col in upper_tri.columns:
        for row in upper_tri.index:
            corr_value = upper_tri.loc[row, col]
            if pd.notnull(corr_value) and corr_value > threshold:
                if row in to_drop or col in to_drop:
                    continue

                if strategy == 'nulls':
                    nulls_row = df[row].isnull().sum()
                    nulls_col = df[col].isnull().sum()
                    drop = row if nulls_row > nulls_col else col

                elif strategy == 'target_corr':
                    if target not in df.columns:
                        raise ValueError("Target column not found in DataFrame.")
                    corr_row = abs(df[[row, target]].corr().iloc[0, 1])
                    corr_col = abs(df[[col, target]].corr().iloc[0, 1])
                    drop = row if corr_row < corr_col else col

                else:
                    raise ValueError("Invalid strategy. Choose 'nulls' or 'target_corr'.")

                to_drop.add(drop)

    reduced_df = df.drop(columns=list(to_drop))

    return reduced_df, list(to_drop)


def impute_median_by_group(df, features, group_cols=['Month', 'Location']):
    """
    Impute missing values in specified columns using the median.
    First, impute using group-level median (by group_cols).
    If group-level median is missing, fall back to -1.

    Parameters:
        df (pd.DataFrame): Original dataframe.
        features (list): List of columns to impute.
        group_cols (list): Columns to group by for median calculation.

    Returns:
        tuple:
            - pd.DataFrame: Dataframe with imputed values.
            - dict: Nested dictionary of group-level medians {group_key: {feature: median}}.
    """
    df_imputed = df.copy()
    valid_features = [col for col in features if col in df_imputed.columns]

    if not valid_features:
        return df_imputed, {}

    group_medians_df = df_imputed.groupby(group_cols)[valid_features].median()

    # Convert index to tuple keys for JSON serialization
    group_medians = {
        '|'.join(map(str, index)): row.dropna().to_dict()
        for index, row in group_medians_df.iterrows()
    }

    # Apply imputations
    for feature in valid_features:
        df_imputed[feature] = df_imputed.groupby(group_cols)[feature].transform(lambda x: x.fillna(x.median()))
        df_imputed[feature] = df_imputed[feature].fillna(-1)

    return df_imputed, group_medians

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'
    
def label_encoder(df, columns):
    df = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # Optional: store encoders if you need to inverse transform later
    
    return df, encoders  


def fill_missing_categories(df, columns, fill_value='Missing'):
    """
    Fills missing values in the specified categorical columns with a given fill value.
    Silently skips columns that are not present in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to process.
        fill_value (str): Value used to fill missing values (default is 'Missing').

    Returns:
        pd.DataFrame: DataFrame with missing values filled in specified columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


def drop_null_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    logger.info(f"Dropping rows with null values in target '{target_col}'")
    before = df.shape[0]
    df = df.dropna(subset=[target_col])
    after = df.shape[0]
    logger.info(f"Dropped {before - after} rows with null target values. New shape: {df.shape}")
    return df


def process_dates(df: pd.DataFrame, date='Date') -> pd.DataFrame:
    logger.info("Converting 'Date' to datetime and extracting 'Month'")
    df[date] = pd.to_datetime(df[date], errors='coerce')
    if df[date].isnull().any():
        n_invalid = df[date].isnull().sum()
        logger.warning(f"Found {n_invalid} invalid 'Date' entries; dropping them.")
        df = df.dropna(subset=[date])
    df['Month'] = df[date].dt.month
    return df


def map_wind_direction(df: pd.DataFrame, wind_mapping: dict) -> pd.DataFrame:
    logger.info("Mapping wind direction categorical features to degrees")
    wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in wind_cols:
        if col in df.columns:
            mapped_col = f"{col}_deg"
            df[mapped_col] = df[col].map(wind_mapping)
            missing_count = df[mapped_col].isnull().sum()
            if missing_count > 0:
                logger.warning(f"{missing_count} missing mappings in '{col}', assigned NaN in '{mapped_col}'")
        else:
            logger.warning(f"Expected column '{col}' not found in dataframe")
    return df


def apply_label_encoders(df, encoders, fallback_value=-1):
    """
    Aplica múltiples LabelEncoders a un DataFrame, manejando categorías desconocidas.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        encoders (dict): Diccionario {columna: LabelEncoder}.
        fallback_value (int): Valor para categorías no vistas.

    Retorna:
        pd.DataFrame: DataFrame con columnas codificadas.
    """
    df_encoded = df.copy()
    for col, encoder in encoders.items():
        if col not in df_encoded.columns:
            continue

        known_classes = set(encoder.classes_)

        def encode_value(val):
            if pd.isna(val) or val not in known_classes:
                return fallback_value
            return encoder.transform([val])[0]

        df_encoded[col] = df_encoded[col].apply(encode_value)

    return df_encoded


def apply_group_median_imputation(df, group_medians, features, group_cols=['Month', 'Location'], fallback=-1):
    """
    Impute missing values in specified columns using precomputed group-level medians.

    Parameters:
        df (pd.DataFrame): DataFrame with missing values.
        group_medians (dict): Dict con medians per group in format {'Month|Location': {feature: value}}.
        features (list): Features to impute.
        group_cols (list): Group by features (default=['Month', 'Location']).
        fallback (float): Value to use if there's not group median.

    Returns:
        pd.DataFrame: DataFrame con los valores imputados.
    """
    df_imputed = df.copy()
    valid_features = [f for f in features if f in df.columns]

    if not valid_features:
        return df_imputed

    for idx, row in df.iterrows():
        key = '|'.join(map(str, [row[col] for col in group_cols]))
        for feature in valid_features:
            if pd.isna(row[feature]):
                value = group_medians.get(key, {}).get(feature, fallback)
                df_imputed.at[idx, feature] = value

    return df_imputed

