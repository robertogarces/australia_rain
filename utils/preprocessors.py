import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    
    print(f"Rows removed due to upper {100*(1-percentile):.2f}% outliers: {initial_rows - final_rows}")
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



def impute_with_minus_one(df, columns):
    df = df.copy()

    for col in columns:
        df[col] = df[col].fillna(-1)
    return df


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
