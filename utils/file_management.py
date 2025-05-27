import yaml
from pathlib import Path
import logging
import pandas as pd
import json
import pickle
from typing import Tuple, Dict

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load a YAML file and returns a dictionary"""
    logger.info(f"Loading yaml file from {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw dataset from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Raw dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def load_configs(*paths: Path) -> Dict[str, Dict]:
    """
    Loads multiple YAML configuration files and returns a dictionary where keys are filenames (without extension)
    and values are the parsed YAML content.
    """
    logger.info("Loading configuration files...")
    configs = {}
    try:
        for path in paths:
            config_name = path.stem  # e.g., "settings"
            configs[config_name] = load_config(path)
        logger.info("Configuration files loaded successfully.")
        return configs
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        raise


def save_json(data: dict, path: Path, description: str = None) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"{description} saved to {path}")
    except Exception as e:
        logger.error(f"Error saving {description}: {e}")
        raise

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_json(path: Path, description: str = None) -> dict:
    """Load a JSON file from the specified path.

    Args:
        path (Path): Path to the JSON file.
        description (str, optional): Description used for logging purposes.

    Returns:
        dict: The contents of the JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"{description or 'JSON file'} loaded from {path}")
        return data
    except FileNotFoundError:
        logger.error(f"{description or 'JSON file'} not found at {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description or 'JSON file'} from {path}: {e}")
        raise


def save_pickle(obj: object, path: Path, description: str = None) -> None:
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"{description} saved to {path}")
    except Exception as e:
        logger.error(f"Error saving {description}: {e}")
        raise

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
        logger.info(f"Processed data saved to {path} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def load_pickle(path: Path, description: str = None) -> object:
    """
    Loads a pickle file from the given path.

    Args:
        path (Path): Path to the pickle file.
        description (str): Description of the object being loaded (used for logging).

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"{description} loaded from {path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading {description}: {e}")
        raise
