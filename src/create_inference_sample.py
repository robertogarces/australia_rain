import pandas as pd
import logging
import sys
import os
sys.path.append('../')

from config.paths import RAW_DATA_PATH, CONFIG_PATH
from utils.file_management import load_config

# Setup logging con formato profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Starting inference sample creation ===")

    try:
        dataset_path = RAW_DATA_PATH / "weatherAUS.csv"
        logger.info(f"Loading raw dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"Raw dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    try:
        config_path = CONFIG_PATH / "settings.yaml"
        config = load_config(config_path)
        logger.info("Configuration file loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

    # Extraer parámetros
    try:
        sample_cfg = config['inference_sample']
        sample_size = sample_cfg['sample_size']
        random_state = sample_cfg['random_state']
        logger.info(f"Sampling {sample_size} rows with random_state={random_state}")
    except KeyError as e:
        logger.error(f"Missing config key: {e}")
        raise

    # Tomar muestra
    try:
        # Tomar muestra para inferencia
        inference_df = df.sample(n=sample_size, random_state=random_state)
        inference_path = RAW_DATA_PATH / "inference_sample.csv"
        inference_df.to_csv(inference_path, index=False)
        logger.info(f"Inference sample saved to {inference_path} with shape {inference_df.shape}")

        # Guardar el resto como dataset de entrenamiento
        training_df = df.drop(inference_df.index)
        training_path = RAW_DATA_PATH / "training_data.csv"
        training_df.to_csv(training_path, index=False)
        logger.info(f"Training sample saved to {training_path} with shape {training_df.shape}")

    except Exception as e:
        logger.error(f"Error creating or saving sample: {e}")
        raise

    logger.info("=== Inference sample creation completed successfully ===")

if __name__ == "__main__":
    main()

