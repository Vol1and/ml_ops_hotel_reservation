import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Succesfully read {file_path}")
            return config

    except Exception as e:
        logger.error(f"Error while reading {file_path}")
        raise CustomException(f"Error while reading {file_path}", e)


def load_data(file_path):
    try:
        logger.info(f"Loading data from file {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"Error while loading data from {file_path}")
        raise CustomException(f"Error while reading {file_path}", e)