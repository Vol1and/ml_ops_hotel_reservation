import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.bucket_filename = self.config['bucket_filename']
        self.train_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data ingestion started on bucket {self.bucket_name} and filename {self.bucket_filename}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.get_blob(self.bucket_filename)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV file downloaded to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error while downloading CSV file: {e}")
            raise CustomException("Error while downloading CSV file", e)

    def split_data(self):
        try:
            logger.info(f"Starting the splitting process")
            data = pd.read_csv(RAW_FILE_PATH)

            train, test = train_test_split(data, train_size=self.train_ratio)

            train.to_csv(TRAIN_FILE_PATH)
            test.to_csv(TEST_FILE_PATH)

            logger.info(f"Train and test data splitted successfully to: train - {TRAIN_FILE_PATH}; test - {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error while downloading CSV file: {e}")
            raise CustomException("Error while downloading CSV file", e)

    def run(self):
        try:
            logger.info(f"Starting the data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()

            logger.info(f"Data ingestion process completed successfully")
        except Exception as e:
            logger.error(f"CustomException: {str(e)}")

        finally:
            logger.info(f"Finished data ingestion process")


if __name__ == '__main__':
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
