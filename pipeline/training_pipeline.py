from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    data_processor = DataProcessor(train_path=TRAIN_FILE_PATH, test_path=TEST_FILE_PATH, processed_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
    data_processor.run()

    model_training = ModelTraining(test_path=PROCESSED_TEST_DATA_PATH, train_path=PROCESSED_TRAIN_DATA_PATH, model_output_dir=MODEL_OUTPUT_DIR, model_output_path=LGMB_MODEL_PATH)
    model_training.run()