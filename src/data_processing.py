import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class  DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess(self, df):
        try:
            logger.info(f"DataProcessor: Starting preprocessing data {self.train_path} ...")

            logger.info(f"DataProcessor: Dropping unnecessary columns & rows ...")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            categorical_columns = self.config['data_processing']['categorical_columns']
            numerical_columns = self.config['data_processing']['numerical_columns']

            logger.info(f"DataProcessor: Applying label encoding ...")

            label_encoder = LabelEncoder()

            mappings = {}

            for col in categorical_columns:
                try:
                    df[col] = label_encoder.fit_transform(df[col])
                    mappings[col] = {label: code for label, code in
                                     zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
                except Exception as e:
                    logger.error(f"DataProcessor: Error while preprocessing categorical step {col}: {e}")
                    raise CustomException(f"DataProcessor: Error while preprocessing categorical step {col}: {e}")

            logger.info(f"DataProcessor: Handling skewness")

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[numerical_columns].apply(lambda x: x.skew())

            skewed_columns = skewness[skewness > skew_threshold].index

            for col in skewed_columns:
                try:
                    df[col] = np.log1p(df[col])
                except Exception as e:
                    logger.error(f"DataProcessor: Error while preprocessing numerical step {col}: {e}")
                    raise CustomException(f"DataProcessor: Error while preprocessing numerical step {col}: {e}")

            return df
        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            raise CustomException(e)

    def balance_data(self, df):
        try:
            logger.info(f"DataProcessor: Starting balancing data {self.train_path} ...")
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            smote = SMOTE(random_state=42)

            X_res, y_res = smote.fit_resample(X, y)
            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df['booking_status'] = y_res

            return balanced_df
        except Exception as e:

            logger.error(f"Error while balancing data step {e}")
            raise CustomException(e)

    def select_features(self, df):
        try:
            logger.info(f"DataProcessor: Starting feature selection ...")

            no_of_features = self.config['data_processing']['no_of_features']
            model = RandomForestClassifier(random_state=42)

            X = df.drop(columns='booking_status')
            y = df['booking_status']

            model.fit(X, y)

            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})

            top_features = feature_importance_df['feature'].head(no_of_features).values.tolist()

            top_features_df = df[top_features + ['booking_status']]

            return top_features_df
        except Exception as e:
            logger.error(f"Error while selecting features {e}")
            raise CustomException(f"Error while selecting features {e}")

    def save_data(self, df, file_path):
        try:
            logger.info(f"DataProcessor: Saving data ...")
            df.to_csv(file_path, index=False)
            logger.info(f"DataProcessor: Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error while saving data {e}")
            raise CustomException(f"Error while saving data {e}")

    def run(self):
        try:
            logger.info(f"DataProcessor: starting data processing ...")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess(train_df)
            test_df = self.preprocess(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info(f"DataProcessor: finished data processing")
        except Exception as e:
            logger.error(f"Error while DataProcessor run {e}")
            raise CustomException(e)

if __name__ == '__main__':
    data_processor = DataProcessor(train_path=TRAIN_FILE_PATH, test_path=TEST_FILE_PATH, processed_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
    data_processor.run()
