import os

import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from config import model_params
from config.model_params import LIGHTGBM_PARAMS, RANDOM_SEARCH_PARAMS
from config.paths_config import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH, LGMB_MODEL_PATH, MODEL_OUTPUT_DIR
from src.custom_exception import CustomException
from src.logger import get_logger
from utils.common_functions import load_data
import lightgbm as lgb
import mlflow

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_dir, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.model_params = model_params
        self.model_output_dir = model_output_dir

    def load_and_split_data(self):
        try:
            logger.info(f'ModelTraining: loading data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'ModelTraining: loading data from {self.test_path}')
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info(f'ModelTraining: split data successfully')

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f'ModelTraining: error during loading and splitting data: {e}')
            raise CustomException(f'ModelTraining: error during loading and splitting data: {e}')

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info(f'ModelTraining: generating model')

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring'],
            )

            logger.info(f'ModelTraining: hyperparameters tuning started')

            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_

            logger.info(f'ModelTraining: hyperparameters tuning finished')
            logger.info(f'ModelTraining: best params: {best_params}')

            return best_model
        except Exception as e:
            logger.error(f'ModelTraining: error during training LightGMB: {e}')
            raise CustomException(f'ModelTraining: error during training LightGMB: {e}')

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info(f'ModelTraining: starting evaluation')

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"ModelTraining: accuracy score is {accuracy}")
            logger.info(f"ModelTraining: recall score is {recall}")
            logger.info(f"ModelTraining: precision score is {precision}")
            logger.info(f"ModelTraining: f1 score is {f1}")

            logger.info(f'ModelTraining: finished evaluation')

            return {
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        except Exception as e:
            logger.error(f'ModelTraining: error during evaluating model: {e}')
            raise CustomException(f'ModelTraining: error during evaluating model: {e}')

    def save_model(self, model):
        try:
            logger.info(f'ModelTraining: saving model')

            os.makedirs(self.model_output_dir, exist_ok=True)

            joblib.dump(model, self.model_output_path)
            logger.info(f'ModelTraining: model saved successfully')

        except Exception as e:
            logger.error(f'ModelTraining: error during saving model: {e}')
            raise CustomException(f'ModelTraining: error during saving model: {e}')

    def run(self):
        try:
            with mlflow.start_run():
                logger.info(f'ModelTraining: starting model training')

                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)

                logger.info(f'ModelTraining: logging model')
                mlflow.log_artifact(self.model_output_path, artifact_path='model')
                logger.info(f'ModelTraining: logging params')
                mlflow.log_params(best_lgbm_model.get_params())
                logger.info(f'ModelTraining: logging metrics')
                mlflow.log_metrics(metrics)

                self.save_model(best_lgbm_model)
                logger.info(f'ModelTraining: model training finished successfully')
        except Exception as e:
            logger.error(f'ModelTraining: error during model training: {e}')
            raise CustomException(f'ModelTraining: error model training: {e}')

if __name__ == '__main__':
    data_ingestion = ModelTraining(test_path=PROCESSED_TEST_DATA_PATH, train_path=PROCESSED_TRAIN_DATA_PATH, model_output_dir=MODEL_OUTPUT_DIR, model_output_path=LGMB_MODEL_PATH)
    data_ingestion.run()
