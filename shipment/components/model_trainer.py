import os
import sys
from typing import List, Tuple, Union, Any
import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline  

from shipment.logger import logging
from shipment.exception import ShippingException

from shipment.entity.config_entity import ModelTrainerConfig
from shipment.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from shipment.constant import MODEL_CONFIG_FILE
from shipment.utils.main_utils import MainUtils



class CostModel:
    """
    Represents a cost prediction model with preprocessing steps.

    Attributes:
        preprocessing_object (Pipeline): The preprocessing pipeline for feature transformation.
        trained_model_object (BaseEstimator): The trained model object for prediction.
    """
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: BaseEstimator):
        """
        Initializes the CostModel object.

        Args:
            preprocessing_object (Pipeline): The preprocessing pipeline (e.g., ColumnTransformer).
            trained_model_object (BaseEstimator): The trained prediction model (e.g., LinearRegression).
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> np.ndarray:  
        """
        Predicts the cost using the trained model and preprocesses the input.

        Args:
            X (pd.DataFrame): The input features DataFrame.

        Returns:
            np.ndarray: The predicted cost values.

        Raises:
            ShippingException: If an error occurs during prediction.
        """
        logging.info("Starting prediction")
        try:
            transformed_feature = self.preprocessing_object.transform(X)
            preds = self.trained_model_object.predict(transformed_feature)
            logging.info("Prediction completed")
            return preds
        except Exception as e:
            raise ShippingException(e, sys) from e

    def __repr__(self) -> str:
        """
        Provides a developer-friendly representation of the CostModel object.

        Returns:
            str: The representation string.
        """
        return f"CostModel(preprocessing_object={self.preprocessing_object}, trained_model_object={self.trained_model_object})"

    def __str__(self) -> str:
        """
        Provides a user-friendly representation of the CostModel object.

        Returns:
            str: The representation string.
        """
        return f"Cost Prediction Model using {type(self.trained_model_object).__name__} algorithm"



class ModelTrainer:
    """
    Class to train and evaluate machine learning models for shipment cost prediction.
    """
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        

    def get_trained_models(
        self, train_df: DataFrame, test_df: DataFrame  
    ) -> List[Tuple[float, BaseEstimator, str]]:
        """
        Trains and evaluates multiple models based on the configuration.

        Reads the model configurations from the YAML file, splits the transformed data into
        train and test sets, trains each model, and returns a list of tuples containing the
        model's performance score, the trained model object, and the model's name.

        Args:
            train_df (DataFrame): The training data DataFrame.
            test_df (DataFrame): The testing data DataFrame.

        Returns:
            List[Tuple[float, BaseEstimator, str]]: A list of tuples, each containing:
                - model score (float)
                - trained model object (BaseEstimator)
                - model name (str)

        Raises:
            ShippingException: If an error occurs during model training or evaluation.
        """
        logging.info("Started model training and evaluation")
        try:
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            models_list = list(model_config["train_model"].keys())

            x_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            x_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            tuned_model_list = [
                self.model_trainer_config.UTILS.get_tuned_model(model_name, x_train, y_train, x_test, y_test)
                for model_name in models_list
            ]

            logging.info("Finished model training and evaluation")
            return tuned_model_list

        except Exception as e:
            raise ShippingException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        """
        Initiates the model training process.

        Loads transformed data, trains and evaluates multiple models, 
        selects the best model, and saves the best model and artifacts.

        Returns:
            ModelTrainerArtifacts: Artifacts containing the path to the saved best model.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            logging.info("Created directory for model trainer artifacts")

            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_df = pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)

            list_of_trained_models: List[Tuple[float, BaseEstimator, str]] = self.get_trained_models(train_df, test_df)
            (best_model_score, best_model) = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(
                list_of_trained_models
            )
          
            logging.info(f"Best model found: {best_model.__class__.__name__}, Score: {best_model_score}")

            preprocessor_obj = self.model_trainer_config.UTILS.load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # Check if the best model outperforms the baseline model
            model_config = self.model_trainer_config.UTILS.read_yaml_file(MODEL_CONFIG_FILE)
            base_model_score = float(model_config["base_model_score"])
            if best_model_score >= base_model_score:
            
                # Update model score in YAML (optional, uncomment if you want to update)
                #self.model_trainer_config.utils.update_model_score(best_model_score)

                # Create CostModel object and save it
                cost_model = CostModel(preprocessor_obj, best_model)
                model_file_path = self.model_trainer_config.UTILS.save_object(
                    self.model_trainer_config.TRAINED_MODEL_FILE_PATH, cost_model
                )
            else:
                logging.info("No best model found with score more than base score")
                raise "No best model found with score more than base score "


            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_file_path=model_file_path)
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")

            return model_trainer_artifacts

        except Exception as e:
            raise ShippingException(e, sys) from e
