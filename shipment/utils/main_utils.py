import shutil
import sys
from typing import Dict, Tuple, List

import dill
import xgboost
import numpy as np
import yaml
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators
from yaml import safe_dump

from shipment.constant import *
from shipment.exception import ShippingException
from shipment.logger import logger


class MainUtils:
    """
    Utility class for common machine learning and file handling tasks.
    """

    def read_yaml_file(self, filename: str) -> dict:
        """
        Reads a YAML file and returns its contents as a dictionary.

        Args:
            filename (str): Path to the YAML file.

        Returns:
            dict: The loaded YAML data as a dictionary.

        Raises:
            ShippingException: If there is an error reading the file.
        """
        logger.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise ShippingException(e, sys) from e

    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> None:
        """
        Converts a JSON object to YAML and writes it to a file.

        Args:
            json_file (dict): The JSON object to convert.
            yaml_file_path (str): The path to the YAML file where the data will be written.

        Raises:
            ShippingException: If there is an error writing the file.
        """
        logger.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            with open(yaml_file_path, "w") as stream:
                yaml.dump(json_file, stream)
        except Exception as e:
            raise ShippingException(e, sys) from e

    def save_numpy_array_data(self, file_path: str, array: np.ndarray) -> str:
        """
        Saves a NumPy array to a file.

        Args:
            file_path (str): The path to the file where the array will be saved.
            array (np.ndarray): The NumPy array to save.

        Returns:
            str: The path to the saved file.

        Raises:
            ShippingException: If there is an error saving the array.
        """
        logger.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            return file_path
        except Exception as e:
            raise ShippingException(e, sys) from e

    def load_numpy_array_data(self, file_path: str) -> np.ndarray:
        """
        Loads a NumPy array from a file.

        Args:
            file_path (str): The path to the file containing the array.

        Returns:
            np.ndarray: The loaded NumPy array.

        Raises:
            ShippingException: If there is an error loading the array.
        """
        logger.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)
        except Exception as e:
            raise ShippingException(e, sys) from e

    def get_tuned_model(
        self,
        model_name: str,
        train_x: DataFrame,
        train_y: DataFrame,
        test_x: DataFrame,
        test_y: DataFrame,
    ) -> Tuple[float, BaseEstimator, str]:  
        """
        Gets a tuned model by finding the best parameters using GridSearchCV.

        Args:
            model_name (str): The name of the model to use.
            train_x (DataFrame): The training data features.
            train_y (DataFrame): The training data target.
            test_x (DataFrame): The testing data features.
            test_y (DataFrame): The testing data target.

        Returns:
            Tuple[float, BaseEstimator, str]: A tuple containing:
                - The best model score (R2 score)
                - The tuned model object
                - The name of the model class

        Raises:
            ShippingException: If there is an error during model tuning or evaluation.
        """
        logger.info("Entered the get_tuned_model method of MainUtils class")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model, train_x, train_y)
            model.set_params(**model_best_params)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            model_score = self.get_model_score(test_y, preds)
            return model_score, model, model.__class__.__name__
        except Exception as e:
            raise ShippingException(e, sys) from e



    @staticmethod
    def get_model_score(test_y: DataFrame, preds: DataFrame) -> float:
        """
        Calculates the R2 score for the given predictions.

        Args:
            test_y (DataFrame): The true target values.
            preds (DataFrame): The predicted target values.

        Returns:
            float: The R2 score.

        Raises:
            ShippingException: If there is an error calculating the score.
        """
        logger.info("Entered the get_model_score method of MainUtils class")
        try:
            model_score = r2_score(test_y, preds)
            logger.info("Model score is {}".format(model_score))
            return model_score
        except Exception as e:
            raise ShippingException(e, sys) from e

    @staticmethod
    def get_base_model(model_name: str) -> BaseEstimator: 
        """
        Gets a base model object based on the model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            BaseEstimator: An instance of the specified base model class.

        Raises:
            ShippingException: If the model name is invalid or not found.
        """
        logger.info("Entered the get_base_model method of MainUtils class")
        try:
            if model_name.lower().startswith("xgb") is True:
                model = xgboost.__dict__[model_name]()
            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)
                model = all_estimators().__getitem__(model_idx)[1]()
            return model
        except Exception as e:
            raise ShippingException(e, sys) from e




    def get_model_params(
        self, model: BaseEstimator, x_train: DataFrame, y_train: DataFrame
    ) -> Dict:
        """
        Finds the best hyperparameters for a model using GridSearchCV.

        Args:
            model (BaseEstimator): The model to tune.
            x_train (DataFrame): The training data features.
            y_train (DataFrame): The training data target.

        Returns:
            Dict: A dictionary of the best hyperparameters found.

        Raises:
            ShippingException: If there is an error during the GridSearchCV process.
        """
        logger.info("Entered the get_model_params method of MainUtils class")
        try:
            VERBOSE = 3
            CV = 2
            N_JOBS = -1

            model_name = model.__class__.__name__
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            model_param_grid = model_config["train_model"][model_name]
            model_grid = GridSearchCV(
                model, model_param_grid, verbose=VERBOSE, cv=CV, n_jobs=N_JOBS
            )
            model_grid.fit(x_train, y_train)
            return model_grid.best_params_
        except Exception as e:
            raise ShippingException(e, sys) from e



    @staticmethod
    def save_object(file_path: str, obj: object) -> str: 
        """
        Saves a Python object to a file using dill.

        Args:
            file_path (str): The path to the file where the object will be saved.
            obj (object): The Python object to save.

        Returns:
            str: The path to the saved file.

        Raises:
            ShippingException: If there is an error saving the object.
        """
        logger.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
            return file_path  
        except Exception as e:
            raise ShippingException(e, sys) from e

    @staticmethod
    def get_best_model_with_name_and_score(
        model_list: List[Tuple[float, BaseEstimator, str]] 
    ) -> Tuple[BaseEstimator, float]:
        """
        Finds the model with the highest score from a list of models with scores and names.

        Args:
            model_list (List[Tuple[float, BaseEstimator, str]]): A list of tuples where each tuple contains:
                - model score (float)
                - model object (BaseEstimator)
                - model name (str)

        Returns:
            Tuple[BaseEstimator, float]: A tuple containing the best model and its score.

        Raises:
            ShippingException: If the model list is empty or an error occurs.
        """
        logger.info(
            "Entered the get_best_model_with_name_and_score method of MainUtils class"
        )
        try:
            if not model_list:
                raise ValueError("Model list is empty.")
            return max(model_list, key=lambda x: x[0])[:2]  # Get only the best model and score
        except Exception as e:
            raise ShippingException(e, sys) from e

    @staticmethod
    def load_object(file_path: str) -> object:
        """
        Loads a Python object from a file using dill.

        Args:
            file_path (str): The path to the file containing the object.

        Returns:
            object: The loaded Python object.

        Raises:
            ShippingException: If there is an error loading the object.
        """
        logger.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj =  dill.load(file_obj)
            logger.info("Exited the load_object method of MainUtils class")
            return obj
        except Exception as e:
            raise ShippingException(e, sys) from e

    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        """
        Creates a zip archive of a folder.

        Args:
            file_name (str): The name of the zip file to create (without the .zip extension).
            folder_name (str): The name of the folder to archive.

        Raises:
            ShippingException: If there is an error creating the zip archive.
        """
        logger.info("Entered the create_artifacts_zip method of MainUtils class")
        try:
            shutil.make_archive(file_name, "zip", folder_name)
            logger.info("Exited the create_artifacts_zip method of MainUtils class")
        except Exception as e:
            raise ShippingException(e, sys) from e

    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        """
        Unzips a zip archive into a folder.

        Args:
            filename (str): The name of the zip file to unzip.
            folder_name (str): The name of the folder where the contents will be extracted.

        Raises:
            ShippingException: If there is an error unzipping the file.
        """
        logger.info("Entered the unzip_file method of MainUtils class")
        try:
            shutil.unpack_archive(filename, folder_name)
            logger.info("Exited the unzip_file method of MainUtils class")
        except Exception as e:
            raise ShippingException(e, sys) from e

    def update_model_score(self, best_model_score: float) -> None:
        """
        Updates the best model score in the model configuration file.

        Args:
            best_model_score (float): The new best model score.

        Raises:
            ShippingException: If there is an error reading or writing the configuration file.
        """
        logger.info("Entered the update_model_score method of MainUtils class")
        try:
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            model_config["base_model_score"] = str(best_model_score)  # Convert to string for YAML
            with open(MODEL_CONFIG_FILE, "w+") as fp:
                safe_dump(model_config, fp, sort_keys=False)
            logger.info("Exited the update_model_score method of MainUtils class")
        except Exception as e:
            raise ShippingException(e, sys) from e

