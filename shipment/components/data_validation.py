import os
import sys
import json
from typing import List, Dict, Union, Tuple, Any

import pandas as pd
from pandas import DataFrame

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from shipment.logger import logger
from shipment.exception import ShippingException

from shipment.entity.config_entity import DataValidationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
)


class DataValidation:
    """
    Class for validating data against a predefined schema.
    """
    
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initializes the DataValidation object.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts from the data ingestion stage.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_validation_config = data_validation_config

    def validate_schema_columns(self, df: DataFrame) -> bool:
        """
        Validates the columns of a DataFrame against a predefined schema.

        Args:
            df (DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the schema validation passes, False otherwise.

        Raises:
            ShippingException: If an error occurs during validation.
        """
        logger.info("Validating dataframe columns against the schema.")
        try:
            schema_columns: List[Dict[str, Union[str, float, int]]] = self.data_validation_config.SCHEMA_CONFIG["columns"]  #[{'Customer Id': 'object'},
                                                                                                                            #  {'Artist Name': 'object'},
                                                                                                                            #  {'Artist Reputation': 'float64'},
                                                                                                                            #  {'Height': 'float64'},
                                                                                                                            #  {'Width': 'float64'},
                                                                                                                            #  {'Weight': 'float64'}, ... ]
            
            # Compare the number of columns, ensuring all columns exist
            if len(df.columns) != len(schema_columns):
                validation_status = False
                logger.info("Dataframe columns do not match the schema.")
            else:
                validation_status = True
                logger.info("Dataframe columns validated successfully.")

            return validation_status

        except Exception as e:
            raise ShippingException(e, sys) from e
    

    def is_numerical_column_exists(self, df: DataFrame) -> bool:
        """
        Validates if all the specified numerical columns exist in the DataFrame.

        Args:
            df (DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all numerical columns exist, False otherwise.

        Raises:
            ShippingException: If an error occurs during validation.
        """
        logger.info("Checking if all numerical columns exist in the dataframe")
        try:
            numerical_columns: List[str] = self.data_validation_config.SCHEMA_CONFIG["numerical_columns"] 
            
            missing_columns = set(numerical_columns) - set(df.columns)  
            if missing_columns:
                for column in missing_columns:
                    logger.info(f"Numerical column - {column} not found in dataframe")
                return False  
            else:
                logger.info("All numerical columns are present in the dataframe")
                return True

        except Exception as e:
            raise ShippingException(e, sys) from e
    

    def is_categorical_column_exists(self, df: DataFrame) -> bool:
        """
        Validates if all the specified categorical columns exist in the DataFrame.

        Args:
            df (DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all categorical columns exist, False otherwise.

        Raises:
            ShippingException: If an error occurs during validation.
        """
        logger.info("Checking if all categorical columns exist in the dataframe")
        try:
            categorical_columns: List[str] = self.data_validation_config.SCHEMA_CONFIG["categorical_columns"]

            missing_columns = set(categorical_columns) - set(df.columns)  
            if missing_columns:
                for column in missing_columns:
                    logger.info(f"Categorical column - {column} not found in dataframe")
                return False 
            else:
                logger.info("All categorical columns are present in the dataframe")
                return True

        except Exception as e:
            raise ShippingException(e, sys) from e
    

    def validate_dataset_schema_columns(self) -> Tuple[bool, bool]:
        """
        Validates the columns of both train and test DataFrames against the schema.

        Returns:
            Tuple[bool, bool]: A tuple where the first element indicates if the train DataFrame
                schema is valid, and the second element indicates if the test DataFrame schema is valid.
        """
        logger.info("Validating train and test dataset schema columns")
        try:
            train_schema_status = self.validate_schema_columns(self.train_set)
            test_schema_status = self.validate_schema_columns(self.test_set)
            return train_schema_status, test_schema_status
        except Exception as e:
            raise ShippingException(e, sys) from e

    def validate_is_numerical_column_exists(self) -> Tuple[bool, bool]:
        """
        Validates if all numerical columns exist in both train and test DataFrames.

        Returns:
            Tuple[bool, bool]: A tuple where the first element indicates if all numerical
                columns exist in the train DataFrame, and the second element indicates if 
                all numerical columns exist in the test DataFrame.
        """
        logger.info(
            "Validating if numerical columns exist in train and test datasets"
        )
        try:
            train_num_cols_exist = self.is_numerical_column_exists(self.train_set)
            test_num_cols_exist = self.is_numerical_column_exists(self.test_set)
            return train_num_cols_exist, test_num_cols_exist
        except Exception as e:
            raise ShippingException(e, sys) from e

    def validate_is_categorical_column_exists(self) -> Tuple[bool, bool]:
        """
        Validates if all categorical columns exist in both train and test DataFrames.

        Returns:
            Tuple[bool, bool]: A tuple where the first element indicates if all categorical
                columns exist in the train DataFrame, and the second element indicates if 
                all categorical columns exist in the test DataFrame.
        """
        logger.info(
            "Validating if categorical columns exist in train and test datasets"
        )
        try:
            train_cat_cols_exist = self.is_categorical_column_exists(self.train_set)
            test_cat_cols_exist = self.is_categorical_column_exists(self.test_set)
            return train_cat_cols_exist, test_cat_cols_exist
        except Exception as e:
            raise ShippingException(e, sys) from e


    def detect_dataset_drift(
        self, reference: DataFrame, production: DataFrame, get_ratio: bool = False
    ) -> Union[bool, float]:
        """
        Detects data drift between a reference dataset and a production dataset.

        Args:
            reference (DataFrame): The reference dataset (e.g., training data).
            production (DataFrame): The production dataset (e.g., new incoming data).
            get_ratio (bool, optional): If True, returns the drift ratio. Otherwise, returns the drift status (True/False). Defaults to False.

        Returns:
            Union[bool, float]: 
                - If `get_ratio` is True, returns the drift ratio (float).
                - If `get_ratio` is False, returns True if drift is detected, False otherwise.

        Raises:
            ShippingException: If an error occurs during drift detection.
        """
        logger.info("Detecting dataset drift...")

        try:
            drift_profile = Profile(sections=[DataDriftProfileSection()])
            drift_profile.calculate(reference, production)

            report = json.loads(drift_profile.json())

            data_drift_file_path = self.data_validation_config.DATA_DRIFT_FILE_PATH
            self.data_validation_config.UTILS.write_json_to_yaml_file(report, data_drift_file_path)

            n_features = report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            if get_ratio:
                return n_drifted_features / n_features 
            else:
                # Return True if the dataset has drifted
                return report["data_drift"]["data"]["metrics"]["dataset_drift"]

        except Exception as e:
            raise ShippingException(e, sys) from e
    

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        Initiates the data validation process.

        Performs schema validation, checks for the existence of numerical and categorical columns,
        detects data drift, and returns the validation results as a DataValidationArtifacts object.

        Returns:
            DataValidationArtifacts: Contains the drift report file path and the validation status.
        """
        logger.info("Starting data validation")
        try:

            self.train_set = pd.read_csv(
                self.data_ingestion_artifacts.train_data_file_path
            )
            self.test_set = pd.read_csv(
                self.data_ingestion_artifacts.test_data_file_path
            )

            os.makedirs(
                self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR,
                exist_ok=True,
            )
            logger.info("Created directory for data validation artifacts")

            train_schema_status, test_schema_status = self.validate_dataset_schema_columns()
            logger.info(
                f"Train schema status: {train_schema_status}, Test schema status: {test_schema_status}"
            )

            train_num_cols_exist, test_num_cols_exist = self.validate_is_numerical_column_exists()
            logger.info(
                f"Train numerical columns status: {train_num_cols_exist}, Test numerical columns status: {test_num_cols_exist}"
            )

            train_cat_cols_exist, test_cat_cols_exist = self.validate_is_categorical_column_exists()
            logger.info(
                f"Train categorical columns status: {train_cat_cols_exist}, Test categorical columns status: {test_cat_cols_exist}"
            )

            drift_detected = self.detect_dataset_drift(self.train_set, self.test_set)
            logger.info(f"Data drift detected: {drift_detected}")
            
            validation_status = (
                train_schema_status
                and test_schema_status
                and train_num_cols_exist
                and test_num_cols_exist
                and train_cat_cols_exist
                and test_cat_cols_exist
                and not drift_detected  # Check if drift is NOT detected
            )

            data_validation_artifacts = DataValidationArtifacts(
                data_drift_file_path=self.data_validation_config.DATA_DRIFT_FILE_PATH,
                validation_status=validation_status,
            )

            logger.info("Data validation completed")
            return data_validation_artifacts

        except Exception as e:
            raise ShippingException(e, sys) from e
