import os
import sys

import pandas as pd
import numpy as np
from pandas import DataFrame
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline  

from shipment.logger import logger
from shipment.exception import ShippingException
from shipment.entity.config_entity import DataTransformationConfig
from shipment.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts, DataValidationArtifacts



class DataTransformation:
    """
    Class to handle data transformation operations for the shipment cost prediction model.
    """

    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifacts
    ):
        """
        Initializes the DataTransformation object with data artifacts and configuration.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): The artifacts from the data ingestion stage.
            data_transformation_config (DataTransformationConfig): The configuration for data transformation.
        """
        logger.info("Initializing DataTransformation class")

        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
            self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)

            self.train_set = self.train_set.drop(columns=self.data_transformation_config.DROP_COLS)
            self.test_set = self.test_set.drop(columns=self.data_transformation_config.DROP_COLS)

            logger.info("DataTransformation initialized successfully")
        except Exception as e:
            raise ShippingException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline: 
        """
        Creates and returns a data transformer pipeline object for preprocessing data.

        Returns:
            Pipeline: A scikit-learn Pipeline object containing transformers for data preprocessing.

        Raises:
            ShippingException: If there is an error during the creation of the pipeline.
        """
        logger.info("Entered get_data_transformer_object method")

        try:
            schema_config = self.data_transformation_config.SCHEMA_CONFIG

            numerical_columns = schema_config["numerical_columns"]
            onehot_columns = schema_config["onehot_columns"]
            binary_columns = schema_config["binary_columns"]

            logger.info(f"Found these columns in the schema:\n numerical_columns: {numerical_columns}\n onehot_columns: {onehot_columns}\n binary_columns: {binary_columns}")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            binary_transformer = BinaryEncoder()
            logger.info(f"Initialized transformers: \n numeric_transformer: {numeric_transformer}\n oh_transformer: {oh_transformer}\n binary_transformer: {binary_transformer}")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, onehot_columns),
                    ("BinaryEncoder", binary_transformer, binary_columns),
                    ("StandardScaler", numeric_transformer, numerical_columns),
                ]
            )

            logger.info(f"Created preprocessor: \n {preprocessor}")

            return Pipeline(steps=[('preprocessor', preprocessor)])

        except Exception as e:
            raise ShippingException(e, sys) from e


    @staticmethod
    def _outlier_capping(col: str, df: DataFrame) -> DataFrame:
        """
        Performs outlier capping for a given column in a DataFrame.

        This method identifies outliers using the Interquartile Range (IQR) method and 
        replaces them with the upper or lower limit calculated from the IQR.

        Args:
            col (str): The name of the column to cap outliers for.
            df (DataFrame): The DataFrame containing the data.

        Returns:
            DataFrame: The DataFrame with capped outliers.

        Raises:
            ShippingException: If an error occurs during outlier capping.
        """
        logger.info(f"Performing outlier capping for column: {col}")
        try:
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col] < lower_limit), col] = lower_limit

            logger.info(f"Outlier capping completed for column: {col}")
            return df

        except Exception as e:
            raise ShippingException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Initiates the data transformation process.

        Preprocesses the train and test datasets, applies outlier capping, splits into features
        and target, transforms the features using the defined preprocessor, and saves the 
        transformed data and preprocessor object as artifacts.

        Returns:
            DataTransformationArtifacts: An object holding paths to the transformed data and preprocessor.

        Raises:
            ShippingException: If an error occurs during the transformation process.
        """
        logger.info("Starting data transformation")
        try:
            if self.data_validation_artifact.validation_status:
                os.makedirs(self.data_transformation_config.TRANSFORMED_TRAIN_DATA_DIR, exist_ok=True)
                os.makedirs(self.data_transformation_config.TRANSFORMED_TEST_DATA_DIR, exist_ok=True)

                preprocessor = self.get_data_transformer_object()

                schema_config = self.data_transformation_config.SCHEMA_CONFIG
                target_column_name = schema_config["target_column"]
                numerical_columns = schema_config["numerical_columns"]

                # Outlier capping
                continuous_columns = [
                    col for col in numerical_columns if len(self.train_set[col].unique()) >= 25
                ]
                for col in continuous_columns:
                    self.train_set = self._outlier_capping(col, self.train_set)
                    self.test_set = self._outlier_capping(col, self.test_set)


                input_feature_train_df = self.train_set.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = self.train_set[target_column_name]

                input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = self.test_set[target_column_name]

                logger.info("Applying preprocessing object on training and testing datasets.")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # Combine features and target
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                transformed_train_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                    self.data_transformation_config.TRANSFORMED_TRAIN_FILE_PATH, train_arr
                )
                transformed_test_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                    self.data_transformation_config.TRANSFORMED_TEST_FILE_PATH, test_arr
                )

                preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                    self.data_transformation_config.PREPROCESSOR_FILE_PATH, preprocessor
                )

                data_transformation_artifacts = DataTransformationArtifacts(
                    transformed_object_file_path=preprocessor_obj_file,
                    transformed_train_file_path=transformed_train_file,
                    transformed_test_file_path=transformed_test_file,
                )

                logger.info("Data transformation completed")
                return data_transformation_artifacts
            
            else:
                logger.info("Data Validation Stage failed !!!")
                raise Exception("Data Validation Stage failed !!!")

        except Exception as e:
            raise ShippingException(e, sys) from e

