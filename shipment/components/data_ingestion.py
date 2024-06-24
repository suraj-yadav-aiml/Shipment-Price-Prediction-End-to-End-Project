import sys
import os

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from shipment.logger import logger
from shipment.exception import ShippingException

from shipment.entity.config_entity import DataIngestionConfig
from shipment.entity.artifacts_entity import DataIngestionArtifacts

from shipment.configuration.mongodb_operations import MongoDBOperation
from shipment.constant import TEST_SIZE


class DataIngestion:
    """
    Class for managing the data ingestion process from MongoDB.
    """

    def __init__(
        self, 
        data_ingestion_config: DataIngestionConfig, 
        mongo_op: MongoDBOperation
    ):
        """
        Initializes a DataIngestion object with the given configuration and MongoDB operations.

        Args:
            data_ingestion_config (DataIngestionConfig): The configuration for data ingestion.
            mongo_op (MongoDBOperation): An object to handle MongoDB operations.
        """
        self.data_ingestion_config = data_ingestion_config
        self.mongo_op = mongo_op

    def get_data_from_mongodb(self) -> DataFrame:
        """
        Fetches data from MongoDB as a Pandas DataFrame.

        Returns:
            DataFrame: The fetched data.

        Raises:
            ShippingException: If there is an error during data fetching.
        """
        logger.info("Fetching data from MongoDB")
        try:
            df = self.mongo_op.get_collection_as_dataframe(
                db_name=self.data_ingestion_config.DB_NAME,
                collection_name=self.data_ingestion_config.COLLECTION_NAME,
            )
            logger.info(f"Fetched {len(df)} rows from MongoDB")
            return df
        except Exception as e:
            raise ShippingException(e, sys) from e

    def split_data_as_train_test(
        self, df: DataFrame
    ) -> DataIngestionArtifacts:
        """
        Splits the input DataFrame into train and test sets and saves them as CSV files.

        Args:
            df (DataFrame): The DataFrame to split.

        Returns:
            DataIngestionArtifacts: An object containing paths to the saved train and test files.

        Raises:
            ShippingException: If there is an error during splitting or saving.
        """
        logger.info("Splitting data into train and test sets")
        try:
            os.makedirs(
                self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR, exist_ok=True
            )
            os.makedirs(
                self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR, exist_ok=True
            )
            logger.info("Directories for data artifacts created.")

            train_set, test_set = train_test_split(df, test_size=TEST_SIZE)

            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_FILE_PATH, index=False)
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_FILE_PATH, index=False)

            logger.info("Train and test data saved to CSV files.")
            data_ingestion_artifacts = DataIngestionArtifacts(
                train_data_file_path=self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                test_data_file_path=self.data_ingestion_config.TEST_DATA_FILE_PATH,
            )
            return data_ingestion_artifacts

        except Exception as e:
            raise ShippingException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Initiates the entire data ingestion process.

        Fetches data from MongoDB, drops specified columns, handles missing values, 
        splits the data into train and test sets, and saves the results.

        Returns:
            DataIngestionArtifacts: An object containing paths to the saved train and test files.

        Raises:
            ShippingException: If there is an error during the data ingestion process.
        """
        logger.info("Initiating data ingestion")
        try:
            df = self.get_data_from_mongodb()
            df = df.drop(self.data_ingestion_config.DROP_COLS, axis=1)
            df = df.dropna()
            
            logger.info("Data preprocessing completed.")
            return self.split_data_as_train_test(df)
        except Exception as e:
            raise ShippingException(e, sys) from e
