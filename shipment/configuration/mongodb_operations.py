import sys
import pandas as pd
from typing import Collection, List, Dict, Any
from pandas import DataFrame

from pymongo.database import Database
from pymongo import MongoClient

from shipment.constant import DB_URL
from shipment.exception import ShippingException
from shipment.logger import logger


class MongoDBOperation:
    """
    Class for handling MongoDB database operations.
    """

    def __init__(self):
        """
        Initializes the MongoDB client using the provided connection string.
        """
        self.DB_URL = DB_URL  
        self.client = MongoClient(self.DB_URL)  

    def get_database(self, db_name: str) -> Database:
        """
        Gets a MongoDB database object.

        Args:
            db_name (str): Name of the database.

        Returns:
            Database: The database object.

        Raises:
            ShippingException: If there's an error accessing the database.
        """
        logger.info(f"Entered get_database method: db_name={db_name}")
        try:
            db = self.client[db_name]  
            logger.info(f"Connected to database: {db_name}")
            return db
        except Exception as e:
            raise ShippingException(e, sys) from e

    def get_collection(self, database: Database, collection_name: str) -> Collection:
        """
        Gets a collection from a MongoDB database.

        Args:
            database (Database): The database object.
            collection_name (str): Name of the collection.

        Returns:
            Collection: The collection object.

        Raises:
            ShippingException: If there's an error accessing the collection.
        """
        logger.info(f"Entered get_collection method: collection_name={collection_name}")
        try:
            collection = database[collection_name] 
            logger.info(f"Connected to collection: {collection_name}")
            return collection
        except Exception as e:
            raise ShippingException(e, sys) from e

    def get_collection_as_dataframe(
        self, db_name: str, collection_name: str
    ) -> DataFrame:
        """
        Retrieves a collection from MongoDB and converts it into a Pandas DataFrame.

        Args:
            db_name (str): Name of the database.
            collection_name (str): Name of the collection.

        Returns:
            DataFrame: A Pandas DataFrame containing the collection data.

        Raises:
            ShippingException: If there's an error fetching data or converting to DataFrame.
        """
        logger.info(f"Entered get_collection_as_dataframe method: db_name={db_name}, collection_name={collection_name}")
        try:
            database = self.get_database(db_name)
            collection = self.get_collection(database, collection_name)

            df = pd.DataFrame(list(collection.find()))  # Convert cursor to list of dicts, then to DataFrame
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)  # Remove MongoDB's internal ID if present
            logger.info(f"Converted collection {collection_name} to DataFrame")
            return df
        except Exception as e:
            raise ShippingException(e, sys) from e

    def insert_dataframe_as_record(
        self, data_frame: DataFrame, db_name: str, collection_name: str
    ) -> None:
        """
        Inserts a Pandas DataFrame as records into a MongoDB collection.

        Args:
            data_frame (DataFrame): The DataFrame to insert.
            db_name (str): Name of the database.
            collection_name (str): Name of the collection.

        Raises:
            ShippingException: If there's an error inserting records.
        """
        logger.info(f"Entered insert_dataframe_as_record method: db_name={db_name}, collection_name={collection_name}")
        try:
            records: List[Dict[str, Any]] = data_frame.to_dict(orient='records')
            logger.info(f"Converted DataFrame to JSON records")
            database = self.get_database(db_name)
            collection = self.get_collection(database, collection_name)
            collection.insert_many(records)  # Insert multiple records
            logger.info(f"Inserted {len(records)} records into {collection_name} collection")
        except Exception as e:
            raise ShippingException(e, sys) from e

