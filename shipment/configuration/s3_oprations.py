import sys
import os
import pickle
import pandas as pd
from io import BytesIO, StringIO 
from typing import Union, List, Optional, Any

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from pandas import DataFrame

from shipment.logger import logging
from shipment.exception import ShippingException


class S3Operation:
    """
    Class for handling Amazon S3 operations (reading/writing objects, getting buckets, etc.).
    """

    def __init__(self):
        """
        Initializes the S3 client and resource objects.
        """
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    @staticmethod
    def read_object(
        object_name: str, bucket_name: str, decode: bool = True, make_readable: bool = False
    ) -> Union[StringIO, BytesIO, bytes]:
        """
        Reads an object from S3.

        This method is used to fetch the content of a specific object (e.g., a file) from an Amazon S3 bucket. 

        Decoded as Text: By default (decode=True), it assumes the object's content is text-based (e.g., a CSV file) and 
        decodes it into a string.

        Raw Bytes: If decode=False, it returns the raw byte data without attempting to decode it 
        (useful for binary files like images or serialized objects).

        Readable Formats: With make_readable=True, it returns the content wrapped in either a StringIO object (for text) or 
        a BytesIO object (for binary data). This makes the content easy to work with as if it were a file-like object

        Args:
            object_name (str): The name (key) of the object in the S3 bucket.
            bucket_name (str): The name of the S3 bucket.
            decode (bool, optional): Whether to decode the object's content as a string. Defaults to True.
            make_readable (bool, optional): Whether to return the content as a StringIO (for text) 
                                             or BytesIO (for binary) object. Defaults to False.

        Returns:
            Union[StringIO, BytesIO, bytes]: 
                - If `decode` is True and `make_readable` is True, returns a StringIO object.
                - If `decode` is True and `make_readable` is False, returns a string.
                - If `decode` is False and `make_readable` is True, returns a BytesIO object.
                - If `decode` is False and `make_readable` is False, returns raw bytes.

        Raises:
            ShippingException: If there's an error reading the object.
        """
        logging.info(f"Reading object '{object_name}' from bucket '{bucket_name}'")
        try:
            obj = boto3.resource("s3").Object(bucket_name, object_name)
            body = obj.get()["Body"].read()

            if decode:
                content = body.decode()
                return StringIO(content) if make_readable else content
            else:
                return BytesIO(body) if make_readable else body

        except ClientError as e:
            raise ShippingException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Gets an S3 bucket object by name.

        Args:
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Bucket: The S3 bucket object.

        Raises:
            ShippingException: If the bucket does not exist or there's an error accessing it.
        """
        logging.info(f"Getting bucket '{bucket_name}'")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            bucket.load()  # Check if the bucket exists
            return bucket
        except ClientError as e:
            raise ShippingException(e, sys) from e

    def is_model_present(self, bucket_name: str, s3_model_key: str) -> bool:
        """
        Checks if a model file with the given key exists in the S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            s3_model_key (str): The key (path) of the model file within the bucket.

        Returns:
            bool: True if the model file exists, False otherwise.
        """
        logging.info(f"Checking if model '{s3_model_key}' exists in bucket '{bucket_name}'")
        try:
            bucket = self.get_bucket(bucket_name)
            objs = list(bucket.objects.filter(Prefix=s3_model_key))
            return any(obj.key == s3_model_key for obj in objs)
        except ClientError as e:
            raise ShippingException(e, sys) from e

    def get_file_object(
        self, filename: str, bucket_name: str
    ) -> Union[List[object], object]:
        """
        Gets file objects from an S3 bucket based on the filename.

        This method is used to get a list of objects that match a given filename 
        or prefix within a specified bucket.

        Args:
            filename (str): The filename (or prefix) to filter objects.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Union[List[object], object]: A list of matching S3 objects if multiple are found, 
                or a single S3 object if only one is found.

        Raises:
            ShippingException: If an error occurs while fetching objects.
        """

        logging.info(f"Getting file object '{filename}' from bucket '{bucket_name}'")

        try:
            bucket = self.get_bucket(bucket_name)
            objs = list(bucket.objects.filter(Prefix=filename))

            if len(objs) == 0:
                raise ShippingException(f"No file object found with the name {filename} in bucket {bucket_name}", sys)

            # If only one object is found, it returns that object directly.
            # If multiple objects are found, it returns the list of objects.
            return objs[0] if len(objs) == 1 else objs
        
        except Exception as e:
            raise ShippingException(e, sys) from e
        
    def load_model(self, model_name: str, bucket_name: str, model_dir: Optional[str] = None) -> Any:
        """
        Loads a model from an S3 bucket.

        Args:
            model_name (str): Name of the model file.
            bucket_name (str): Name of the S3 bucket.
            model_dir (Optional[str], optional): Directory within the bucket where the model is stored. 
                                                    Defaults to None.

        Returns:
            Any: The loaded model object.

        Raises:
            ShippingException: If an error occurs during loading.
        """
        logging.info(f"Loading model '{model_name}' from bucket '{bucket_name}'")
        try:
            model_path = model_name if model_dir is None else os.path.join(model_dir, model_name)
            f_obj = self.get_file_object(model_path, bucket_name)
            model_obj = self.read_object(f_obj.key, bucket_name ,decode=False)  # Read as bytes
            model = pickle.loads(model_obj)  
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            raise ShippingException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Creates a folder in an S3 bucket.

        Args:
            folder_name (str): Name of the folder to create.
            bucket_name (str): Name of the S3 bucket.

        Raises:
            ShippingException: If an error occurs during folder creation.
        """
        logging.info(f"Creating folder '{folder_name}' in bucket '{bucket_name}'")
        try:
            self.s3_resource.Object(bucket_name, folder_name + "/").put()
        except ClientError as e:
            logging.exception(e)
            raise ShippingException(e, sys) from e

    def upload_file(
        self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True
    ) -> None:
        """
        Uploads a file to an S3 bucket.

        Args:
            from_filename (str): Local path to the file.
            to_filename (str): Name (key) of the file in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool, optional): Whether to remove the local file after upload. Defaults to True.
        """
        logging.info(f"Uploading '{from_filename}' to '{to_filename}' in bucket '{bucket_name}'")
        try:
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)
            if remove:
                os.remove(from_filename)
                logging.info("Removed local file")
        except ClientError as e:
            raise ShippingException(e, sys) from e


    def upload_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Uploads an entire folder to an S3 bucket.

        Args:
            folder_name (str): Path to the local folder.
            bucket_name (str): Name of the S3 bucket.
        """
        logging.info(f"Uploading folder '{folder_name}' to bucket '{bucket_name}'")
        try:
            for root, _, files in os.walk(folder_name):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_key = os.path.relpath(local_path, folder_name)
                    self.upload_file(local_path, s3_key, bucket_name, remove=False)
        except Exception as e:
            raise ShippingException(e, sys) from e

    def upload_df_as_csv(
        self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str
    ) -> None:
        """
        Uploads a Pandas DataFrame to S3 as a CSV file.

        Args:
            data_frame (DataFrame): The DataFrame to upload.
            local_filename (str): Temporary local path to save the CSV file.
            bucket_filename (str): Name (key) of the CSV file in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.
        """
        logging.info(f"Uploading DataFrame as CSV to '{bucket_filename}' in bucket '{bucket_name}'")
        try:
            data_frame.to_csv(local_filename, index=False)
            self.upload_file(local_filename, bucket_filename, bucket_name)
        except Exception as e:
            raise ShippingException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Reads a CSV file from an S3 object and returns a DataFrame.

        Args:
            object_ (object): The S3 object representing the CSV file.

        Returns:
            DataFrame: The DataFrame containing the data from the CSV file.
        """
        logging.info("Reading DataFrame from S3 object")
        try:
            content = self.read_object(object_.key, object_.bucket_name, make_readable=True)  
            return pd.read_csv(content, na_values="na")
        except Exception as e:
            raise ShippingException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Reads a CSV file directly from S3 and returns a DataFrame.

        Args:
            filename (str): Name (key) of the CSV file in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.

        Returns:
            DataFrame: The DataFrame containing the data from the CSV file.
        """
        logging.info(f"Reading CSV '{filename}' from bucket '{bucket_name}'")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(csv_obj)
        except Exception as e:
            raise ShippingException(e, sys) from e
