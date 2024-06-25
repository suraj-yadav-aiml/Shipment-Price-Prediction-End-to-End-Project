import sys
from typing import Dict, List, Any,Union

import pandas as pd
import numpy as np
from pandas import DataFrame

from shipment.logger import logging
from shipment.exception import ShippingException
from shipment.constant import *
from shipment.configuration.s3_oprations import S3Operation
from shipment.components.model_trainer import CostModel


class ShippingData:
    """
    Represents shipping data used for making predictions.

    Attributes:
        artist (float): Artist reputation.
        height (float): Height of the shipment.
        width (float): Width of the shipment.
        weight (float): Weight of the shipment.
        material (str): Material of the shipment.
        priceOfSculpture (float): Price of the sculpture.
        baseShippingPrice (float): Base shipping price.
        international (str): Whether the shipment is international.
        expressShipment (str): Whether the shipment is express.
        installationIncluded (str): Whether installation is included.
        transport (str): Mode of transport.
        fragile (str): Whether the shipment is fragile.
        customerInformation (str): Customer information.
        remoteLocation (str): Whether the location is remote.
    """
    def __init__(
        self,
        artist: float,
        height: float,
        width: float,
        weight: float,
        material: str,
        priceOfSculpture: float,
        baseShippingPrice: float,
        international: str,
        expressShipment: str,
        installationIncluded: str,
        transport: str,
        fragile: str,
        customerInformation: str,
        remoteLocation: str,
    ):
        self.artist = artist
        self.height = height
        self.width = width
        self.weight = weight
        self.material = material
        self.priceOfSculpture = priceOfSculpture
        self.baseShippingPrice = baseShippingPrice
        self.international = international
        self.expressShipment = expressShipment
        self.installationIncluded = installationIncluded
        self.transport = transport
        self.fragile = fragile
        self.customerInformation = customerInformation
        self.remoteLocation = remoteLocation

    def get_data(self) -> Dict:  
        """
        Gets shipping data as a dictionary for model input.

        Returns:
            Dict[str, List[Any]]: A dictionary where keys are feature names and values are lists containing the feature value.
        """
        logging.info("Entered get_data method of ShippingData class")

        # Save the features as a dictionary
        input_data: Dict = {
            "Artist Reputation": [self.artist],
            "Height": [self.height],
            "Width": [self.width],
            "Weight": [self.weight],
            "Material": [self.material],
            "Price Of Sculpture": [self.priceOfSculpture],
            "Base Shipping Price": [self.baseShippingPrice],
            "International": [self.international],
            "Express Shipment": [self.expressShipment],
            "Installation Included": [self.installationIncluded],
            "Transport": [self.transport],
            "Fragile": [self.fragile],
            "Customer Information": [self.customerInformation],
            "Remote Location": [self.remoteLocation],
        }
        
        logging.info("Exited get_data method of ShippingData class")
        return input_data

    def get_input_data_frame(self) -> DataFrame:
        """
        Converts shipping data to a Pandas DataFrame.

        Returns:
            DataFrame: A DataFrame containing the shipping data.
        """
        logging.info("Entered get_input_data_frame method of ShippingData class")
        try:
            return pd.DataFrame(self.get_data())
        except Exception as e:
            raise ShippingException(e, sys) from e


class CostPredictor:
    """
    Class for making cost predictions using a model loaded from S3.
    """

    def __init__(self):
        """
        Initializes the CostPredictor with an S3Operation object for model loading.
        """
        self.s3 = S3Operation()
        self.bucket_name = BUCKET_NAME  

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Predicts the cost using the best model loaded from S3.

        Args:
            X (DataFrame): Input DataFrame containing the features for prediction.

        Returns:
            np.ndarray: Predicted cost values as a NumPy array.

        Raises:
            ShippingException: If there is an error loading the model or making predictions.
        """
        logging.info("Entering predict method of CostPredictor class")
        try:
            best_model: CostModel = self.s3.load_model(MODEL_FILE_NAME, self.bucket_name)  
            logging.info(f"Loaded model: {best_model}")

            result = best_model.predict(X)  
            logging.info("Exiting predict method of CostPredictor class")
            return result

        except Exception as e:
            raise ShippingException(e, sys) from e
