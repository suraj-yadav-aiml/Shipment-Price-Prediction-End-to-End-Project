import sys
from shipment.exception import ShippingException
from shipment.logger import logging

from shipment.configuration.mongodb_operations import MongoDBOperation

from shipment.components.data_ingestion import DataIngestion

from shipment.entity.config_entity import (
    DataIngestionConfig
)

from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts
)



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.mongo_op = MongoDBOperation()

    
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, mongo_op=self.mongo_op
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise ShippingException(e, sys) from e
    
    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise ShippingException(e, sys) from e     


