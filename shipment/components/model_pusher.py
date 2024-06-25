import sys
from typing import Optional

from shipment.exception import ShippingException
from shipment.logger import logging

from shipment.configuration.s3_oprations import S3Operation

from shipment.entity.artifacts_entity import (
    DataTransformationArtifacts,
    ModelPusherArtifacts,
    ModelTrainerArtifacts,
)
from shipment.entity.config_entity import ModelPusherConfig


class ModelPusher:
    """
    Facilitates pushing the best trained model to an S3 bucket for deployment.

    This class handles the process of taking the trained model artifact produced by the 
    ModelTrainer and uploading it to a designated location in an S3 bucket
    """

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
        s3: S3Operation,
    ):
        """
        Initializes a ModelPusher object.

        Args:
            model_pusher_config (ModelPusherConfig): Configuration object for model pushing.
            model_trainer_artifacts (ModelTrainerArtifacts): Artifacts containing the trained model path.
            data_transformation_artifacts (DataTransformationArtifacts): Artifacts from data transformation.
            s3 (S3Operation): An object for handling S3 interactions.
        """
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.s3 = s3

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Initiates the model pushing process.

        Uploads the best trained model from the model trainer artifacts to the S3 bucket 
        specified in the configuration.

        Returns:
            ModelPusherArtifacts: An object containing details about the uploaded model in S3.

        Raises:
            ShippingException: If an error occurs during the model pushing process.
        """

        try:
            logging.info("Initiating model pushing")

            # Upload the trained model to the S3 bucket
            self.s3.upload_file(
                from_filename=self.model_trainer_artifacts.trained_model_file_path,
                to_filename=self.model_pusher_config.S3_MODEL_KEY_PATH,
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                remove=False,  # Do not remove the local file after upload
            )

            logging.info(f"Model pushed to S3 bucket: {self.model_pusher_config.BUCKET_NAME}")
            logging.info(f"Model path in S3: {self.model_pusher_config.S3_MODEL_KEY_PATH}")

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )
            logging.info("Model pushing completed")
            return model_pusher_artifact

        except Exception as e:
            raise ShippingException(e, sys) from e
