import sys
from dataclasses import dataclass
from typing import Optional ,Any

import pandas as pd

from shipment.logger import logger
from shipment.exception import ShippingException

from shipment.entity.config_entity import ModelEvaluationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifact
)
from shipment.constant import *


@dataclass
class EvaluateModelResponse:
    """
    Encapsulates the result of model evaluation.
    
    Attributes:
        trained_model_r2_score (float): R-squared score of the trained model.
        s3_model_r2_score (float or None): R-squared score of the model from S3 (or None if not available).
        is_model_accepted (bool): Whether the trained model is accepted based on the comparison.
        difference (float): The difference in R-squared scores between the two models.
    """
    trained_model_r2_score: float
    s3_model_r2_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    Class for evaluating and comparing machine learning models for shipment cost prediction.
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifacts,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifacts,
    ):
        """
        Initializes the ModelEvaluation object.

        Args:
            model_trainer_artifact (ModelTrainerArtifacts): Artifacts from the model training stage.
            model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation.
            data_ingestion_artifact (DataIngestionArtifacts): Artifacts from the data ingestion stage.
        """
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def get_s3_model(self) -> Optional[Any]: 
        """
        Retrieves the model from the S3 bucket if it exists.

        Returns:
            Optional[Any]: The loaded model object if found in S3, None otherwise.
        """
        logger.info("Checking for existing model in S3")
        if self.model_evaluation_config.S3_OPERATIONS.is_model_present(
            BUCKET_NAME, S3_MODEL_NAME
        ):
            return self.model_evaluation_config.S3_OPERATIONS.load_model(
                MODEL_FILE_NAME, BUCKET_NAME
            )
        else:
            logger.info("No existing model found in S3")
            return None


    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluates the trained model against the existing S3 model (if available).

        Calculates the R-squared scores for both models, compares them, and determines 
        whether the trained model should be accepted.

        Returns:
            EvaluateModelResponse: A dataclass containing evaluation results.
        """
        logger.info("Starting model evaluation")

        try:
            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_data_file_path
            )
            test_df = test_df.drop(columns=self.model_evaluation_config.DROP_COLS)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            
            trained_model = self.model_evaluation_config.UTILS.load_object(
                self.model_trainer_artifact.trained_model_file_path
            )
            y_pred_trained = trained_model.predict(x)
            trained_model_r2_score = self.model_evaluation_config.UTILS.get_model_score(y, y_pred_trained)
            
            # Load and evaluate the S3 model
            s3_model = self.get_s3_model()

            if s3_model is not None:
                y_pred_s3 = s3_model.predict(x)
                s3_model_r2_score = self.model_evaluation_config.UTILS.get_model_score(y, y_pred_s3)
            else:
                s3_model_r2_score = None  # Assign None if no S3 model exists

            # Determine if the trained model is better
            tmp_best_model_score = 0.0 if s3_model_r2_score is None else s3_model_r2_score
            is_model_accepted = trained_model_r2_score > tmp_best_model_score

            difference = trained_model_r2_score - tmp_best_model_score

            evaluation_response = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                s3_model_r2_score=s3_model_r2_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )

            logger.info(
                f"Model evaluation completed. Trained model score: {trained_model_r2_score}, S3 model score: {s3_model_r2_score}, Accepted: {is_model_accepted}"
            )

            return evaluation_response

        except Exception as e:
            raise ShippingException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates the model evaluation process.

        Calls evaluate_model() to get the evaluation results, then creates a ModelEvaluationArtifact
        with the results.

        Returns:
            ModelEvaluationArtifact: The model evaluation artifact containing the evaluation results.
        """
        logger.info("Initiating model evaluation")
        try:
            evaluate_model_response = self.evaluate_model()
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )
            logger.info("Model evaluation completed")
            return model_evaluation_artifact
        except Exception as e:
            raise ShippingException(e, sys) from e
