import logging
import os
from datetime import datetime
from typing import Union

from from_root import from_root 
from shipment.constant import LOGGING_DIR


class CustomLogger:
    """
    A custom logger class for creating and configuring loggers.
    """

    def __init__(self, log_file: Union[str, None] = None, log_level: int = logging.INFO):
        """
        Initializes the CustomLogger object.

        Args:
            log_file (str, optional): The path to the log file. If None, a default timestamped file is used.
            log_level (int, optional): The logging level to set (e.g., logging.INFO, logging.DEBUG).
        """
        self.log_file = log_file
        self.log_level = log_level

    def get_log_file_name(self) -> str:
        """Generates a log file name based on the current timestamp.

        Returns:
            str: Log file name in the format "DD_MM_YYYY_HH_MM_SS.log".
        """
        return f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

    def create_logger(self) -> logging.Logger:
        """
        Creates and configures a logger object to write logs to a file.

        Returns:
            logging.Logger: The configured logger object.
        """

        if self.log_file is None:
            self.log_file = self.get_log_file_name()

        log_path = os.path.join(from_root(), LOGGING_DIR, self.log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logging.basicConfig(
            filename=log_path,
            format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
            level=self.log_level
        )

        return logging.getLogger(__name__) 



logger = CustomLogger().create_logger()


