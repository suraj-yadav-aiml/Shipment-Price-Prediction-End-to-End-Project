import os
import sys 

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed information from an exception and formats it into a string.

    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module for accessing exception information.

    Returns:
        str: A formatted error message containing the file name, line number, and error details.
    """
    _, _, exc_tb = error_detail.exc_info()  
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script name [{os.path.split(file_name)[1]}] "
        f"line number [{line_number}] error message [{error}]"
    )


class ShippingException(Exception):
    """
    Custom exception class for handling shipping-related errors.
    """

    def __init__(self, error_message: str, error_detail: sys): 
        """
        Initializes a ShippingException instance.

        Args:
            error_message (str): The error message.
            error_detail (sys): The sys module for detailed information.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        """
        Returns a string representation of the exception.
        """
        return self.error_message
