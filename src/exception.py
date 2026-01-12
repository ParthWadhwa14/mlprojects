import sys

def error_message_details(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown file"
        line_number = "Unknown line"

    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(
            error_message, error_details
        )

    def __str__(self):
        return self.error_message
