import traceback
import sys


class CustomException(Exception):
    def __init__(self, message, error_detail = sys):
        super().__init__(message)
        self.error_message = self.get_detail(error_message=message, error_detail=error_detail)

    @staticmethod
    def get_detail(error_message, error_detail = sys):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error occured in {file_name}, line {line_number} : {exc_tb.tb_lineno}"

    def __str__(self):
        return self.error_message