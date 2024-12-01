import sys
from src.logger import logging

# This function represents how error message should look like in the file w.r.t. custom exception
def error_message_detail(error,error_detail:sys): # error detail is sys type
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)) # file name, line number, error message

    return error_message

#create own CustomException class. Populate error_message variable from error_message_detail function
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # Since we are inhereting the Exception class so we need to inherit the init function.
        self.error_message=error_message_detail(error_message,error_detail=error_detail) # initialize error_messge and call the custom error function
    
    def __str__(self):
        return self.error_message # inhereting str function to print error message

# Checking whether its working or not

# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero")
#         raise CustomException(e, sys)