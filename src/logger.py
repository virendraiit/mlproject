# Logger file logs every execution in the script 
# so that if any exception occure it would be logged/writtten in a text file

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #date/time wise file
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # Get the current working directory
os.makedirs(logs_path,exist_ok=True) # Even if directory exist keep that file

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# As we need to overwrite the file(here logfile) we need to set it in config(basicconfig)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__=="__main__":
#     logging.info("Logging has started")