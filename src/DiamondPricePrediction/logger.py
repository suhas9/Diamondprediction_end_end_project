import logging 
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
Log_path = os.path.join(os.getcwd(),"log")

os.makedirs(Log_path,exist_ok=True)

Log_filepath = os.path.join(Log_path,Log_file)

logging.basicConfig(level=logging.INFO,
                    filename=Log_filepath,
                    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s - %(message)s")