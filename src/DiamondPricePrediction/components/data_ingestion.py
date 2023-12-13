import pandas as pd
import numpy as np
from src.DiamondPricePrediction.logger import logging
from sklearn.model_selection import train_test_split
from src.DiamondPricePrediction.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
import os
import sys


class DataIngestionConfig:
    raw_data_path:str = os.path.join("Artifacts","raw.csv")
    trained_data_path:str = os.path.join("Artifacts","train.csv")
    test_data_path:str = os.path.join("Artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def Initiate_data_ingestion(self):
        logging.info("data ingestion started")

        try:
            data = pd.read_csv(Path(os.path.join('notebooks/data','gemstone.csv')))
            logging.info('I have read the dataset as dataframe')

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("I have saved the raw data in artifact folder")
            
            logging.info('here i have performed train_test_spilt')
            train_data,test_data = train_test_split(data,test_size=0.30)
            logging.info('train test split completed')

            train_data.to_csv(self.ingestion_config.trained_data_path,index = False)
            test_data.to_csv(self.ingestion_config.test_data_path,index = False)
            logging.info("data ingestion path completed")

            return(
                self.ingestion_config.trained_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception during occured at data ngestion stage')
            raise CustomException(e,sys)