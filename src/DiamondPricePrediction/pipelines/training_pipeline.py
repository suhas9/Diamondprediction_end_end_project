from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException
import os,sys
import pandas as pd
import numpy as np

obj = DataIngestion()
obj.Initiate_data_ingestion()