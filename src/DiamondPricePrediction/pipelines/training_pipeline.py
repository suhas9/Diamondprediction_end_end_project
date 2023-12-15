from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_training import ModelTrainer
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException
import os,sys
import pandas as pd
import numpy as np

obj = DataIngestion()
train_data_path,test_data_path = obj.Initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_tainer_obj = ModelTrainer(train_arr,test_arr)
model_tainer_obj.Initiate_model_trainer(train_arr,test_arr)