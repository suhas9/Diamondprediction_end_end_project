import os,sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException


from src.DiamondPricePrediction.utils.utils import save_object
from src.DiamondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('Artifacts','model.pkl')



class ModelTrainer:
    def __init__(self):
       self.model_trainer_config = ModelTrainingConfig()

    def Initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variable from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train,X_test,y_test,models)
            print(model_report)
            print('\n===============================================================\n')
            logging.info(f'model report:{model_report}')

            #to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f'Best model found,Model Name:{best_model_name},R2 score:{best_model_score}')
            print('\n=========================================================================\n')
            logging.info(f'Best model found ,Model name:{best_model_name},R2 score:{best_model_score}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info('preprocessor pickel file saved')
            return (
                train_array,
                test_array
            )


        except Exception as e:
            raise CustomException(e,sys)