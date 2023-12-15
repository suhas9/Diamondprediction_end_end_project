import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.join(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    def evalute_model(X_train,y_train,X_test,y_test,models):
        try:
            report = []
            for i in range(len(models)):
                model = list(models.values()[i])
                #train model
                model.fit(X_train,y_train)
        except Exception as e:
            raise CustomException(e,sys)