import os
import sys
import pandas as pd
import numpy as np
from src.DiamondPricePrediction.exception import CustomException
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            presprocessor_path = os.path.join('Artifacts','preprocessor.pkl')
            model_path = os.path.join('Artifacts','model.pkl')


            preprocessor_obj = load_object(presprocessor_path)
            model_obj = load_object(model_path)

            scaled_data = preprocessor_obj.transform(features)
            pred = model_obj.predict(scaled_data)
            return pred

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:float):
        self.carat = carat
        self.depth = depth
        self.table =  table 
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def Get_data_as_dataframe(self):
        try:
            custom_data_input_dict  = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        except Exception as e:
            logging.info("Eception has occured in prediction pipeline")
            raise CustomException(e,sys)