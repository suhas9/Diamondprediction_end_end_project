import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException

from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.DiamondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preporcessor_obj_file_path = os.path.join('Artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("data tranformation initiated")
            #define which column is categorical nad numerical and which should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            #Defined the custom ranking for each ordinal variable
            cut_categories = {'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'}
            clarity_categories = {'I1', 'SI2', 'SI2', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'}
            color_categories = {'D', 'E', 'F', 'G', 'H', 'I', 'J'}

            logging.info("pipeline initiated")
            # Numerical pipeline
            num_pipeline = pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ("scler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most frequent')),
                    ('Ordinalencoding',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipline',cat_pipeline,categorical_cols)
            ])
            return preprocessor


        except Exception as e:
            logging.info('Error occured in the initiate_datatransformation')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  =pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info(f"Train Dataframe Head: \n {train_df.head().to_string()}")
            logging.info(f'Test dataframe Head:\n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation()

            traget_column_name = 'price'
            drop_column = [traget_column_name,'id']

            input_feature_train_df = train_df.drop(column=drop_column,axis = 1)
            target_feature_train_df  = train_df[traget_column_name]
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[traget_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preporcessor_obj_file_path,
                obj = preprocessing_obj
                )
            
            return (
                train_arr,
                test_arr
            )



        except Exception as e:
            logging.info('Error occured in the initiate_datatransformation')
            raise CustomException(e,sys)


