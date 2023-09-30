import os
import sys
import numpy as ny
import pandas as pd
from src.log import logging
from src.utils import save_obj
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class TransformConfig:
    preprocessor_filepath:str = os.path.join('storage', 'preprocessor.dill')

class Transformation:

    def __init__(self):
        self.transform_config = TransformConfig()

    def get_transformation(self):
        try:
            logging.info('Data pipeline Initiated')
            preprocessor = Pipeline([('pca', PCA(n_components=5)),('scaler', StandardScaler())])
            logging.info('Data pipeline Completed')
            return preprocessor
        except Exception as e:
            logging.info('Error while getting transformation in Data Transformation')
            raise CustomException(e,sys) # type: ignore

    def start_transformation(self, train_path, test_path):
        try:
            # Getting X and Y from Train and Test
            Train = pd.read_csv(train_path)
            Test = pd.read_csv(test_path)
            # These columns are selected after doing Feature Engneering
            selected_columns = ['worst concave points', 'worst area', 'worst perimeter', 'worst radius', 'mean concave points', 'target']
            logging.info("'worst concave points', 'worst area', 'worst perimeter', 'worst radius' & 'mean concave points' along with 'target' columns are selected after doing Feature Engneering in notebook" )
            train = Train[selected_columns]
            test = Test[selected_columns]
            X_train = train.drop(columns=['target'], axis=1)
            Y_train = train['target']
            X_test = test.drop(columns=['target'], axis=1)
            Y_test = test['target']
            logging.info('Data fetched for Model training')

            # Scaling through pipeline
            preprocessor = self.get_transformation() 
            X_Train = preprocessor.fit_transform(X_train)
            X_Test = preprocessor.transform(X_test)
            logging.info('scaling is completed')

            # Making of array for Train and Test
            train_arr = ny.c_[X_Train, ny.array(Y_train)]
            test_arr = ny.c_[X_Test, ny.array(Y_test)]

            save_obj(filepath=self.transform_config.preprocessor_filepath, obj=preprocessor)
            logging.info('Data Transformation is completed')
            return (train_arr, test_arr, self.transform_config.preprocessor_filepath)

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)  # type: ignore