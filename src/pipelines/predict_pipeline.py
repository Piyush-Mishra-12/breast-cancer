from src.exception import CustomException
from src.log import logging
from flask import request
from src import utils
import os, sys, pandas

class PredictingPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('storage','preprocessor.dill')
            model_path = os.path.join('storage', 'model.dill')
            preprocessor = utils.load_object(preprocessor_path)
            model = utils.load_object(model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in Prediction Pipelines')
            raise CustomException(e,sys) # type: ignore


class CustomData:
    def __init__(self, worst_concave_points:float, worst_area:float, worst_perimeter:float, worst_radius:float, mean_concave_points:float):
        self.worst_concave_points = worst_concave_points
        self.worst_area = worst_area
        self.worst_perimeter = worst_perimeter
        self.worst_radius = worst_radius
        self.mean_concave_points = mean_concave_points

    def get_data(self):
        try:
            custom_data = {'worst concave points':[self.worst_concave_points], 'worst area':[self.worst_area],
                           'worst perimeter':[self.worst_perimeter], 'worst radius':[self.worst_radius],  
                           'mean concave points':[self.mean_concave_points]}
            df = pandas.DataFrame(custom_data)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occured while loading data from user')
            raise CustomException(e,sys) # type: ignore