import os
import sys
import dill
import numpy as ny
import pandas as pd
from src.log import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from pymongo import MongoClient


def export_collection_as_dataframe(c_name, db_name):
    try:
        MONGO_DB_URL = "mongodb+srv://Piyush:Mfkc1QAe0V4f9fgN@cluster0.opvarp6.mongodb.net/?retryWrites=true&w=majority"
        mongo_client = MongoClient(MONGO_DB_URL)
        collection = mongo_client[db_name][c_name]
        df = pd.DataFrame(list(collection.find()))
        if '_id' in df.columns.to_list():
            df = df.drop(columns=['_id'], axis=1)
        df.replace({'na':ny.nan}, inplace=True)
        return df
    except Exception as e:
        logging.info('Error in collecting file from mongoDB')
        raise CustomException(e,sys) # type: ignore

def save_obj(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.info('Error in saving dill file')
        raise CustomException(e,sys) # type: ignore

def load_object(filepath):
    try:
        with open(filepath,'rb') as file_obj:
            logging.info('Loading object from utils is completed')
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Error occured while loading object from utils')
        raise CustomException(e,sys) # type: ignore

def evaluate(x, y, models):
            try:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=47)
                report = {}
                for model, m in models.items():
                    m.fit(x_train, y_train)
                    y_pred = m.predict(x_test)
                    cv_scores = cross_val_score(m, x_train, y_train, cv=10)
                    accuracy_cv = cv_scores.mean()
                    accuracy = accuracy_score(y_test, y_pred)
                    report[model] = accuracy
                return report
            except Exception as e:
                logging.info('Error occured while evaluting model from utils')
                raise CustomException(e, sys) # type: ignore