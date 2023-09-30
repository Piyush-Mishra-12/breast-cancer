import os
import sys
from src import utils
from src.log import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@dataclass
class TrainerConfig:
    trainer_filepath:str = os.path.join('storage', 'model.dill')


class Model:
    def __init__(self, preprocessor_obj, model_obj):
        self.preprocessor_obj = preprocessor_obj
        self.model_obj = model_obj
    def predict(self, x):
        transformed_feature = self.preprocessor_obj.transform(x)
        return self.model_obj.predict(transformed_feature)
    def __repr__(self):
        return f'{type(self.model_obj).__name__}()'
    def __str__(self):
        return f'{type(self.model_obj).__name__}()'


class Trainer:
    def __init__(self):
        self.trainer_config = TrainerConfig()
    
    def start_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info('Splitting training and test datasets')
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            # Models and their names
            models = {'Support Vector Classifier': SVC(), 'Decision Tree': DecisionTreeClassifier(),
                      'Random Forest': RandomForestClassifier(), 'K-Nearest Neighbors': KNeighborsClassifier()}
            
            logging.info('Extracting model config file path')
            model_report = utils.evaluate(x=x_train, y=y_train, models=models)
            
            # To get best model
            bmodel_name, bmodel_score = max(model_report.items(), key=lambda item: item[1])
            bmodel = models[bmodel_name]
            if bmodel_score < 0.6:
                raise Exception('No model is good enough')
            logging.info(f'Best model which is selected is {bmodel_name} with score of {bmodel_score}')
            
            preprocessor_obj = utils.load_object(filepath=preprocessor_path)
            custom_model = Model(preprocessor_obj=preprocessor_obj, model_obj=bmodel)
            logging.info(f'Saving model at path: {self.trainer_config.trainer_filepath}')

            utils.save_obj(filepath=self.trainer_config.trainer_filepath, obj=custom_model)
            predicted = bmodel.predict(x_test)
            r2 = accuracy_score(y_test, y_pred= predicted)
            return r2
        except Exception as e:
            logging.info('Error in Data Training')
            raise CustomException(e,sys) # type: ignore