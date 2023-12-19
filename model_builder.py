import os 
import sys 
from src_.logge import logging
from src_.exception import CustomException
from dataclasses import dataclass
import pickle 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
@dataclass
class model_build_config:
    train_obj_path = os.path.join("artifacts","model.pkl")

class model_build:
    def __init__(self):
        self.model_build_path = model_build_config()
    def initiate_model_building(self,train_arr,test_arr,preprocessing_path):
        try:
            logging.info('model building started')
           

            xtrain = train_arr[:,:-1]
            ytrain = train_arr[:,-1]
            xtest = test_arr[:,:-1]
            ytest = test_arr[:,-1]

            rfr = RandomForestRegressor()
            rfr.fit(xtrain,ytrain)
            pred = rfr.predict(xtest)

            logging.info(f'saving model object')
            
            os.makedirs(os.path.dirname(self.model_build_path.train_obj_path),exist_ok=True)

            with open(self.model_build_path.train_obj_path,'wb') as ob:
                pickle.dump(rfr,ob)

            rsquare = r2_score(ytest,pred)
        
            return rsquare




        except Exception as e:
            raise CustomException(e,sys)


        