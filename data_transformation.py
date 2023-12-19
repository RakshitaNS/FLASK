import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import sys
from src_.exception import CustomException
from src_.logge import logging
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from dataclasses import dataclass
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocesso.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            le =LabelEncoder()
            train_df = train_df.drop(columns=['Unnamed: 0','flight'])
            test_df = test_df.drop(columns=['Unnamed: 0','flight'])
            for i in train_df.columns:
                if train_df[i].dtype=='O':
                    train_df[i] = le.fit_transform(train_df[i])
            for j in test_df.columns:
                if test_df[j].dtype=='O':
                    test_df[j] = le.transform(test_df[j])

            train_arr = np.array(train_df)
            test_arr =np.array(test_df)
            logging.info(f"Saved preprocessing object.")
            #split into independent and dependent feature 
            # feature_train_df = pd.DataFrame(train_df[:,:-1])
            # target_train_df = pd.DataFrame(train_df[:,-1])

            # feature_test_df = pd.DataFrame(test_df[:,:-1])
            # target_test_df = pd.DataFrame(test_df[:,-1])
            #perform label encoding 
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),exist_ok=True)

            #storing the preprocessor.pkl file in the created directory 
            file_path =self.data_transformation_config.preprocessor_obj_file_path
            with open(file_path,'wb') as obj_path:
                pickle.dump(le,obj_path)


            return (train_arr,
                test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        

        except Exception as e:
            raise CustomException(e,sys)
        


    

            



# class transfomrer_object:
#     le =LabelEncoder()
#     for i in data.columns:
#         if data[i].dtype=='O':
#             data[i] =le.fit_transform(data[i])

