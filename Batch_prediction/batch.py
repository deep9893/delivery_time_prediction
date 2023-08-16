from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
 
from src.logger import logging
from src.exception import CustomException
import pickle
from src.utils import load_model


PREDICTION_FOLDER ="batch_predictions"
PREDICTION_CSV = "prediction_csv"
PREDICTION_FILE = "output.csv"
FEATURE_ENG_FOLDER = "feature_eng"


ROOT_DIR = os.getcwd()
BATH_PREDICTION = os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)

class batch_prediction:
    def __init__(self,input_file_path,
                 model_file_path,
                 transformer_file_path,
                 feature_engineering_file_path)-> None:
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path
        
    def start_batch_prediction(self):
        try:
            # load the feature engineering pipeline path
            with open(self.feature_engineering_file_path,'rb') as f:
                feature_pipeline= pickle.load(f)
                
            # load the data transformation pipeline path
            with open(self.transformer_file_path,'rb') as f:
                processor = pickle.load(f)
                
            # load the model seperately
            model = load_model(file_path=self.model_file_path)
            
            # create feature engineering pipeline
            
            feature_engineering_pipeline = Pipeline([
                ("feature_engineering", feature_pipeline)
            ])
                
            df = pd.read_csv(self.input_file_path)
            df.to_csv("delivery_time_predictions.csv")
            
            # apply the feature engineering pipelinesteps
            
            df = feature_engineering_pipeline.transform(df)
            
            df.to_csv("feature_engineering.csv")
            
            FEATURE_ENGINEERING_PATH =  FEATURE_ENG
            os.makedirs(FEATURE_ENGINEERING_PATH, exist_ok=True)
            
            file_path = os.path.join(FEATURE_ENGINEERING_PATH,"batch_feature_eng.csv")
            
            df.to_csv(file_path, index = False)
            
            # time_taken
            
            df = df.drop('Time_taken (min)', axis = 1)
            df.to_csv("time_taken_droped.csv")
            
            transformed_data = processor.transform(df)
            
            file_path = os.path.join(FEATURE_ENGINEERING_PATH,"processor.csv")
            predictions = model.predict(transformed_data)
               
            
            df_prediction = pd.DataFrame(predictions, columns=['predictions'])
            
            
            
            BATH_PREDICTION_PATH = BATH_PREDICTION
            os.path.join(BATH_PREDICTION_PATH,exist_ok = True)
            csv_path = os.path.join(BATH_PREDICTION_PATH,'output.csv')
            
            df_prediction.to_csv(csv_path,index=False)
            logging.info(f"batch prediction done")
               
               
        except Exception as e:
            raise CustomException(e,sys)