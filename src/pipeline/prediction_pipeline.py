from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.utils import *
from src.exception import CustomException


class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = PREPROCESING_OBJ_FILE
            model_path = MODEL_FILE_PATH
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            data_scale = preprocessor.transform(features)
            
            pred = model.predict(data_scale)
            
            return pred
            
        except Exception as e:
            logging.info("error occurs in prediction pipeline")
            CustomException(e,sys)
            

class CustomData:
    def __init__(self,
                    Delivery_person_Age : int,
                    Delivery_person_Ratings : float,
                    Vehicle_condition : int,
                    Weather_conditions : str,
                    Road_traffic_density : str,
                    multiple_deliveries : int,
                    distance : float,
                    Type_of_order : str,
                    Type_of_vehicle : str,
                    Festival : str,
                    City : str ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Vehicle_condition = Vehicle_condition
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.multiple_deliveries = multiple_deliveries
        self.distance = distance
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Vehicle_condition' : [self.Vehicle_condition],
                'Weather_conditions' : [self.Weather_conditions],
                "Road_traffic_density": [self.Road_traffic_density],
                "multiple_deliveries" : [self.multiple_deliveries],
                "distance" : [self.distance],
                "Type_of_order": [self.Type_of_order],
                "Festival" : [self.Festival],
                "Type_of_vehicle": [self.Type_of_vehicle],
                'City': [self.City]

            }
            
            df = pd.DataFrame(custom_data_input)
            return df
        
        except Exception as e:
            logging.info("error occurs in custom pipeline dataframe")
            CustomException(e,sys)