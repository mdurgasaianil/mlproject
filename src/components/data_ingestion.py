# in this file we will wirte the code to fetching and read the data from different sources and split them into train and test and save them

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # this used to create an class variable


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


# any input that is required while doing data injestion that we will take input from below class
@dataclass
class DataIngestionConfig:
    # Since we using the data class decorator we were no need of creating __init__ for creating a class
    # we can directly creata class
    train_data_path: str = os.path.join('artifacts',"train.csv") # all outputs will be saved under artifact folder, so here the train data will be stored in this path
    test_data_path: str = os.path.join('artifacts',"test.csv") 
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() 

    def initiate_data_ingestion(self): # under this function we can write the code to fetech the data from multiple sources like mongoddb, sql, etc.
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'D:\mlproject\Notebook\Data\stud.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # creating a artifacts folder directory

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    



