import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class dataingestionconfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.config = dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Raw data read successfully")
            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path,index=True, header=True)
            logging.info("Raw data saved to artifacts folder")
            train_set,test_set= train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.config.train_data_path,index=False,header=True)
            test_set.to_csv(self.config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            return(
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()