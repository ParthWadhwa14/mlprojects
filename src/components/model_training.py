import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models
from sklearn.metrics import r2_score

@dataclass
class model_training_config:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class Model_trainer:
    def __init__(self):
        self.model_training_config=model_training_config()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting the data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Linearregression":LinearRegression(),
                "ridge":Ridge(),
                "Lasso":Lasso(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "SVR":SVR(),
                "CatBoostRegressor":CatBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
}
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No model found")
            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2=r2_score(predicted,y_test)
            return r2

        except Exception as e:
            raise CustomException(e,sys)