import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
           
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sys
from src.exception import CustomException
from src.logger import logging

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            
            logging.info(f"Training {model_name}")

            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


    