from src.stockPrediction.logger import logger
from src.stockPrediction.entity.config_entity import DataTrainingConfig
from src.stockPrediction.utils.common import create_directories, save_object

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import numpy as np

class DataTrainer:
    def __init__(self, config: DataTrainingConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def train(self):
        train_df = pd.read_csv(self.config.train_data_path, )
        logger.info(f"train_df Shape = {train_df.shape}")
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info(f"test_df Shape = {test_df.shape}")
        X_train = train_df.iloc[:,0:60]
        logger.info(f"X_train Shape = {X_train.shape}")
        y_train = train_df.iloc[:,60]
        logger.info(f"y_train Shape = {y_train.shape}")
        X_test = test_df.iloc[:,0:60]
        y_test = test_df.iloc[:,60]
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(8))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        model.fit(X_train, y_train,  epochs=100)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        logger.info(f"RMSE == {rmse}")
        
        logger.info("TRAINING MODELS")
        
        save_object(os.path.join(self.config.root_dir, self.config.model_name), model)
        return ''
        
    def evaluate_models(self,X_train,y_train, X_test,y_test, models:dict, params:dict):
        model_keys = models.keys()
        report = {}
        
        for model_name in model_keys:
            model = models[model_name]
            parameters = params[model_name]

            # GridSearchCV will get best hypermaters for each model
            gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, refit=True)
            gs.fit(X_train, y_train)

            # now test the model with training data

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)
            report[model_name] = {
                'model' : model,
                'R2_score_test' : test_model_score,
                'R2_score_train' : train_model_score,
                'best_params': gs.best_params_
            }
        logger.info(f'Model Evaluation report: \n{report}')
        return report