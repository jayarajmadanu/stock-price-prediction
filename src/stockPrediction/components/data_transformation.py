from src.stockPrediction.entity.config_entity import DataTransformationConfig
from src.stockPrediction.logger import logger
from src.stockPrediction.utils.common import create_directories, save_object

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import datetime
import numpy as np

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def summarise_df(self,df, dataset_summary_path, title:str = ""):
        with open(dataset_summary_path, 'a') as f:
            f.write(title + '\n')
            f.write("df.info() \n")
            f.write(df.info())
            f.write("df.describe(include='number') \n")
            f.write(df.describe(include='number'))
            f.write("df.describe(include='object') \n")
            f.write(df.describe(include='object'))
            
    def transform_data(self) -> ColumnTransformer:
        try:
            ## NOTE: Colunm Transformer will change the order of colunms after applying transformation, so check the value of index mentioned in colunmTransformer after eact CT        
            tr1 = ColumnTransformer([
                ("scalar",MinMaxScaler(feature_range=(0,1)), [0])
            ], remainder='passthrough')
            
            pipeline = Pipeline(
                steps=[
                    ('tr1', tr1),
                ]
            )
            return pipeline
        except Exception as e:
            logger.info(e)
            
    def initiate_data_transformation(self):
        dataset_file_path = self.config.dataset_file_path
        window_size = self.config.window_size
        df = pd.read_csv(dataset_file_path)
        close_df = df[['DATE', 'CLOSE']]
        #close_df['DATE'] = datetime.datetime.strftime(close_df['DATE'], '%Y-%m-%d')
        #close_df['DATE'] = close_df['DATE'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))
        
        # SBI stock value fell down to 297 INR from 2900 INR on Nov 20th 2014 because of stock split of ratio 1:10, let us devide the price of stock before Nov 20th 2014 by 10 to get accurate data
        if(self.config.stock_symbol == 'SBIN'):
            close_df['CLOSE'] = df.apply(lambda row : row['CLOSE']/10 if datetime.datetime.date(datetime.datetime.strptime(row['DATE'], '%Y-%m-%d')) < datetime.date(2014,11,20) else row['CLOSE'] , axis=1)
         
        # Resample dataset to have weekly data   
        close_df['DATE'] = close_df['DATE'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))
        close_df.set_index('DATE', inplace=True, drop=True)
        close_df = close_df['CLOSE'].resample('W').last()
        close_df = pd.DataFrame(close_df)
        
        print(f'len of df = {close_df.shape}')
        close_df = close_df['CLOSE'].values
        training_data_len = int(np.ceil( len(close_df) * self.config.train_size ))
        train_data = close_df[0:int(training_data_len)]
        test_data = close_df[training_data_len - window_size: ]
        
        print(f'len of close_df = {close_df.shape}')
        print(f'len of train = {len(train_data)}')
        print(f'len of test = {len(test_data)}')
        preprocessor = self.transform_data()
        
        scaled_train_data = preprocessor.fit_transform(np.reshape(train_data, (-1, 1)))
        scaled_test_data = preprocessor.transform(np.reshape(test_data, (-1, 1)))
        
        x_train, y_train = self.create_dataset(data=scaled_train_data, window_size=window_size)
        x_test, y_test = self.create_dataset(data=scaled_test_data, window_size=window_size)
        
        train_dataset = np.c_[np.reshape(x_train, (x_train.shape[0], x_train.shape[1])), np.array(y_train)]
        test_dataset = np.c_[np.reshape(x_test, (x_test.shape[0], x_test.shape[1])), np.array(y_test)]
        
        train_dataset = pd.DataFrame(train_dataset)
        logger.info(f"Created train dataset at location {self.config.train_dataset_file_path} with shape {train_dataset.shape}")
        train_dataset.to_csv(self.config.train_dataset_file_path, index=False)
        test_dataset = pd.DataFrame(test_dataset)
        logger.info(f"Created test dataset at location {self.config.test_dataset_file_path} with shape {test_dataset.shape}")
        test_dataset.to_csv(self.config.test_dataset_file_path, index=False)
        
        scaled_data = preprocessor.transform(np.reshape(close_df, (-1, 1)))
        x, y = self.create_dataset(data=scaled_data, window_size=window_size)
        processed_data = np.c_[np.reshape(x, (x.shape[0], x.shape[1])), np.array(y)]
        processed_dataset = pd.DataFrame(processed_data)
        logger.info(f"Created processed_data dataset at location {self.config.processed_dataset_file_path} with shape {train_dataset.shape}")
        processed_dataset.to_csv(self.config.processed_dataset_file_path, index=False)
        
        save_object(self.config.preprocessor_obj_path, preprocessor)
    
    def create_dataset(self, data, window_size):
        
        x = []
        y = []

        for i in range(window_size, len(data)):
            x.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        # Convert the x and y_train to numpy arrays
        x, y = np.array(x), np.array(y)

        # Reshape the data
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        print(f'len of created data = {x.shape} and y = {y.shape}')
        return x, y
                
