from src.stockPrediction.entity.config_entity import DataIngestionConfig
from src.stockPrediction.logger import logger
import os
from src.stockPrediction.utils.common import create_directories
from jugaad_data.nse import stock_df 
import datetime

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def downloadDataSet(self):
        symbol = self.config.symbol
        start_year = self.config.start_year
        file_path = self.config.local_data_file_path
        logger.info(f'Downloading data of stock symbol{symbol} from start date {start_year}')
        df = stock_df(symbol= symbol, from_date=datetime.date(start_year,1,1), to_date=datetime.date.today(), series="EQ")
        df.to_csv(file_path)
        logger.info(f'Downloaded dataset into location {file_path}')
            
    
    