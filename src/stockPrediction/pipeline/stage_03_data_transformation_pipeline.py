from src.stockPrediction.config.configuration import ConfigurationManager
from src.stockPrediction.logger import logger
from src.stockPrediction.entity.config_entity import DataTransformationConfig
from src.stockPrediction.utils.common import create_directories
from src.stockPrediction.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self, config: DataTransformationConfig = None):
        if config == None:
            config = ConfigurationManager().get_data_transformation_config()
        self.config = config
        
    def main(self):
        data_transformation = DataTransformation(config= self.config)
        data_transformation.initiate_data_transformation()
        
if __name__ == '__main__':
    try:
        obj = DataTransformationPipeline()
        obj.main()
    except Exception as e:
        raise e