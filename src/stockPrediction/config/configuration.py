from src.stockPrediction.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.stockPrediction.utils.common import create_directories, read_yaml
from src.stockPrediction.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, DataTrainingConfig, ModelEvaluationConfig, PredictionConfig

class ConfigurationManager:
    def __init__( 
                 self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH,
                 schema_file_path = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config['artifacts_root']])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        data_ingestion_config = DataIngestionConfig(
            symbol= config['symbol'],
            local_data_file_path= config['local_data_file_path'],
            start_year= config['start_year']
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = self.schema['columns']
        data_validation_config = DataValidationConfig(
            dataset_file_path=config['dataset_file_path'],
            root_dir=config['root_dir'],
            validation_status_file_path=config['validation_status_file_path'],
            dataset_schema=schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        data_transformation_config = DataTransformationConfig(
            root_dir= config['root_dir'],
            dataset_file_path= config['dataset_file_path'],
            preprocessor_obj_path= config['preprocessor_obj_path'],
            processed_dataset_file_path= config['processed_dataset_file_path'],
            dataset_summary_path=config['dataset_summary_path'],
            targer_colunm= self.schema['targer_colunm'],
            train_size= float(config['train_size']),
            stock_symbol= config['stock_symbol'],
            train_dataset_file_path= config['train_dataset_file_path'],
            test_dataset_file_path= config['test_dataset_file_path'],
            window_size = config['window_size']
        )
        return data_transformation_config
    
    def get_data_training_config(self) -> DataTrainingConfig:
        config = self.config['model_trainer']
        params = self.params['model_trainer']
        data_training_config = DataTrainingConfig(
            model_name = config['model_name'],
            models = '',
            params= params['params'],
            root_dir= config['root_dir'],
            test_data_path= config['test_data_path'],
            train_data_path= config['train_data_path']            
        )
        return data_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config['root_dir'],
            mlflow_uri=config['mlflow_uri'],
            model_path=config['model_path'],
            test_data_path=config['test_data_path']
        )
        return model_evaluation_config
        
    
    def get_prediction_config(self)-> PredictionConfig:
        config = self.config['prediction_config']
        prediction_config = PredictionConfig(
            model_path=config['model_path'],
            preprocessor_path=config['preprocessor_path']
        )
        return prediction_config