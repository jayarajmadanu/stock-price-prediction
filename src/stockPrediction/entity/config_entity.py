from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    symbol: Path
    local_data_file_path: Path
    start_year: int
    
@dataclass
class DataValidationConfig:
    root_dir: Path
    dataset_file_path: Path
    validation_status_file_path: Path
    dataset_schema: dict
    
@dataclass
class DataTransformationConfig:
    root_dir: Path
    dataset_file_path: Path
    processed_dataset_file_path: Path
    preprocessor_obj_path: Path
    dataset_summary_path: Path
    targer_colunm: str
    train_size: float
    stock_symbol: str
    train_dataset_file_path: Path
    test_dataset_file_path: Path
    window_size: int
    
@dataclass
class DataTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    models: dict
    params: dict
    
@dataclass
class ModelEvaluationConfig:
    root_dir:Path
    mlflow_uri: str
    test_data_path: Path
    model_path: Path
    
@dataclass
class PredictionConfig:
    preprocessor_path: Path
    model_path: Path
    