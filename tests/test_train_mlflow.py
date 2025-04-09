import pytest
import pandas as pd
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock
from src.model.train_mlflow import TrainingPipeline

@patch("src.model.train_mlflow.mlflow")
@patch("src.utils.utils.ModelUtils.save_model")
@patch("src.model.train_mlflow.TrainingPipeline.save_confusion_matrix", return_value="mock_cm_path.png")
@patch("src.data.load_data.DataLoader.download_data")
@patch("src.data.load_data.DataLoader.load_data")
@patch("src.data.preprocess_data.DataPreprocessor.preprocess_data")
@patch("src.data.preprocess_data.DataPreprocessor.encode_features")
@patch("src.utils.utils.ModelUtils.ensure_directory_exists")
def test_training_pipeline_runs_successfully(
    mock_ensure_dirs,
    mock_encode_features,
    mock_preprocess_data,
    mock_load_data,
    mock_download_data,
    mock_save_conf_matrix,
    mock_save_model,
    mock_mlflow
):
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['main_cat'] = y

    # Setup mocks
    mock_load_data.return_value = df
    mock_preprocess_data.return_value = df
    mock_encode_features.return_value = df.drop(columns=['main_cat'])

    # Mock MLflow methods
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Initialize the TrainingPipeline
    pipeline = TrainingPipeline()

    # Run the pipeline
    pipeline.run()

    # Assert pipeline completed steps
    mock_ensure_dirs.assert_called()
    mock_download_data.assert_called_once()
    mock_load_data.assert_called_once()
    mock_save_model.assert_called_once()
    mock_save_conf_matrix.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()