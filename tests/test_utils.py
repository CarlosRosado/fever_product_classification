import os
import pytest
import joblib
from unittest.mock import patch
from src.utils.utils import ModelUtils

# Mock constants
TEST_MODEL = {"key": "value"}  # Example model object
TEST_MODEL_PATH = "test_models/test_model.pkl"
TEST_DIRECTORY = "test_models/"

@pytest.fixture
def setup_test_directory():
    """
    Fixture to set up a test directory and clean it up after tests.
    """
    os.makedirs(TEST_DIRECTORY, exist_ok=True)
    yield
    for root, dirs, files in os.walk(TEST_DIRECTORY, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(TEST_DIRECTORY)

def test_save_model(setup_test_directory):
    """
    Test the save_model method of the ModelUtils class.
    """
    ModelUtils.save_model(TEST_MODEL, TEST_MODEL_PATH)
    assert os.path.exists(TEST_MODEL_PATH), "Model file was not saved."

    # Verify the saved model content
    loaded_model = joblib.load(TEST_MODEL_PATH)
    assert loaded_model == TEST_MODEL, "Saved model content does not match the original."

def test_load_model(setup_test_directory):
    """
    Test the load_model method of the ModelUtils class.
    """
    # Save a test model first
    joblib.dump(TEST_MODEL, TEST_MODEL_PATH)

    # Load the model using the method
    loaded_model = ModelUtils.load_model(TEST_MODEL_PATH)
    assert loaded_model == TEST_MODEL, "Loaded model content does not match the original."

def test_ensure_directory_exists(setup_test_directory):
    """
    Test the ensure_directory_exists method of the ModelUtils class.
    """
    test_dir = os.path.join(TEST_DIRECTORY, "new_directory")
    ModelUtils.ensure_directory_exists(test_dir)
    assert os.path.exists(test_dir), "Directory was not created."

    # Test when the directory already exists
    with patch("os.makedirs") as mock_makedirs:
        ModelUtils.ensure_directory_exists(test_dir)
        mock_makedirs.assert_not_called()