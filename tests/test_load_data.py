import os
import gzip
import json
import pytest
from unittest.mock import patch
from src.data.load_data import DataLoader

# Mock constants
TEST_DATA_URL = "https://example.com/dataset.jsonl.gz"
TEST_LOCAL_PATH = "test_data_files/"
TEST_GZIP_FILE = os.path.join(TEST_LOCAL_PATH, "test_dataset.jsonl.gz")
TEST_JSONL_CONTENT = [
    {"id": 1, "name": "Product 1"},
    {"id": 2, "name": "Product 2"}
]

@pytest.fixture
def setup_test_directory():
    """
    Fixture to set up a test directory and clean it up after tests.
    """
    os.makedirs(TEST_LOCAL_PATH, exist_ok=True)
    yield
    for root, dirs, files in os.walk(TEST_LOCAL_PATH, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(TEST_LOCAL_PATH)

def create_test_gzip_file(file_path, content):
    """
    Helper function to create a test gzip file with JSONL content.
    """
    with gzip.open(file_path, "wt", encoding="utf-8") as gz_file:
        for record in content:
            gz_file.write(json.dumps(record) + "\n")

def test_download_data(setup_test_directory):
    """
    Test the download_data method of the DataLoader class.
    """
    data_loader = DataLoader(data_url=TEST_DATA_URL, local_data_path=TEST_LOCAL_PATH)
    with patch("src.data.load_data.gdown.download") as mock_download:
        mock_download.return_value = TEST_GZIP_FILE
        # Call the method being tested
        data_loader.download_data()
        mock_download.assert_called_once_with(
            TEST_DATA_URL,
            os.path.join(TEST_LOCAL_PATH, "dataset.jsonl.gz"),
            quiet=False
        )
        # Simulate the file creation
        with open(os.path.join(TEST_LOCAL_PATH, "dataset.jsonl.gz"), "w") as f:
            f.write("dummy content")
        # Assert the file exists
        assert os.path.exists(os.path.join(TEST_LOCAL_PATH, "dataset.jsonl.gz"))

def test_parse(setup_test_directory):
    """
    Test the parse method of the DataLoader class.
    """
    create_test_gzip_file(TEST_GZIP_FILE, TEST_JSONL_CONTENT)
    parsed_data = list(DataLoader.parse(TEST_GZIP_FILE))
    assert len(parsed_data) == len(TEST_JSONL_CONTENT)
    assert parsed_data == TEST_JSONL_CONTENT

def test_load_data(setup_test_directory):
    """
    Test the load_data method of the DataLoader class.
    """
    data_loader = DataLoader(local_data_path=TEST_LOCAL_PATH)
    create_test_gzip_file(TEST_GZIP_FILE, TEST_JSONL_CONTENT)
    loaded_data = data_loader.load_data()
    assert len(loaded_data) == len(TEST_JSONL_CONTENT)
    assert loaded_data == TEST_JSONL_CONTENT

def test_load_data_no_directory():
    """
    Test load_data when the directory does not exist.
    """
    data_loader = DataLoader(local_data_path="non_existent_directory/")
    with pytest.raises(FileNotFoundError, match="The directory .* does not exist. Please download the dataset first."):
        data_loader.load_data()

def test_load_data_no_gzip_files(setup_test_directory):
    """
    Test load_data when no gzip files are found in the directory.
    """
    data_loader = DataLoader(local_data_path=TEST_LOCAL_PATH)
    with pytest.raises(FileNotFoundError, match="No gzip files found in the directory .*"):
        data_loader.load_data()