import logging
import gzip
import json
import os
import gdown
from glob import glob

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataLoader:
    """
    A class to handle downloading, parsing, and loading data from gzip files.
    """

    def __init__(self, data_url="https://drive.google.com/uc?id=1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3", local_data_path="data_files/"):
        """
        Initialize the DataLoader with default or custom configurations.

        Args:
            data_url (str): The URL to download the dataset from
            local_data_path (str): The local directory where the dataset will be saved.
        """
        self.data_url = data_url
        self.local_data_path = local_data_path

    def download_data(self):
        """
        Download the dataset from the configured URL and save it to the local directory.

        Raises:
            Exception: If there is an error during the download process.

        Returns:
            None
        """
        try:
            os.makedirs(self.local_data_path, exist_ok=True)
            file_name = os.path.join(self.local_data_path, "dataset.jsonl.gz")
            if not os.path.exists(file_name):
                logging.info(f"Downloading dataset from {self.data_url}...")
                gdown.download(self.data_url, file_name, quiet=False)
                logging.info(f"Dataset downloaded and saved to {file_name}.")
            else:
                logging.info(f"Dataset already exists at {file_name}.")
        except Exception as e:
            logging.error(f"Error downloading dataset: {e}")
            raise

    @staticmethod
    def parse(path):
        """
        Read data from a gzip file and yield JSON objects line by line.

        Args:
            path (str): The path to the gzip file to be parsed.

        Raises:
            Exception: If there is an error while reading or parsing the file.

        Yields:
            dict: A JSON object parsed from each line of the gzip file.
        """
        try:
            with gzip.open(path, 'r') as g:
                for line in g:
                    yield json.loads(line)
        except Exception as e:
            logging.error(f"Error parsing file {path}: {e}")
            raise

    def load_data(self):
        """
        Load data from all gzip files in the configured local directory.

        Raises:
            FileNotFoundError: If the directory does not exist or no gzip files are found.
            Exception: If there is an error during the data loading process.

        Returns:
            list: A list of JSON objects loaded from all gzip files in the directory.
        """
        try:
            if not os.path.exists(self.local_data_path):
                raise FileNotFoundError(f"The directory {self.local_data_path} does not exist. Please download the dataset first.")
            
            all_data = []
            gz_files = glob(os.path.join(self.local_data_path, "*.gz"))
            if not gz_files:
                raise FileNotFoundError(f"No gzip files found in the directory {self.local_data_path}.")
            
            logging.info(f"Found {len(gz_files)} gzip file(s) in {self.local_data_path}.")
            for file_path in gz_files:
                logging.info(f"Loading data from {file_path}...")
                all_data.extend(self.parse(file_path))
            
            logging.info(f"Data loading completed. Total records loaded: {len(all_data)}")
            return all_data
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise