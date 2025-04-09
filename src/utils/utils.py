import logging
import os
import joblib

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelUtils:
    """
    A utils class for model-related operations
    """

    @staticmethod
    def save_model(model, model_output_path):
        """
        Save the trained model to the specified path.

        Args:
            model: The trained model to save.
            model_output_path (str): The file path where the model will be saved.

        Raises:
            Exception: If there is an error during the save process.

        Returns:
            None
        """
        try:
            ModelUtils.ensure_directory_exists(os.path.dirname(model_output_path))
            joblib.dump(model, model_output_path)
            logging.info(f"Model saved to {model_output_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    @staticmethod
    def load_model(filepath):
        """
        Load a model from the specified file path.

        Args:
            filepath (str): The file path from which to load the model.

        Raises:
            FileNotFoundError: If the model file is not found.
            Exception: If there is an error during the load process.

        Returns:
            model: The loaded model.
        """
        try:
            logging.info(f"Loading model from {filepath}...")
            model = joblib.load(filepath)
            logging.info("Model loaded successfully.")
            return model
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    @staticmethod
    def ensure_directory_exists(directory_path):
        """
        Ensure that a directory exists. If it doesn't exist, create it.

        Args:
            directory_path (str): The path of the directory to check or create.

        Raises:
            Exception: If there is an error during directory creation.

        Returns:
            None
        """
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                logging.info(f"Directory created: {directory_path}")
            else:
                logging.info(f"Directory already exists: {directory_path}")
        except Exception as e:
            logging.error(f"Error ensuring directory exists: {e}")
            raise
