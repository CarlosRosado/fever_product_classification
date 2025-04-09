import os
import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load configuration from .env
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH")


class ModelEvaluator:
    """
    A class to handle model evaluation tasks
    """

    @staticmethod
    def evaluate_training(model, X_train, y_train):
        """
        Evaluate the model on the training dataset.

        Args:
            model: The trained model to evaluate.
            X_train (pd.DataFrame or np.ndarray): The training features.
            y_train (pd.Series or np.ndarray): The true labels for the training dataset.

        Raises:
            Exception: If there is an error during training evaluation.

        Returns:
            None
        """
        try:
            logging.info("Evaluating the model on the training dataset...")
            y_pred = model.predict(X_train)

            # Calculate metrics
            accuracy = accuracy_score(y_train, y_pred)
            logging.info(f"Training Accuracy: {accuracy:.4f}")

            logging.info("\nTraining Classification Report:")
            logging.info("\n" + classification_report(y_train, y_pred))

            logging.info("\nTraining Confusion Matrix:")
            logging.info("\n" + str(confusion_matrix(y_train, y_pred)))
        except Exception as e:
            logging.error(f"Error during training evaluation: {e}")
            raise

    @staticmethod
    def evaluate_predictions(y_true, y_pred, dataset_name="Validation"):
        """
        Evaluate predictions made by the model.

        Args:
            y_true (pd.Series or np.ndarray): The true labels for the dataset.
            y_pred (pd.Series or np.ndarray): The predicted labels by the model.
            dataset_name (str): The name of the dataset being evaluated (default is "Validation").

        Raises:
            Exception: If there is an error during prediction evaluation.

        Returns:
            None
        """
        try:
            logging.info(f"Evaluating predictions on the {dataset_name} dataset...")

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"{dataset_name} Accuracy: {accuracy:.4f}")

            logging.info(f"\n{dataset_name} Classification Report:")
            logging.info("\n" + classification_report(y_true, y_pred))

            logging.info(f"\n{dataset_name} Confusion Matrix:")
            logging.info("\n" + str(confusion_matrix(y_true, y_pred)))
        except Exception as e:
            logging.error(f"Error during {dataset_name} evaluation: {e}")
            raise

