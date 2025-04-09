import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from src.data.load_data import DataLoader
from src.data.preprocess_data import DataPreprocessor
from src.utils.utils import ModelUtils
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TrainingPipeline:
    """
    A class to handle the training pipeline for the Fever Product Classification model.
    """

    def __init__(self):
        """
        Initialize the TrainingPipeline with configuration from environment variables.
        """
        self.data_url = os.getenv("DATA_URL")
        self.local_data_path = os.getenv("LOCAL_DATA_PATH", "data_files/")
        self.model_output_path = os.getenv("MODEL_OUTPUT_PATH", "models/")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Initialize helper classes
        self.data_loader = DataLoader(data_url=self.data_url, local_data_path=self.local_data_path)
        self.data_preprocessor = DataPreprocessor()

    @staticmethod
    def save_confusion_matrix(cm, output_dir, file_name="confusion_matrix.png"):
        """
        Save the confusion matrix as an image file.

        Args:
            cm (ndarray): The confusion matrix to save.
            output_dir (str): The directory where the image will be saved.
            file_name (str): The name of the image file.

        Returns:
            str: The path to the saved confusion matrix image.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(output_dir, file_name)
        plt.savefig(cm_path)
        plt.close()
        return cm_path

    def run(self):
        """
        Execute the training pipeline with MLflow integration.

        This method handles:
        - Data downloading and preprocessing
        - Model training and evaluation
        - Logging metrics and artifacts to MLflow
        - Saving the trained model locally

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            # Start MLflow experiment
            mlflow.set_experiment("fever_product_classification")
            with mlflow.start_run(run_name="Fever Random Forest Training"):
                logging.info("Starting training pipeline...")

                # Ensure the data and model directories exist
                logging.info("Ensuring data and model directories exist...")
                ModelUtils.ensure_directory_exists(self.local_data_path)
                ModelUtils.ensure_directory_exists(self.model_output_path)

                # Ensure the dataset is downloaded
                logging.info("Downloading dataset...")
                self.data_loader.download_data()

                # Load all raw data from the folder
                logging.info("Loading data from all gzip files in the folder...")
                raw_data = self.data_loader.load_data()

                # Preprocess the data
                logging.info("Preprocessing the data...")
                processed_data = self.data_preprocessor.preprocess_data(raw_data)
                data = pd.DataFrame(processed_data)

                # Separate features (X) and target (y)
                logging.info("Separating features and target...")
                X = data.drop(columns=['main_cat'])
                y = data['main_cat']

                # Encode non-numeric features
                logging.info("Encoding non-numeric features...")
                X = self.data_preprocessor.encode_features(X)

                # Split the data into training and testing sets
                logging.info("Splitting the data into training and testing sets...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Apply SMOTE to balance the training dataset
                logging.info("Applying SMOTE to balance the training dataset...")
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logging.info(f"Training set size after SMOTE: {X_train.shape[0]} samples")

                # Initialize and train the Random Forest model
                logging.info("Initializing and training the Random Forest model...")
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    n_jobs=-1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Log parameters to MLflow
                logging.info("Logging parameters to MLflow...")
                mlflow.log_param("n_estimators", 50)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("min_samples_split", 10)
                mlflow.log_param("min_samples_leaf", 5)
                mlflow.log_param("max_features", "sqrt")
                mlflow.log_param("class_weight", "balanced")
                mlflow.log_param("random_state", 42)

                # Make predictions
                logging.info("Making predictions...")
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics
                logging.info("Calculating metrics...")
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                cm = confusion_matrix(y_test, y_pred)

                # Log metrics to MLflow
                logging.info("Logging metrics to MLflow...")
                metrics = {"accuracy": accuracy, "auc": auc}
                mlflow.log_metrics(metrics)

                # Save and log confusion matrix
                logging.info("Saving and logging confusion matrix...")
                cm_path = self.save_confusion_matrix(cm, self.model_output_path, file_name="random_forest_cm.png")
                mlflow.log_artifact(cm_path)

                # Log model to MLflow
                logging.info("Logging the model to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="fever_random_forest_model",
                    registered_model_name="Fever_Random_Forest",
                    input_example=X_test[:5]
                )

                # Save the model locally
                logging.info("Saving the model locally...")
                ModelUtils.save_model(model, os.path.join(self.model_output_path, "fever_random_forest_model.pkl"))

                logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()