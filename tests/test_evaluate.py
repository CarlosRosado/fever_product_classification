import pytest
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.model.evaluate import ModelEvaluator

@pytest.fixture
def mock_training_data():
    """
    Fixture to provide mock training data.
    """
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_train = [0, 1, 0, 1]
    return X_train, y_train

@pytest.fixture
def mock_model():
    """
    Fixture to provide a mock model.
    """
    model = LogisticRegression()
    model.fit([[1, 2], [2, 3], [3, 4], [4, 5]], [0, 1, 0, 1])  # Train the model
    return model

def test_evaluate_training(mock_model, mock_training_data):
    """
    Test the evaluate_training method of the ModelEvaluator class.
    """
    X_train, y_train = mock_training_data

    with patch("src.model.evaluate.logging.info") as mock_logging_info:
        ModelEvaluator.evaluate_training(mock_model, X_train, y_train)

        # Check if logging.info was called with accuracy
        accuracy = accuracy_score(y_train, mock_model.predict(X_train))
        mock_logging_info.assert_any_call("Evaluating the model on the training dataset...")
        mock_logging_info.assert_any_call(f"Training Accuracy: {accuracy:.4f}")

        # Check if classification report and confusion matrix were logged
        classification_report_str = classification_report(y_train, mock_model.predict(X_train))
        confusion_matrix_str = str(confusion_matrix(y_train, mock_model.predict(X_train)))
        mock_logging_info.assert_any_call("\nTraining Classification Report:")
        mock_logging_info.assert_any_call("\n" + classification_report_str)
        mock_logging_info.assert_any_call("\nTraining Confusion Matrix:")
        mock_logging_info.assert_any_call("\n" + confusion_matrix_str)

def test_evaluate_predictions():
    """
    Test the evaluate_predictions method of the ModelEvaluator class.
    """
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    with patch("src.model.evaluate.logging.info") as mock_logging_info:
        ModelEvaluator.evaluate_predictions(y_true, y_pred, dataset_name="Validation")

        # Check if logging.info was called with accuracy
        accuracy = accuracy_score(y_true, y_pred)
        mock_logging_info.assert_any_call("Evaluating predictions on the Validation dataset...")
        mock_logging_info.assert_any_call(f"Validation Accuracy: {accuracy:.4f}")

        # Check if classification report and confusion matrix were logged
        classification_report_str = classification_report(y_true, y_pred)
        confusion_matrix_str = str(confusion_matrix(y_true, y_pred))
        mock_logging_info.assert_any_call(f"\nValidation Classification Report:")
        mock_logging_info.assert_any_call("\n" + classification_report_str)
        mock_logging_info.assert_any_call(f"\nValidation Confusion Matrix:")
        mock_logging_info.assert_any_call("\n" + confusion_matrix_str)