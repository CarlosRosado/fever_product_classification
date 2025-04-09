import pytest
import pandas as pd
from src.data.preprocess_data import DataPreprocessor

@pytest.fixture
def raw_data():
    """
    Fixture to provide sample raw data for testing.
    """
    return [
        {
            "also_buy": ["B001", "B002"],
            "also_view": ["B003", "B004", "B005"],
            "asin": "B000123",
            "brand": "Example Brand",
            "category": ["Electronics", "Accessories"],
            "description": ["This is a sample product description."],
            "feature": ["Feature 1", "Feature 2"],
            "image": ["img1.jpg", "img2.jpg"],
            "price": "$19.99",
            "title": "Sample Product Title",
            "main_cat": "Electronics"
        },
        {
            "also_buy": [],
            "also_view": [],
            "asin": "B000456",
            "brand": "Another Brand",
            "category": ["Home", "Kitchen"],
            "description": ["Another product description."],
            "feature": ["Feature A"],
            "image": ["img3.jpg"],
            "price": "$29.99",
            "title": "Another Product",
            "main_cat": "Home"
        }
    ]

@pytest.fixture
def preprocessor():
    """
    Fixture to provide an instance of the DataPreprocessor class.
    """
    return DataPreprocessor()

def test_clean_text(preprocessor):
    """
    Test the clean_text method of the DataPreprocessor class.
    """
    text = "This is a SAMPLE text with SPECIAL characters! 123"
    cleaned_text = preprocessor.clean_text(text)
    assert cleaned_text == "sample text special character"

def test_preprocess_data(preprocessor, raw_data):
    """
    Test the preprocess_data method of the DataPreprocessor class.
    """
    processed_data = preprocessor.preprocess_data(raw_data)
    assert len(processed_data) == 2
    assert processed_data[0]["also_buy_count"] == 2
    assert processed_data[0]["also_view_count"] == 3
    assert processed_data[0]["price"] == 19.99
    assert processed_data[0]["brand"] == "example brand"
    assert processed_data[0]["description"] == "sample product description"
    assert processed_data[0]["feature"] == "feature feature"
    assert processed_data[0]["image_count"] == 2
    assert processed_data[0]["title"] == "sample product title"

def test_encode_features(preprocessor):
    """
    Test the encode_features method of the DataPreprocessor class.
    """
    data = pd.DataFrame({
        "brand": ["Brand A", "Brand B", "Brand A"],
        "category": ["Electronics", "Home", "Electronics"],
        "price": [19.99, 29.99, 19.99]
    })
    encoded_data = preprocessor.encode_features(data)
    assert "brand" in encoded_data.columns
    assert "category" in encoded_data.columns
    assert encoded_data["brand"].dtype == "int64"
    assert encoded_data["category"].dtype == "int64"
    assert encoded_data["price"].dtype == "float64"