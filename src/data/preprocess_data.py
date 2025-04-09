import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataPreprocessor:
    """
    A class to handle data preprocessing: text cleaning, feature extraction, 
    and encoding categorical features.
    """

    def __init__(self):
        """
        Initialize the DataPreprocessor
        """
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Clean and preprocess text data

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned and lemmatized text.

        Raises:
            Exception: If there is an error during text cleaning.
        """
        try:
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)  
            tokens = [word for word in text.split() if word not in self.stopwords]
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            raise

    def preprocess_data(self, raw_data):
        """
        Preprocess raw data to extract and clean the required fields.

        Args:
            raw_data (list): A list of dictionaries containing raw data.

        Returns:
            list: A list of dictionaries with processed data.

        Raises:
            Exception: If there is an error during data preprocessing.
        """
        try:
            logging.info("Starting data preprocessing...")
            processed_data = []
            for item in raw_data:
                try:
                    # Clean and convert the price field
                    price = float(item['price'].replace('$', '').replace(',', '')) if item.get('price') else 0.0
                except (ValueError, AttributeError):
                    price = 0.0

                processed_item = {
                    'also_buy_count': len(item.get('also_buy', [])),
                    'also_view_count': len(item.get('also_view', [])),
                    'asin': item.get('asin', ''),
                    'brand': self.clean_text(item.get('brand', '')),
                    'category': item.get('category', []),
                    'description': self.clean_text(' '.join(item.get('description', []))) if item.get('description') else '',
                    'feature': self.clean_text(' '.join(item.get('feature', []))) if item.get('feature') else '',
                    'image_count': len(item.get('image', [])),
                    'price': price,
                    'title': self.clean_text(item.get('title', '')),
                    'main_cat': item.get('main_cat', '')
                }
                processed_data.append(processed_item)
            logging.info(f"Data preprocessing completed. Total records processed: {len(processed_data)}")
            return processed_data
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def encode_features(self, data):
        """
        Convert non-numeric columns in the DataFrame to numeric using LabelEncoder.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to be encoded.

        Returns:
            pd.DataFrame: The DataFrame with encoded features.

        Raises:
            Exception: If there is an error during feature encoding.
        """
        try:
            for column in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
            return data
        except Exception as e:
            logging.error(f"Error encoding features: {e}")
            raise
