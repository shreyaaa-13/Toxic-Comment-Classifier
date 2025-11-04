"""
Data Preprocessing Module for Toxic Comment Classifier
Handles text cleaning, tokenization, and feature extraction
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextPreprocessor:
    """
    Comprehensive text preprocessing for toxic comment classification
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.tokenizer = None
        
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of tokens
        """
        tokens = nltk.word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column='comment_text'):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        print(f"Preprocessing {len(df)} comments...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        print("Preprocessing complete!")
        return df
    
    def create_tfidf_features(self, texts, max_features=5000, fit=True):
        """
        Create TF-IDF features for traditional ML models
        
        Args:
            texts (list): List of preprocessed texts
            max_features (int): Maximum number of features
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            array: TF-IDF feature matrix
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,
                max_df=0.95
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted yet!")
            features = self.tfidf_vectorizer.transform(texts)
        
        return features
    
    def create_lstm_sequences(self, texts, max_words=10000, max_len=100, fit=True):
        """
        Create sequences for LSTM model
        
        Args:
            texts (list): List of preprocessed texts
            max_words (int): Maximum vocabulary size
            max_len (int): Maximum sequence length
            fit (bool): Whether to fit the tokenizer
            
        Returns:
            array: Padded sequences
        """
        if fit:
            self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted yet!")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        return padded_sequences
    
    def get_vocab_size(self):
        """Get vocabulary size from tokenizer"""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer.word_index) + 1


def load_and_prepare_data(file_path, sample_size=None, binary_classification=True):
    """
    Load and prepare dataset for training
    
    Args:
        file_path (str): Path to CSV file
        sample_size (int): Number of samples to use (None for all)
        binary_classification (bool): Whether to use binary classification
        
    Returns:
        tuple: (X_text, y_labels, df)
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} samples")
    
    # For Kaggle dataset, create binary label
    if binary_classification:
        toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Check if columns exist
        if all(col in df.columns for col in toxic_columns):
            df['is_toxic'] = (df[toxic_columns].sum(axis=1) > 0).astype(int)
        elif 'is_toxic' not in df.columns:
            raise ValueError("Dataset must have either toxic label columns or 'is_toxic' column")
    
    X_text = df['comment_text'].values
    y_labels = df['is_toxic'].values
    
    print(f"Toxic comments: {y_labels.sum()} ({y_labels.sum()/len(y_labels)*100:.2f}%)")
    print(f"Non-toxic comments: {len(y_labels) - y_labels.sum()} ({(len(y_labels) - y_labels.sum())/len(y_labels)*100:.2f}%)")
    
    return X_text, y_labels, df


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    test_text = "This is a TEST comment with URLs http://example.com and @mentions #hashtags!!!"
    processed = preprocessor.preprocess_text(test_text)
    
    print("Original:", test_text)
    print("Processed:", processed)
