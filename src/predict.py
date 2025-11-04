"""
Prediction Module for Toxic Comment Classifier
Handles loading models and making predictions
"""

import os
import joblib
import numpy as np
from tensorflow import keras
from preprocessing import TextPreprocessor


class ToxicCommentPredictor:
    """
    Predictor class for making toxicity predictions
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        
        # Model placeholders
        self.nb_model = None
        self.lr_model = None
        self.lstm_model = None
        self.tfidf_vectorizer = None
        self.lstm_tokenizer = None
        
        # Model loaded flags
        self.nb_loaded = False
        self.lr_loaded = False
        self.lstm_loaded = False
    
    def load_naive_bayes(self):
        """Load Naive Bayes model"""
        try:
            model_path = os.path.join(self.models_dir, 'naive_bayes_model.pkl')
            vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            
            self.nb_model = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(vectorizer_path)
            self.nb_loaded = True
            print("✅ Naive Bayes model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading Naive Bayes model: {e}")
            return False
    
    def load_logistic_regression(self):
        """Load Logistic Regression model"""
        try:
            model_path = os.path.join(self.models_dir, 'logistic_regression_model.pkl')
            vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            
            self.lr_model = joblib.load(model_path)
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = joblib.load(vectorizer_path)
            self.lr_loaded = True
            print("✅ Logistic Regression model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading Logistic Regression model: {e}")
            return False
    
    def load_lstm(self):
        """Load LSTM model"""
        try:
            model_path = os.path.join(self.models_dir, 'lstm_model.h5')
            tokenizer_path = os.path.join(self.models_dir, 'lstm_tokenizer.pkl')
            
            self.lstm_model = keras.models.load_model(model_path)
            self.lstm_tokenizer = joblib.load(tokenizer_path)
            self.lstm_loaded = True
            print("✅ LSTM model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            return False
    
    def load_all_models(self):
        """Load all models"""
        print("Loading all models...")
        self.load_naive_bayes()
        self.load_logistic_regression()
        self.load_lstm()
        
        loaded_count = sum([self.nb_loaded, self.lr_loaded, self.lstm_loaded])
        print(f"\n✅ {loaded_count}/3 models loaded successfully")
        
        return loaded_count > 0
    
    def predict_naive_bayes(self, text):
        """
        Predict using Naive Bayes
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (prediction, probability)
        """
        if not self.nb_loaded:
            raise ValueError("Naive Bayes model not loaded!")
        
        # Preprocess
        processed = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        features = self.tfidf_vectorizer.transform([processed])
        
        # Predict
        prediction = self.nb_model.predict(features)[0]
        probability = self.nb_model.predict_proba(features)[0]
        
        return prediction, probability[1]  # Return toxic probability
    
    def predict_logistic_regression(self, text):
        """
        Predict using Logistic Regression
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (prediction, probability)
        """
        if not self.lr_loaded:
            raise ValueError("Logistic Regression model not loaded!")
        
        # Preprocess
        processed = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        features = self.tfidf_vectorizer.transform([processed])
        
        # Predict
        prediction = self.lr_model.predict(features)[0]
        probability = self.lr_model.predict_proba(features)[0]
        
        return prediction, probability[1]  # Return toxic probability
    
    def predict_lstm(self, text, max_len=100):
        """
        Predict using LSTM
        
        Args:
            text (str): Input text
            max_len (int): Maximum sequence length
            
        Returns:
            tuple: (prediction, probability)
        """
        if not self.lstm_loaded:
            raise ValueError("LSTM model not loaded!")
        
        # Preprocess
        processed = self.preprocessor.preprocess_text(text)
        
        # Create sequence
        sequence = self.lstm_tokenizer.texts_to_sequences([processed])
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Predict
        probability = self.lstm_model.predict(padded, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def predict_all(self, text):
        """
        Predict using all available models
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Predictions from all models
        """
        results = {}
        
        if self.nb_loaded:
            try:
                pred, prob = self.predict_naive_bayes(text)
                results['Naive Bayes'] = {
                    'prediction': int(pred),
                    'probability': float(prob),
                    'label': 'Toxic' if pred == 1 else 'Non-Toxic'
                }
            except Exception as e:
                results['Naive Bayes'] = {'error': str(e)}
        
        if self.lr_loaded:
            try:
                pred, prob = self.predict_logistic_regression(text)
                results['Logistic Regression'] = {
                    'prediction': int(pred),
                    'probability': float(prob),
                    'label': 'Toxic' if pred == 1 else 'Non-Toxic'
                }
            except Exception as e:
                results['Logistic Regression'] = {'error': str(e)}
        
        if self.lstm_loaded:
            try:
                pred, prob = self.predict_lstm(text)
                results['LSTM'] = {
                    'prediction': int(pred),
                    'probability': float(prob),
                    'label': 'Toxic' if pred == 1 else 'Non-Toxic'
                }
            except Exception as e:
                results['LSTM'] = {'error': str(e)}
        
        return results
    
    def predict_with_model(self, text, model_name='LSTM'):
        """
        Predict using a specific model
        
        Args:
            text (str): Input text
            model_name (str): Model to use ('Naive Bayes', 'Logistic Regression', or 'LSTM')
            
        Returns:
            tuple: (prediction, probability)
        """
        if model_name == 'Naive Bayes':
            return self.predict_naive_bayes(text)
        elif model_name == 'Logistic Regression':
            return self.predict_logistic_regression(text)
        elif model_name == 'LSTM':
            return self.predict_lstm(text)
        else:
            raise ValueError(f"Unknown model: {model_name}")


def main():
    """Test prediction functionality"""
    # Initialize predictor
    predictor = ToxicCommentPredictor(models_dir='models')
    
    # Load models
    if not predictor.load_all_models():
        print("\n⚠️  No models found! Please train models first:")
        print("   python src/train_models.py")
        return
    
    # Test comments
    test_comments = [
        "This is a great article, thanks for sharing!",
        "You are an idiot and nobody likes you",
        "I really appreciate your perspective on this topic",
        "This is stupid and so are you",
        "Interesting point of view, I hadn't considered that"
    ]
    
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    for i, comment in enumerate(test_comments, 1):
        print(f"\n📝 Comment {i}: \"{comment}\"")
        print("-" * 60)
        
        results = predictor.predict_all(comment)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"  {model_name}: Error - {result['error']}")
            else:
                emoji = "🚫" if result['prediction'] == 1 else "✅"
                print(f"  {model_name}: {emoji} {result['label']} (confidence: {result['probability']:.2%})")


if __name__ == "__main__":
    main()
