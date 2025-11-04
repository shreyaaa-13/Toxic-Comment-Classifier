"""
Model Training Module for Toxic Comment Classifier
Trains Naive Bayes, Logistic Regression, and LSTM models
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import TextPreprocessor, load_and_prepare_data


class ToxicCommentClassifier:
    """
    Main classifier class that handles training of all models
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.nb_model = None
        self.lr_model = None
        self.lstm_model = None
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
    
    def train_naive_bayes(self, X_train_tfidf, y_train, X_test_tfidf, y_test):
        """
        Train Naive Bayes model
        
        Args:
            X_train_tfidf: Training features (TF-IDF)
            y_train: Training labels
            X_test_tfidf: Test features (TF-IDF)
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("Training Naive Bayes Model")
        print("="*60)
        
        # Train model
        self.nb_model = MultinomialNB(alpha=0.1)
        self.nb_model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred_train = self.nb_model.predict(X_train_tfidf)
        y_pred_test = self.nb_model.predict(X_test_tfidf)
        
        # Evaluate
        metrics = self._evaluate_model("Naive Bayes", y_train, y_pred_train, y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(self.models_dir, 'naive_bayes_model.pkl')
        joblib.dump(self.nb_model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        return metrics
    
    def train_logistic_regression(self, X_train_tfidf, y_train, X_test_tfidf, y_test):
        """
        Train Logistic Regression model
        
        Args:
            X_train_tfidf: Training features (TF-IDF)
            y_train: Training labels
            X_test_tfidf: Test features (TF-IDF)
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("Training Logistic Regression Model")
        print("="*60)
        
        # Train model
        self.lr_model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='liblinear',
            random_state=42
        )
        self.lr_model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred_train = self.lr_model.predict(X_train_tfidf)
        y_pred_test = self.lr_model.predict(X_test_tfidf)
        
        # Evaluate
        metrics = self._evaluate_model("Logistic Regression", y_train, y_pred_train, y_test, y_pred_test)
        
        # Save model
        model_path = os.path.join(self.models_dir, 'logistic_regression_model.pkl')
        joblib.dump(self.lr_model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        return metrics
    
    def build_lstm_model(self, vocab_size, max_len=100, embedding_dim=128):
        """
        Build LSTM model architecture
        
        Args:
            vocab_size (int): Vocabulary size
            max_len (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_lstm(self, X_train_seq, y_train, X_test_seq, y_test, vocab_size, max_len=100, epochs=10, batch_size=64):
        """
        Train LSTM model
        
        Args:
            X_train_seq: Training sequences
            y_train: Training labels
            X_test_seq: Test sequences
            y_test: Test labels
            vocab_size (int): Vocabulary size
            max_len (int): Maximum sequence length
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Evaluation metrics and training history
        """
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        
        # Build model
        self.lstm_model = self.build_lstm_model(vocab_size, max_len)
        
        print("\nModel Architecture:")
        self.lstm_model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1),
            ModelCheckpoint(
                os.path.join(self.models_dir, 'lstm_model_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\nTraining LSTM...")
        history = self.lstm_model.fit(
            X_train_seq, y_train,
            validation_data=(X_test_seq, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Predictions
        y_pred_train = (self.lstm_model.predict(X_train_seq, verbose=0) > 0.5).astype(int).flatten()
        y_pred_test = (self.lstm_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
        
        # Evaluate
        metrics = self._evaluate_model("LSTM", y_train, y_pred_train, y_test, y_pred_test)
        metrics['history'] = history.history
        
        # Save model
        model_path = os.path.join(self.models_dir, 'lstm_model.h5')
        self.lstm_model.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Plot training history
        self._plot_training_history(history)
        
        return metrics
    
    def _evaluate_model(self, model_name, y_train, y_pred_train, y_test, y_pred_test):
        """
        Evaluate model performance
        
        Args:
            model_name (str): Name of the model
            y_train: True training labels
            y_pred_train: Predicted training labels
            y_test: True test labels
            y_pred_test: Predicted test labels
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n📊 {model_name} - Training Set Performance:")
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, zero_division=0),
            'recall': recall_score(y_train, y_pred_train, zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, zero_division=0)
        }
        
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\n📊 {model_name} - Test Set Performance:")
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, zero_division=0)
        }
        
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Non-Toxic', 'Toxic']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\n🔢 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, model_name)
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Toxic', 'Toxic'],
                    yticklabels=['Non-Toxic', 'Toxic'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = os.path.join(self.models_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {plot_path}")
        plt.close()
    
    def _plot_training_history(self, history):
        """Plot LSTM training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, 'lstm_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved to {plot_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("="*60)
    print("TOXIC COMMENT CLASSIFIER - MODEL TRAINING")
    print("="*60)
    
    # Configuration
    DATA_PATH = 'data/train.csv'
    MODELS_DIR = 'models'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    MAX_WORDS = 10000
    MAX_LEN = 100
    LSTM_EPOCHS = 10
    BATCH_SIZE = 64
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠️  Dataset not found at {DATA_PATH}")
        print("Please either:")
        print("1. Download the Kaggle dataset and place it in data/train.csv")
        print("2. Run: python src/generate_sample_data.py")
        return
    
    # Load data
    X_text, y_labels, df = load_and_prepare_data(DATA_PATH)
    
    # Initialize classifier
    classifier = ToxicCommentClassifier(models_dir=MODELS_DIR)
    
    # Preprocess data
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    df = classifier.preprocessor.preprocess_dataframe(df)
    X_processed = df['processed_text'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Create TF-IDF features for Naive Bayes and Logistic Regression
    print("\n🔧 Creating TF-IDF features...")
    X_train_tfidf = classifier.preprocessor.create_tfidf_features(X_train, max_features=MAX_FEATURES, fit=True)
    X_test_tfidf = classifier.preprocessor.create_tfidf_features(X_test, fit=False)
    
    # Save TF-IDF vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    joblib.dump(classifier.preprocessor.tfidf_vectorizer, vectorizer_path)
    print(f"✅ TF-IDF vectorizer saved to {vectorizer_path}")
    
    # Train Naive Bayes
    nb_metrics = classifier.train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Train Logistic Regression
    lr_metrics = classifier.train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Create sequences for LSTM
    print("\n🔧 Creating sequences for LSTM...")
    X_train_seq = classifier.preprocessor.create_lstm_sequences(X_train, max_words=MAX_WORDS, max_len=MAX_LEN, fit=True)
    X_test_seq = classifier.preprocessor.create_lstm_sequences(X_test, max_len=MAX_LEN, fit=False)
    vocab_size = classifier.preprocessor.get_vocab_size()
    print(f"  Vocabulary size: {vocab_size}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(MODELS_DIR, 'lstm_tokenizer.pkl')
    joblib.dump(classifier.preprocessor.tokenizer, tokenizer_path)
    print(f"✅ LSTM tokenizer saved to {tokenizer_path}")
    
    # Train LSTM
    lstm_metrics = classifier.train_lstm(
        X_train_seq, y_train, X_test_seq, y_test,
        vocab_size=vocab_size, max_len=MAX_LEN,
        epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression', 'LSTM'],
        'Accuracy': [
            nb_metrics['test']['accuracy'],
            lr_metrics['test']['accuracy'],
            lstm_metrics['test']['accuracy']
        ],
        'Precision': [
            nb_metrics['test']['precision'],
            lr_metrics['test']['precision'],
            lstm_metrics['test']['precision']
        ],
        'Recall': [
            nb_metrics['test']['recall'],
            lr_metrics['test']['recall'],
            lstm_metrics['test']['recall']
        ],
        'F1-Score': [
            nb_metrics['test']['f1_score'],
            lr_metrics['test']['f1_score'],
            lstm_metrics['test']['f1_score']
        ]
    })
    
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Model comparison saved to {comparison_path}")
    
    # Plot comparison
    plot_model_comparison(comparison_df, MODELS_DIR)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAll models and artifacts saved to: {MODELS_DIR}/")


def plot_model_comparison(comparison_df, models_dir):
    """Plot model comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, comparison_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(models_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison chart saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
