"""
Configuration file for Toxic Comment Classifier
Centralized settings for all modules
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Data settings
DATA_FILE = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SIZE = None  # None for full dataset, or specify number
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing settings
MAX_FEATURES_TFIDF = 5000  # Maximum features for TF-IDF
MAX_WORDS_LSTM = 10000     # Maximum vocabulary size for LSTM
MAX_SEQUENCE_LENGTH = 100   # Maximum sequence length for LSTM
NGRAM_RANGE = (1, 2)       # N-gram range for TF-IDF

# Model settings
NAIVE_BAYES_ALPHA = 0.1

LOGISTIC_REGRESSION_CONFIG = {
    'max_iter': 1000,
    'C': 1.0,
    'solver': 'liblinear',
    'random_state': RANDOM_STATE
}

LSTM_CONFIG = {
    'embedding_dim': 128,
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units': 64,
    'dropout_rate': 0.3,
    'spatial_dropout': 0.2
}

# Training settings
LSTM_EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.0  # Use test set for validation

# Early stopping
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
MIN_LR = 0.00001

# Model file names
MODEL_FILES = {
    'naive_bayes': 'naive_bayes_model.pkl',
    'logistic_regression': 'logistic_regression_model.pkl',
    'lstm': 'lstm_model.h5',
    'lstm_best': 'lstm_model_best.h5',
    'tfidf_vectorizer': 'tfidf_vectorizer.pkl',
    'lstm_tokenizer': 'lstm_tokenizer.pkl'
}

# Visualization settings
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Web app settings
STREAMLIT_CONFIG = {
    'page_title': 'Toxic Comment Classifier',
    'page_icon': '🛡️',
    'layout': 'wide'
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Sample data generation
SAMPLE_DATA_CONFIG = {
    'n_samples': 2000,
    'toxic_ratio': 0.4,  # 40% toxic, 60% non-toxic
    'random_state': RANDOM_STATE
}

# Evaluation settings
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc'
]

# Word cloud settings
WORDCLOUD_CONFIG = {
    'width': 800,
    'height': 400,
    'background_color': 'white',
    'max_words': 100
}

# Model comparison
COMPARISON_MODELS = [
    'Naive Bayes',
    'Logistic Regression',
    'LSTM'
]


def get_model_path(model_name):
    """Get full path for a model file"""
    if model_name in MODEL_FILES:
        return os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)


if __name__ == "__main__":
    # Display configuration
    print("="*60)
    print("TOXIC COMMENT CLASSIFIER - CONFIGURATION")
    print("="*60)
    
    print(f"\n📁 Paths:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")
    
    print(f"\n⚙️ Data Settings:")
    print(f"  Test Size: {TEST_SIZE}")
    print(f"  Random State: {RANDOM_STATE}")
    
    print(f"\n🔧 Preprocessing:")
    print(f"  TF-IDF Max Features: {MAX_FEATURES_TFIDF}")
    print(f"  LSTM Max Words: {MAX_WORDS_LSTM}")
    print(f"  Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
    
    print(f"\n🤖 Training:")
    print(f"  LSTM Epochs: {LSTM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    print(f"\n📊 Models:")
    for model in COMPARISON_MODELS:
        print(f"  - {model}")
    
    ensure_directories()
    print(f"\n✅ Directories verified")
