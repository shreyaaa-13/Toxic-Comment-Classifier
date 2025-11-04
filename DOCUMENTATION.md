# Toxic Comment Classifier - Complete Documentation

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [API Reference](#api-reference)
6. [Model Details](#model-details)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Contributing](#contributing)

---

## 📋 Project Overview

### Purpose
The Toxic Comment Classifier is an NLP-based system designed to automatically detect toxic comments in online platforms. It uses a combination of traditional machine learning and deep learning approaches to classify text as toxic or non-toxic.

### Key Features
- **Multiple Models**: Naive Bayes, Logistic Regression, and LSTM
- **Real-time Prediction**: Interactive web interface
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Easy to Use**: Simple API and web interface
- **Extensible**: Modular design for easy customization

### Technology Stack
- **Language**: Python 3.8+
- **ML Libraries**: Scikit-learn, TensorFlow/Keras
- **NLP**: NLTK
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Plotly, WordCloud

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface                        │
│              (Streamlit Web App)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Prediction Module                          │
│          (ToxicCommentPredictor)                        │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Naive   │  │ Logistic │  │   LSTM   │
│  Bayes   │  │Regression│  │  Model   │
└──────────┘  └──────────┘  └──────────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Preprocessing Module                          │
│          (TextPreprocessor)                             │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: User enters comment text
2. **Preprocessing**: Text is cleaned, tokenized, and normalized
3. **Feature Extraction**: 
   - TF-IDF vectors for NB/LR
   - Sequences for LSTM
4. **Prediction**: Models make predictions
5. **Output**: Results displayed with confidence scores

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB+ free disk space
- 4GB+ RAM (8GB recommended for LSTM training)

### Step-by-Step Installation

#### 1. Clone/Download Project
```bash
cd "c:\xampp\htdocs\Toxic Comment Classifier"
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download NLTK Data
```bash
python -c "import nltk; nltk.download(['stopwords', 'wordnet', 'punkt', 'omw-1.4'])"
```

#### 4. Automated Setup (Alternative)
```bash
python setup.py
```

#### 5. Verify Installation
```bash
python test_system.py
```

---

## 📖 Usage Guide

### Quick Start

#### 1. Generate Sample Data
```bash
python src/generate_sample_data.py
```

#### 2. Train Models
```bash
python src/train_models.py
```

Or use the batch file (Windows):
```bash
train_models.bat
```

#### 3. Run Web Application
```bash
streamlit run app.py
```

Or use the batch file (Windows):
```bash
run_app.bat
```

### Using the Web Interface

1. **Enter Comment**: Type or paste text in the input box
2. **Select Model**: Choose a specific model or "All Models"
3. **Analyze**: Click the "Analyze Comment" button
4. **View Results**: See prediction with confidence score

### Command Line Usage

#### Test Predictions
```bash
python src/predict.py
```

#### Evaluate Models
```bash
python src/evaluate_models.py
```

### Python API Usage

```python
from src.predict import ToxicCommentPredictor

# Initialize predictor
predictor = ToxicCommentPredictor()
predictor.load_all_models()

# Make prediction
text = "Your comment here"
results = predictor.predict_all(text)

# Access results
for model, result in results.items():
    print(f"{model}: {result['label']} ({result['probability']:.2%})")
```

---

## 🔧 API Reference

### TextPreprocessor Class

#### Methods

**`__init__()`**
- Initializes the preprocessor with stopwords and lemmatizer

**`clean_text(text: str) -> str`**
- Cleans and normalizes text
- Removes URLs, emails, mentions, hashtags, HTML tags
- Converts to lowercase
- Removes punctuation and extra whitespace

**`preprocess_text(text: str) -> str`**
- Complete preprocessing pipeline
- Returns cleaned, tokenized, and lemmatized text

**`create_tfidf_features(texts: list, max_features: int, fit: bool) -> array`**
- Creates TF-IDF feature matrix
- Parameters:
  - `texts`: List of preprocessed texts
  - `max_features`: Maximum number of features (default: 5000)
  - `fit`: Whether to fit the vectorizer (True for training)

**`create_lstm_sequences(texts: list, max_words: int, max_len: int, fit: bool) -> array`**
- Creates padded sequences for LSTM
- Parameters:
  - `texts`: List of preprocessed texts
  - `max_words`: Maximum vocabulary size (default: 10000)
  - `max_len`: Maximum sequence length (default: 100)
  - `fit`: Whether to fit the tokenizer

### ToxicCommentPredictor Class

#### Methods

**`__init__(models_dir: str)`**
- Initializes predictor
- Parameters:
  - `models_dir`: Directory containing trained models

**`load_all_models() -> bool`**
- Loads all available models
- Returns: True if at least one model loaded

**`predict_naive_bayes(text: str) -> tuple`**
- Predicts using Naive Bayes
- Returns: (prediction, probability)

**`predict_logistic_regression(text: str) -> tuple`**
- Predicts using Logistic Regression
- Returns: (prediction, probability)

**`predict_lstm(text: str, max_len: int) -> tuple`**
- Predicts using LSTM
- Returns: (prediction, probability)

**`predict_all(text: str) -> dict`**
- Predicts using all available models
- Returns: Dictionary with results from each model

**`predict_with_model(text: str, model_name: str) -> tuple`**
- Predicts using a specific model
- Parameters:
  - `text`: Input text
  - `model_name`: 'Naive Bayes', 'Logistic Regression', or 'LSTM'
- Returns: (prediction, probability)

---

## 🤖 Model Details

### 1. Naive Bayes

**Type**: Multinomial Naive Bayes

**Features**:
- TF-IDF vectors (max 5000 features)
- Unigrams and bigrams
- Alpha smoothing: 0.1

**Advantages**:
- Fast training and prediction
- Good baseline performance
- Low memory footprint

**Use Cases**:
- Quick prototyping
- Resource-constrained environments
- Real-time applications

### 2. Logistic Regression

**Type**: Binary Logistic Regression

**Features**:
- TF-IDF vectors (max 5000 features)
- Unigrams and bigrams
- L2 regularization (C=1.0)

**Advantages**:
- Interpretable coefficients
- Balanced performance
- Feature importance analysis

**Use Cases**:
- When interpretability is important
- Feature analysis
- Production systems

### 3. LSTM (Long Short-Term Memory)

**Architecture**:
```
Embedding Layer (128 dimensions)
    ↓
Spatial Dropout (0.2)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (1 unit, Sigmoid)
```

**Features**:
- Word embeddings (128 dimensions)
- Bidirectional processing
- Dropout for regularization
- Early stopping

**Advantages**:
- Captures context and sequences
- Best performance
- Handles complex patterns

**Use Cases**:
- When accuracy is critical
- Sufficient computational resources
- Large datasets

---

## ⚙️ Configuration

### config.py Settings

#### Data Settings
```python
TEST_SIZE = 0.2              # Test set proportion
RANDOM_STATE = 42            # Random seed
```

#### Preprocessing
```python
MAX_FEATURES_TFIDF = 5000    # TF-IDF features
MAX_WORDS_LSTM = 10000       # LSTM vocabulary
MAX_SEQUENCE_LENGTH = 100    # LSTM sequence length
```

#### Training
```python
LSTM_EPOCHS = 10             # Training epochs
BATCH_SIZE = 64              # Batch size
EARLY_STOPPING_PATIENCE = 3  # Early stopping
```

### Customization

To modify model parameters, edit `config.py`:

```python
# Example: Increase LSTM complexity
LSTM_CONFIG = {
    'embedding_dim': 256,      # Increased from 128
    'lstm_units_1': 128,       # Increased from 64
    'lstm_units_2': 64,        # Increased from 32
    'dense_units': 128,        # Increased from 64
    'dropout_rate': 0.4,       # Increased from 0.3
}
```

---

## 🔍 Troubleshooting

### Common Issues

#### Issue 1: Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue 2: NLTK Data Not Found
**Symptom**: `LookupError: Resource xxx not found`

**Solution**:
```python
import nltk
nltk.download('all')
```

#### Issue 3: Out of Memory (LSTM Training)
**Symptom**: `ResourceExhaustedError` or system freeze

**Solutions**:
- Reduce batch size in `config.py`
- Use smaller dataset
- Reduce LSTM complexity
- Close other applications

#### Issue 4: Models Not Found
**Symptom**: "No models found" in web app

**Solution**:
```bash
python src/train_models.py
```

#### Issue 5: Slow Predictions
**Solutions**:
- Use Naive Bayes or Logistic Regression
- Reduce MAX_SEQUENCE_LENGTH
- Use GPU for LSTM (if available)

---

## 🚀 Performance Optimization

### Training Optimization

1. **Use GPU for LSTM**:
```python
# Install TensorFlow GPU version
pip install tensorflow-gpu==2.13.0
```

2. **Increase Batch Size**:
```python
BATCH_SIZE = 128  # If you have enough RAM
```

3. **Use Data Generators** (for large datasets):
```python
# Implement in train_models.py
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
```

### Prediction Optimization

1. **Batch Predictions**:
```python
# Predict multiple comments at once
predictions = model.predict(X_batch)
```

2. **Model Caching**:
```python
# Use @st.cache_resource in Streamlit
@st.cache_resource
def load_models():
    return ToxicCommentPredictor()
```

3. **Use Lighter Models**:
- Naive Bayes: ~10ms per prediction
- Logistic Regression: ~15ms per prediction
- LSTM: ~50ms per prediction

---

## 📊 Evaluation Metrics

### Metrics Explained

**Accuracy**: Overall correctness
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Good for balanced datasets

**Precision**: Toxic prediction accuracy
- Formula: TP / (TP + FP)
- Important when false positives are costly

**Recall**: Toxic detection rate
- Formula: TP / (TP + FN)
- Important when false negatives are costly

**F1-Score**: Harmonic mean of precision and recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Balanced metric

**ROC-AUC**: Area under ROC curve
- Measures model's ability to distinguish classes
- Range: 0.5 (random) to 1.0 (perfect)

---

## 🎯 Best Practices

### Data Preparation
1. Use balanced datasets when possible
2. Clean data thoroughly
3. Remove duplicates
4. Handle missing values

### Model Training
1. Always use train/test split
2. Monitor validation metrics
3. Use early stopping
4. Save best models

### Production Deployment
1. Version your models
2. Log predictions
3. Monitor performance
4. Implement feedback loop
5. Regular retraining

---

## 🔄 Future Enhancements

### Planned Features
1. **Multi-label Classification**
   - Classify by toxicity type (obscene, threat, insult, etc.)

2. **Transformer Models**
   - Implement BERT/DistilBERT
   - Better context understanding

3. **API Endpoint**
   - REST API for integration
   - Authentication and rate limiting

4. **Real-time Monitoring**
   - Dashboard for model performance
   - Alert system for drift detection

5. **Explainability**
   - LIME/SHAP integration
   - Highlight toxic words

---

## 📝 License

This project is for educational purposes. Feel free to use and modify.

---

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review QUICKSTART.md
3. Run test_system.py
4. Check error logs

---

## 🙏 Acknowledgments

- Kaggle for the Toxic Comment Classification dataset
- TensorFlow and Scikit-learn teams
- NLTK contributors
- Streamlit team

---

**Last Updated**: 2024
**Version**: 1.0.0
