# Toxic Comment Classifier using NLP

## 🎯 Project Overview
An advanced Natural Language Processing system that classifies online comments as toxic or non-toxic using machine learning and deep learning models.

## 🚀 Features
- **Multiple Models**: Naive Bayes, Logistic Regression, and LSTM
- **Real-time Prediction**: Interactive web interface using Streamlit
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Visualization**: Model comparison charts and word clouds

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. Clone or download this repository
2. Navigate to the project directory:
```bash
cd "Toxic Comment Classifier"
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (run in Python):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

## 📊 Dataset
This project uses the Kaggle "Toxic Comment Classification Challenge" dataset.

**Option 1**: Download from Kaggle
- Visit: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
- Download `train.csv` and place it in the `data/` folder

**Option 2**: Use the sample dataset generator (for testing)
```bash
python src/generate_sample_data.py
```

## 🏃‍♂️ Usage

### 1. Train Models
```bash
python src/train_models.py
```
This will:
- Preprocess the data
- Train all three models (Naive Bayes, Logistic Regression, LSTM)
- Save trained models to `models/` folder
- Generate evaluation reports

### 2. Run the Web Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### 3. Evaluate Models
```bash
python src/evaluate_models.py
```

## 📁 Project Structure
```
Toxic Comment Classifier/
│
├── data/                          # Dataset folder
│   └── train.csv                  # Training data
│
├── models/                        # Saved models
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── lstm_model.h5
│   └── tfidf_vectorizer.pkl
│
├── src/                           # Source code
│   ├── preprocessing.py           # Data preprocessing
│   ├── train_models.py            # Model training
│   ├── evaluate_models.py         # Model evaluation
│   ├── predict.py                 # Prediction utilities
│   └── generate_sample_data.py    # Sample data generator
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Models

### 1. Naive Bayes
- Baseline probabilistic classifier
- Fast training and prediction
- Good for text classification

### 2. Logistic Regression
- Linear model with interpretability
- Balanced performance
- Feature importance analysis

### 3. LSTM (Long Short-Term Memory)
- Deep learning approach
- Captures sequential patterns
- Best context understanding

## 📈 Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Toxic prediction accuracy
- **Recall**: Toxic comment detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🌐 Web Interface Features
- Text input for custom comments
- Real-time toxicity prediction
- Model selection dropdown
- Confidence scores
- Visual feedback (✅ Non-Toxic / 🚫 Toxic)

## 🔮 Future Enhancements
- Multi-label classification (obscene, insult, threat, etc.)
- Transformer models (BERT, DistilBERT)
- Real-time chat moderation integration
- API endpoint for external applications
- Model explainability (LIME/SHAP)

## 📝 License
This project is for educational purposes.

## 👥 Contributing
Contributions are welcome! Please feel free to submit pull requests.

## 📧 Contact
For questions or feedback, please open an issue in the repository.
