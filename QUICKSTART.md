# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data
Run Python and execute:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

Or simply run:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

### Step 3: Generate Sample Data (or use Kaggle dataset)

**Option A: Generate Sample Data**
```bash
python src/generate_sample_data.py
```

**Option B: Use Kaggle Dataset**
1. Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Place `train.csv` in the `data/` folder

### Step 4: Train Models
```bash
python src/train_models.py
```

This will:
- Preprocess the data
- Train Naive Bayes, Logistic Regression, and LSTM models
- Save models to `models/` folder
- Generate evaluation reports and visualizations

**Expected training time:**
- Naive Bayes: ~10 seconds
- Logistic Regression: ~20 seconds
- LSTM: ~5-10 minutes (depends on dataset size and hardware)

### Step 5: Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Optional: Evaluate Models

To run comprehensive evaluation:
```bash
python src/evaluate_models.py
```

This generates:
- ROC curves
- Word clouds
- Detailed metrics comparison

---

## 🧪 Test Predictions (Command Line)

```bash
python src/predict.py
```

---

## 📁 Project Structure After Setup

```
Toxic Comment Classifier/
│
├── data/
│   └── train.csv                      # Dataset (2000 samples)
│
├── models/
│   ├── naive_bayes_model.pkl          # Trained NB model
│   ├── logistic_regression_model.pkl  # Trained LR model
│   ├── lstm_model.h5                  # Trained LSTM model
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── lstm_tokenizer.pkl             # LSTM tokenizer
│   ├── model_comparison.csv           # Performance comparison
│   └── *.png                          # Visualization plots
│
├── src/
│   ├── preprocessing.py               # Data preprocessing
│   ├── train_models.py                # Model training
│   ├── evaluate_models.py             # Model evaluation
│   ├── predict.py                     # Prediction utilities
│   └── generate_sample_data.py        # Sample data generator
│
├── app.py                             # Streamlit web app
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation
```

---

## 🎯 Usage Examples

### Web Interface
1. Open the Streamlit app
2. Enter a comment in the text box
3. Select a model or use "All Models"
4. Click "Analyze Comment"
5. View the prediction results

### Python API
```python
from src.predict import ToxicCommentPredictor

# Initialize predictor
predictor = ToxicCommentPredictor()
predictor.load_all_models()

# Make prediction
text = "This is a great article!"
results = predictor.predict_all(text)

for model, result in results.items():
    print(f"{model}: {result['label']} ({result['probability']:.2%})")
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'nltk'"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "NLTK data not found"
**Solution:** Download NLTK data
```python
import nltk
nltk.download('all')
```

### Issue: "Dataset not found"
**Solution:** Generate sample data or download Kaggle dataset
```bash
python src/generate_sample_data.py
```

### Issue: "Models not found"
**Solution:** Train models first
```bash
python src/train_models.py
```

### Issue: TensorFlow/Keras errors
**Solution:** Ensure compatible versions
```bash
pip install tensorflow==2.13.0 keras==2.13.1
```

---

## 📈 Expected Performance

With sample dataset (2000 samples):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~0.85 | ~0.80 | ~0.75 | ~0.77 |
| Logistic Regression | ~0.88 | ~0.85 | ~0.80 | ~0.82 |
| LSTM | ~0.90 | ~0.88 | ~0.85 | ~0.86 |

*Note: Performance varies based on dataset quality and size*

---

## 🎓 Next Steps

1. **Improve Models:**
   - Use larger dataset from Kaggle
   - Tune hyperparameters
   - Try different architectures

2. **Extend Functionality:**
   - Multi-label classification
   - Add BERT/DistilBERT models
   - Create REST API

3. **Deploy:**
   - Deploy on Streamlit Cloud
   - Create Docker container
   - Integrate with chat platforms

---

## 💡 Tips

- **For better LSTM performance:** Use GPU if available
- **For faster training:** Reduce dataset size during development
- **For production:** Use the Kaggle dataset (150k+ samples)
- **For deployment:** Consider model compression and optimization

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset format

---

Happy Classifying! 🛡️
