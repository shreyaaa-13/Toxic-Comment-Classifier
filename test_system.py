"""
System Test Script
Tests all components of the Toxic Comment Classifier
"""

import os
import sys
import traceback


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*60)
    print("Testing Package Imports")
    print("="*60)
    
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'nltk',
        'tensorflow',
        'keras',
        'streamlit',
        'matplotlib',
        'seaborn',
        'wordcloud',
        'plotly',
        'joblib'
    ]
    
    failed = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True


def test_nltk_data():
    """Test if NLTK data is downloaded"""
    print("\n" + "="*60)
    print("Testing NLTK Data")
    print("="*60)
    
    import nltk
    
    datasets = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
    failed = []
    
    for dataset in datasets:
        try:
            nltk.data.find(f'corpora/{dataset}' if dataset != 'punkt' else f'tokenizers/{dataset}')
            print(f"✅ {dataset}")
        except LookupError:
            print(f"❌ {dataset}")
            failed.append(dataset)
    
    if failed:
        print(f"\n⚠️  Missing NLTK data: {', '.join(failed)}")
        print("Download with: python -c \"import nltk; nltk.download(['stopwords', 'wordnet', 'punkt', 'omw-1.4'])\"")
        return False
    else:
        print("\n✅ All NLTK data available!")
        return True


def test_directories():
    """Test if required directories exist"""
    print("\n" + "="*60)
    print("Testing Directory Structure")
    print("="*60)
    
    directories = ['data', 'models', 'src']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (creating...)")
            os.makedirs(directory, exist_ok=True)
    
    print("\n✅ Directory structure verified!")
    return True


def test_preprocessing():
    """Test preprocessing module"""
    print("\n" + "="*60)
    print("Testing Preprocessing Module")
    print("="*60)
    
    try:
        sys.path.append('src')
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        test_text = "This is a TEST comment with URLs http://example.com and @mentions #hashtags!!!"
        processed = preprocessor.preprocess_text(test_text)
        
        print(f"Original:  {test_text}")
        print(f"Processed: {processed}")
        
        if processed and len(processed) > 0:
            print("\n✅ Preprocessing working correctly!")
            return True
        else:
            print("\n❌ Preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def test_data_availability():
    """Test if dataset is available"""
    print("\n" + "="*60)
    print("Testing Data Availability")
    print("="*60)
    
    data_path = 'data/train.csv'
    
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"✅ Dataset found: {len(df)} samples")
        print(f"   Columns: {df.columns.tolist()}")
        return True
    else:
        print(f"❌ Dataset not found at {data_path}")
        print("   Generate with: python src/generate_sample_data.py")
        return False


def test_models_availability():
    """Test if trained models are available"""
    print("\n" + "="*60)
    print("Testing Model Availability")
    print("="*60)
    
    models = {
        'Naive Bayes': 'models/naive_bayes_model.pkl',
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'LSTM': 'models/lstm_model.h5',
        'TF-IDF Vectorizer': 'models/tfidf_vectorizer.pkl',
        'LSTM Tokenizer': 'models/lstm_tokenizer.pkl'
    }
    
    found = 0
    
    for name, path in models.items():
        if os.path.exists(path):
            print(f"✅ {name}")
            found += 1
        else:
            print(f"❌ {name}")
    
    if found == 0:
        print("\n⚠️  No models found!")
        print("   Train models with: python src/train_models.py")
        return False
    elif found < len(models):
        print(f"\n⚠️  Only {found}/{len(models)} models found")
        return False
    else:
        print(f"\n✅ All models available!")
        return True


def test_prediction():
    """Test prediction functionality"""
    print("\n" + "="*60)
    print("Testing Prediction")
    print("="*60)
    
    try:
        sys.path.append('src')
        from predict import ToxicCommentPredictor
        
        predictor = ToxicCommentPredictor(models_dir='models')
        
        # Try to load models
        loaded = predictor.load_all_models()
        
        if not loaded:
            print("❌ No models loaded")
            return False
        
        # Test prediction
        test_comment = "This is a great article, thanks for sharing!"
        
        print(f"\nTest comment: \"{test_comment}\"")
        
        results = predictor.predict_all(test_comment)
        
        for model_name, result in results.items():
            if 'error' not in result:
                emoji = "🚫" if result['prediction'] == 1 else "✅"
                print(f"{emoji} {model_name}: {result['label']} ({result['probability']:.2%})")
            else:
                print(f"❌ {model_name}: {result['error']}")
        
        print("\n✅ Prediction working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def test_streamlit():
    """Test if Streamlit app can be imported"""
    print("\n" + "="*60)
    print("Testing Streamlit App")
    print("="*60)
    
    if os.path.exists('app.py'):
        print("✅ app.py found")
        
        # Check if streamlit is installed
        try:
            import streamlit
            print("✅ Streamlit installed")
            print("\n💡 Run app with: streamlit run app.py")
            return True
        except ImportError:
            print("❌ Streamlit not installed")
            return False
    else:
        print("❌ app.py not found")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("="*60)
    print("TOXIC COMMENT CLASSIFIER - SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Directory Structure", test_directories),
        ("Preprocessing", test_preprocessing),
        ("Data Availability", test_data_availability),
        ("Model Availability", test_models_availability),
        ("Prediction", test_prediction),
        ("Streamlit App", test_streamlit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. If models not trained: python src/train_models.py")
        print("2. Run web app: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        if not results[4][1]:  # Data availability
            print("\n💡 Quick fix: python src/generate_sample_data.py")
        if not results[5][1]:  # Model availability
            print("💡 Quick fix: python src/train_models.py")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
