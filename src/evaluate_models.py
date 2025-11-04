"""
Model Evaluation Module
Comprehensive evaluation and comparison of trained models
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from wordcloud import WordCloud

from preprocessing import TextPreprocessor, load_and_prepare_data


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
    
    def load_models(self):
        """Load all trained models"""
        models = {}
        
        # Load Naive Bayes
        try:
            models['nb'] = joblib.load(os.path.join(self.models_dir, 'naive_bayes_model.pkl'))
            print("✅ Naive Bayes loaded")
        except:
            print("❌ Naive Bayes not found")
        
        # Load Logistic Regression
        try:
            models['lr'] = joblib.load(os.path.join(self.models_dir, 'logistic_regression_model.pkl'))
            print("✅ Logistic Regression loaded")
        except:
            print("❌ Logistic Regression not found")
        
        # Load LSTM
        try:
            models['lstm'] = keras.models.load_model(os.path.join(self.models_dir, 'lstm_model.h5'))
            print("✅ LSTM loaded")
        except:
            print("❌ LSTM not found")
        
        # Load vectorizers
        try:
            models['tfidf'] = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            models['tokenizer'] = joblib.load(os.path.join(self.models_dir, 'lstm_tokenizer.pkl'))
        except:
            print("❌ Vectorizers not found")
        
        return models
    
    def evaluate_on_test_set(self, data_path='data/train.csv', test_size=0.2):
        """
        Evaluate all models on test set
        
        Args:
            data_path (str): Path to dataset
            test_size (float): Test set proportion
        """
        print("\n" + "="*60)
        print("EVALUATING MODELS ON TEST SET")
        print("="*60)
        
        # Load data
        X_text, y_labels, df = load_and_prepare_data(data_path)
        
        # Preprocess
        df = self.preprocessor.preprocess_dataframe(df)
        X_processed = df['processed_text'].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_labels, test_size=test_size, random_state=42, stratify=y_labels
        )
        
        # Load models
        models = self.load_models()
        
        results = {}
        
        # Evaluate Naive Bayes
        if 'nb' in models and 'tfidf' in models:
            print("\n📊 Evaluating Naive Bayes...")
            X_test_tfidf = models['tfidf'].transform(X_test)
            y_pred = models['nb'].predict(X_test_tfidf)
            y_proba = models['nb'].predict_proba(X_test_tfidf)[:, 1]
            results['Naive Bayes'] = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Evaluate Logistic Regression
        if 'lr' in models and 'tfidf' in models:
            print("\n📊 Evaluating Logistic Regression...")
            X_test_tfidf = models['tfidf'].transform(X_test)
            y_pred = models['lr'].predict(X_test_tfidf)
            y_proba = models['lr'].predict_proba(X_test_tfidf)[:, 1]
            results['Logistic Regression'] = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Evaluate LSTM
        if 'lstm' in models and 'tokenizer' in models:
            print("\n📊 Evaluating LSTM...")
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            sequences = models['tokenizer'].texts_to_sequences(X_test)
            X_test_seq = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
            y_proba = models['lstm'].predict(X_test_seq, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            results['LSTM'] = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Display results
        self._display_results(results)
        
        # Plot ROC curves
        self._plot_roc_curves(results)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        return metrics
    
    def _display_results(self, results):
        """Display evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        comparison_data = []
        
        for model_name, metrics in results.items():
            print(f"\n📊 {model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = os.path.join(self.models_dir, 'evaluation_results.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✅ Results saved to {comparison_path}")
    
    def _plot_roc_curves(self, results):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (model_name, metrics) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_proba'])
            roc_auc = metrics['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.models_dir, 'roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curves saved to {plot_path}")
        plt.close()
    
    def generate_word_clouds(self, data_path='data/train.csv'):
        """Generate word clouds for toxic and non-toxic comments"""
        print("\n" + "="*60)
        print("GENERATING WORD CLOUDS")
        print("="*60)
        
        # Load data
        X_text, y_labels, df = load_and_prepare_data(data_path)
        
        # Preprocess
        df = self.preprocessor.preprocess_dataframe(df)
        
        # Separate toxic and non-toxic
        toxic_text = ' '.join(df[df['is_toxic'] == 1]['processed_text'].values)
        non_toxic_text = ' '.join(df[df['is_toxic'] == 0]['processed_text'].values)
        
        # Create word clouds
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Toxic word cloud
        if toxic_text:
            wordcloud_toxic = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Reds',
                max_words=100
            ).generate(toxic_text)
            
            axes[0].imshow(wordcloud_toxic, interpolation='bilinear')
            axes[0].set_title('Toxic Comments Word Cloud', fontsize=16, fontweight='bold')
            axes[0].axis('off')
        
        # Non-toxic word cloud
        if non_toxic_text:
            wordcloud_non_toxic = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Greens',
                max_words=100
            ).generate(non_toxic_text)
            
            axes[1].imshow(wordcloud_non_toxic, interpolation='bilinear')
            axes[1].set_title('Non-Toxic Comments Word Cloud', fontsize=16, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, 'word_clouds.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Word clouds saved to {plot_path}")
        plt.close()


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("TOXIC COMMENT CLASSIFIER - MODEL EVALUATION")
    print("="*60)
    
    # Check if data exists
    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        print(f"\n⚠️  Dataset not found at {data_path}")
        print("Please ensure the dataset is available.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(models_dir='models')
    
    # Evaluate models
    results = evaluator.evaluate_on_test_set(data_path=data_path)
    
    # Generate word clouds
    evaluator.generate_word_clouds(data_path=data_path)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
