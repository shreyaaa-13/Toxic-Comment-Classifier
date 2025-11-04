"""
Streamlit Web Application for Toxic Comment Classifier
Interactive UI for real-time toxicity prediction
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append('src')

from predict import ToxicCommentPredictor


# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .toxic-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #d32f2f;
        border-left: 5px solid #b71c1c;
        margin: 1rem 0;
        color: white;
    }
    .toxic-box h2 {
        color: white;
        margin: 0 0 1rem 0;
    }
    .toxic-box p {
        color: white;
        margin: 0.5rem 0;
    }
    .non-toxic-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #2e7d32;
        border-left: 5px solid #1b5e20;
        margin: 1rem 0;
        color: white;
    }
    .non-toxic-box h2 {
        color: white;
        margin: 0 0 1rem 0;
    }
    .non-toxic-box p {
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f5f5f5;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load predictor with caching"""
    predictor = ToxicCommentPredictor(models_dir='models')
    predictor.load_all_models()
    return predictor


def display_prediction_result(result, model_name):
    """Display prediction result with styling"""
    if 'error' in result:
        st.error(f"❌ Error: {result['error']}")
        return
    
    prediction = result['prediction']
    probability = result['probability']
    label = result['label']
    
    # Display result box
    if prediction == 1:
        st.markdown(f"""
            <div class="toxic-box">
                <h2>🚫 TOXIC COMMENT DETECTED</h2>
                <p style="font-size: 1.2rem;">Model: <strong>{model_name}</strong></p>
                <p style="font-size: 1.1rem;">Confidence: <strong>{probability:.2%}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="non-toxic-box">
                <h2>✅ NON-TOXIC COMMENT</h2>
                <p style="font-size: 1.2rem;">Model: <strong>{model_name}</strong></p>
                <p style="font-size: 1.1rem;">Confidence: <strong>{(1-probability):.2%}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    if prediction == 1:
        st.progress(probability)
    else:
        st.progress(1 - probability)


def create_comparison_chart(results):
    """Create comparison chart for all models"""
    models = []
    probabilities = []
    predictions = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            models.append(model_name)
            probabilities.append(result['probability'])
            predictions.append(result['label'])
    
    if not models:
        return None
    
    # Create dataframe
    df = pd.DataFrame({
        'Model': models,
        'Toxic Probability': probabilities,
        'Prediction': predictions
    })
    
    # Create bar chart
    fig = go.Figure()
    
    colors = ['#f44336' if pred == 'Toxic' else '#4caf50' for pred in predictions]
    
    fig.add_trace(go.Bar(
        x=df['Model'],
        y=df['Toxic Probability'],
        marker_color=colors,
        text=[f"{p:.2%}" for p in probabilities],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Toxic Probability: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Model',
        yaxis_title='Toxic Probability',
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Toxic Comment Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Comment Moderation using NLP</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=100)
        st.title("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["All Models", "Naive Bayes", "Logistic Regression", "LSTM"],
            help="Choose which model to use for prediction"
        )
        
        st.markdown("---")
        
        # Information
        st.subheader("📊 About")
        st.info("""
        This application uses machine learning and deep learning models to classify comments as toxic or non-toxic.
        
        **Models:**
        - 🔹 Naive Bayes
        - 🔹 Logistic Regression
        - 🔹 LSTM (Deep Learning)
        """)
        
        st.markdown("---")
        
        # Examples
        st.subheader("💡 Example Comments")
        
        example_toxic = st.button("Try Toxic Example")
        example_non_toxic = st.button("Try Non-Toxic Example")
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📈 Model Info")
        if os.path.exists('models/model_comparison.csv'):
            comparison_df = pd.read_csv('models/model_comparison.csv')
            st.dataframe(comparison_df, use_container_width=True)
    
    # Load predictor
    try:
        predictor = load_predictor()
        
        if not (predictor.nb_loaded or predictor.lr_loaded or predictor.lstm_loaded):
            st.error("""
            ❌ **No models found!**
            
            Please train the models first:
            1. Generate sample data: `python src/generate_sample_data.py`
            2. Train models: `python src/train_models.py`
            """)
            return
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Comment to Analyze")
        
        # Text input
        default_text = ""
        
        if example_toxic:
            default_text = "You are an idiot and nobody likes you"
        elif example_non_toxic:
            default_text = "This is a great article, thanks for sharing!"
        
        user_input = st.text_area(
            "Comment Text",
            value=default_text,
            height=150,
            placeholder="Type or paste a comment here...",
            help="Enter the comment you want to analyze for toxicity"
        )
        
        # Predict button
        predict_button = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Enter** a comment in the text box
        2. **Select** a model (or use all)
        3. **Click** Analyze Comment
        4. **View** the toxicity prediction
        
        The models analyze:
        - Word patterns
        - Context
        - Sentiment
        - Language structure
        """)
    
    # Prediction
    if predict_button and user_input.strip():
        with st.spinner("🔄 Analyzing comment..."):
            
            if model_choice == "All Models":
                # Predict with all models
                results = predictor.predict_all(user_input)
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Display individual results
                cols = st.columns(len(results))
                
                for idx, (model_name, result) in enumerate(results.items()):
                    with cols[idx]:
                        if 'error' not in result:
                            prediction = result['prediction']
                            probability = result['probability']
                            
                            # Metric display
                            emoji = "🚫" if prediction == 1 else "✅"
                            label = result['label']
                            
                            st.metric(
                                label=f"{emoji} {model_name}",
                                value=label,
                                delta=f"{probability:.2%} confidence"
                            )
                        else:
                            st.error(f"❌ {model_name}\n{result['error']}")
                
                # Comparison chart
                st.markdown("---")
                fig = create_comparison_chart(results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Consensus
                st.markdown("---")
                predictions = [r['prediction'] for r in results.values() if 'error' not in r]
                if predictions:
                    consensus = sum(predictions) / len(predictions)
                    
                    if consensus >= 0.67:
                        st.error("🚫 **CONSENSUS: TOXIC** - Majority of models detected toxicity")
                    elif consensus <= 0.33:
                        st.success("✅ **CONSENSUS: NON-TOXIC** - Majority of models found no toxicity")
                    else:
                        st.warning("⚠️ **CONSENSUS: UNCERTAIN** - Models disagree on classification")
            
            else:
                # Predict with single model
                try:
                    prediction, probability = predictor.predict_with_model(user_input, model_choice)
                    
                    result = {
                        'prediction': int(prediction),
                        'probability': float(probability),
                        'label': 'Toxic' if prediction == 1 else 'Non-Toxic'
                    }
                    
                    st.markdown("---")
                    display_prediction_result(result, model_choice)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    elif predict_button:
        st.warning("⚠️ Please enter a comment to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>Built with ❤️ using Streamlit, Scikit-learn, and TensorFlow</p>
            <p>🛡️ Promoting safer online communities through AI</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
