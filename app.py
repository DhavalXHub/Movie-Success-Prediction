"""
Pre-Release Movie Success Prediction - Streamlit App
Interactive web app for predicting movie success BEFORE release using only pre-release data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pre-Release Movie Success Predictor",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stNumberInput>div>div>input {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSlider>div>div>div {
        background-color: #2d2d2d;
    }
    h1 {
        color: #ffffff;
        text-align: center;
    }
    .prediction-box {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .hit {
        color: #51cf66;
        font-size: 24px;
        font-weight: bold;
    }
    .flop {
        color: #ff6b6b;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1>🎬 Pre-Release Movie Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Predict movie success before release using only pre-release data</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model and preprocessors
@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        model = joblib.load('movie_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Try to load genre encoder if it exists
        genre_encoder = None
        if os.path.exists('genre_encoder.pkl'):
            genre_encoder = joblib.load('genre_encoder.pkl')
        
        return model, scaler, feature_names, genre_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please run train_model.py first to train the model.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load model
model, scaler, feature_names, genre_encoder = load_model()

# Get model accuracy (if available)
@st.cache_data
def load_model_info():
    """Load model information"""
    try:
        if os.path.exists('model_info.pkl'):
            return joblib.load('model_info.pkl')
        return {}
    except:
        return {}

model_info = load_model_info()
model_accuracy = model_info.get('accuracy_str', model_info.get('accuracy', 'N/A'))
model_timestamp = model_info.get('timestamp', 'N/A')
model_name = model_info.get('model_name', 'Random Forest Classifier')

# Sidebar with model info
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(f"**Model:** {model_name}")
    st.markdown(f"**Accuracy:** {model_accuracy}")
    st.markdown(f"**Last Updated:** {model_timestamp}")
    if 'precision' in model_info:
        st.markdown(f"**Precision:** {model_info.get('precision', 'N/A'):.4f}")
        st.markdown(f"**Recall:** {model_info.get('recall', 'N/A'):.4f}")
        st.markdown(f"**F1-Score:** {model_info.get('f1', 'N/A')}")
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter pre-release movie details
    2. Click 'Predict' button
    3. View prediction results
    """)
    st.markdown("---")
    st.markdown("### Note")
    st.info("This model uses only pre-release features. Post-release data like ratings and votes are excluded to avoid data leakage.")

# Main form
st.markdown("### Enter Pre-Release Movie Details")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Genre input
    if genre_encoder is not None:
        try:
            genre_options = list(genre_encoder.classes_)
            genre = st.selectbox("Genre", genre_options, key="genre")
            genre_encoded = genre_encoder.transform([genre])[0]
        except Exception as e:
            st.warning(f"Genre encoder error: {e}")
            genre_encoded = 0
    else:
        st.info("Genre encoder not found. Using default value.")
        genre_encoded = st.number_input("Genre (Encoded)", min_value=0, value=0, key="genre")
    
    # Budget
    budget = st.number_input("Budget (in dollars)", min_value=0, value=50000000, step=1000000, key="budget")
    
    # Runtime
    runtime = st.number_input("Runtime (in minutes)", min_value=0, value=120, step=5, key="runtime")
    
    # Release Year
    release_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2024, key="year")

with col2:
    # Cast Popularity (slider 0-100)
    cast_popularity = st.slider("Cast Popularity (0-100)", min_value=0, max_value=100, value=50, step=1, key="cast_pop")
    st.caption("Estimated popularity of the cast")
    
    # Director Popularity (slider 0-100)
    director_popularity = st.slider("Director Popularity (0-100)", min_value=0, max_value=100, value=50, step=1, key="director_pop")
    st.caption("Estimated popularity of the director")
    
    # Production Company Score (slider 0-100)
    production_company_score = st.slider("Production Company Score (0-100)", min_value=0, max_value=100, value=50, step=1, key="company_score")
    st.caption("Historical success rate of the production company")

# Predict button
st.markdown("---")
predict_button = st.button("🔮 Predict Movie Success", type="primary", use_container_width=True)

# Prediction logic
if predict_button:
    try:
        # Prepare input features in the same order as training
        input_features = {}
        for feature in feature_names:
            if feature == 'genre_encoded':
                input_features[feature] = genre_encoded
            elif feature == 'budget':
                input_features[feature] = budget
            elif feature == 'runtime':
                input_features[feature] = runtime
            elif feature == 'release_year':
                input_features[feature] = release_year
            elif feature == 'cast_popularity':
                input_features[feature] = cast_popularity
            elif feature == 'director_popularity':
                input_features[feature] = director_popularity
            elif feature == 'production_company_score':
                input_features[feature] = production_company_score
            else:
                # For any other features, use default/median values
                input_features[feature] = 0
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([input_features])
        input_df = input_df[feature_names]  # Ensure correct order
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Result box
        if prediction == 1:
            result_html = f"""
            <div class="prediction-box">
                <p class="hit">✅ HIT</p>
                <p>This movie is predicted to be a <strong>success</strong>!</p>
            </div>
            """
        else:
            result_html = f"""
            <div class="prediction-box">
                <p class="flop">❌ FLOP</p>
                <p>This movie is predicted to be a <strong>flop</strong>.</p>
            </div>
            """
        
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Probability metrics
        st.markdown("### Prediction Probabilities")
        prob_hit = prediction_proba[1] * 100
        prob_flop = prediction_proba[0] * 100
        
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric("Flop Probability", f"{prob_flop:.2f}%")
            st.progress(prob_flop / 100)
        
        with col_prob2:
            st.metric("Hit Probability", f"{prob_hit:.2f}%")
            st.progress(prob_hit / 100)
        
        # Model Accuracy
        st.markdown("---")
        st.markdown("### Model Information")
        if isinstance(model_accuracy, str):
            st.metric("Model Accuracy (trained)", model_accuracy)
        else:
            st.metric("Model Accuracy (trained)", f"{model_accuracy*100:.2f}%")
        
        # Additional insights
        st.markdown("---")
        st.markdown("### Insights")
        
        if prediction == 1:
            st.success(f"🎉 Based on pre-release features, this movie has a {prob_hit:.1f}% chance of being a hit!")
            if budget > 0:
                st.info(f"💰 Budget: ${budget:,.0f}")
        else:
            st.warning(f"⚠️ Based on pre-release features, this movie has a {prob_flop:.1f}% chance of being a flop.")
            if budget > 0:
                st.info(f"💰 Budget: ${budget:,.0f}")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.info("Please check that all input fields are filled correctly.")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Pre-Release Movie Success Prediction System</p>
    <p>Model trained using TMDB dataset</p>
    <p style='font-size: 12px;'>This model excludes post-release features like ratings or votes to avoid data leakage.</p>
</div>
""", unsafe_allow_html=True)
