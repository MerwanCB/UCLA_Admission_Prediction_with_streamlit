import pickle
import pandas as pd
import streamlit as st
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="UCLA Admission Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- Title and Description ---
st.title("🎓 UCLA Admission Chance Predictor")
st.write(
    """
    Predicts a student's chances of admission to a Master's program at UCLA.
    A 'High Chance' is predicted if the probability is 80% or higher.
    """
)

# --- Load Pre-trained Model and Scaler ---
MODEL_PATH = "models/mlp_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load pickle files (model and scaler)
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Ensure the model and scaler are saved correctly.")
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

mlp_model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)

# Stop execution if model or scaler failed to load
if mlp_model is None or scaler is None:
    st.stop()

# --- Define Expected Feature Order ---
# Ensure this matches the training script's feature order after get_dummies
expected_feature_order = [
    'GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA',
    'University_Rating_1', 'University_Rating_2', 'University_Rating_3',
    'University_Rating_4', 'University_Rating_5',
    'Research_0', 'Research_1'
]

# --- Input Form ---
with st.form("admission_inputs"):
    st.subheader("Applicant Profile Details")

    # Input fields for user data
    gre_score = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, value=315, step=1)
    toefl_score = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=105, step=1)
    university_rating = st.selectbox("University Rating (1=Low, 5=High)", options=[1, 2, 3, 4, 5], index=2)
    sop_score = st.slider("SOP Strength (1.0=Low, 5.0=High)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    lor_score = st.slider("LOR Strength (1.0=Low, 5.0=High)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    cgpa = st.number_input("Undergraduate CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, step=0.1, format="%.2f")
    research_exp = st.selectbox("Research Experience", options=["Yes", "No"], index=0)

    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")

# --- Process Inputs and Predict ---
if submitted:
    # Initialize feature dictionary with zeros
    input_data = {feature: 0 for feature in expected_feature_order}

    # Populate with user inputs
    input_data['GRE_Score'] = gre_score
    input_data['TOEFL_Score'] = toefl_score
    input_data['SOP'] = sop_score
    input_data['LOR'] = lor_score
    input_data['CGPA'] = cgpa

    # Handle one-hot encoding for categorical features
    rating_col = f'University_Rating_{university_rating}'
    if rating_col in input_data:
        input_data[rating_col] = 1

    # Research experience (1=Yes, 0=No)
    if research_exp == "Yes":
        input_data['Research_1'] = 1
    else:
        input_data['Research_0'] = 1

    # Create DataFrame in the correct order
    try:
        input_df = pd.DataFrame([input_data], columns=expected_feature_order)
    except Exception as e:
        st.error(f"Error creating input DataFrame: {e}")
        st.stop()

    # Scale features
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error scaling input features: {e}")
        st.stop()

    # Make predictions
    try:
        prediction = mlp_model.predict(input_scaled)
        prediction_proba = mlp_model.predict_proba(input_scaled)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Display results
    st.subheader("Prediction Result:")
    probability_high_chance = prediction_proba[0][1]  # Probability of 'High Chance'

    if prediction[0] == 1:
        st.success(f"✅ High Chance of Admission")
    else:
        st.warning(f"⚠️ Lower Chance of Admission")

    st.write(f"*(Model's estimated probability: {probability_high_chance:.2%})*")
    st.write("---")

# --- Sidebar Information ---
st.sidebar.header("About the Model")
st.sidebar.write(
    """
    This prediction uses a Multi-Layer Perceptron (MLP) Neural Network trained on
    historical admission data. Factors include GRE/TOEFL scores, CGPA, university rating,
    SOP/LOR strength, and research experience.
    """
)

# Display loss curve if available
LOSS_CURVE_PATH = "reports/figures/mlp_loss_curve.png"
if os.path.exists(LOSS_CURVE_PATH):
    st.sidebar.image(LOSS_CURVE_PATH, caption="Model Training Loss Curve")
else:
    st.sidebar.write("(Loss curve image not found)")

st.sidebar.info("Disclaimer: Predictions are based on historical data and do not guarantee admission.")