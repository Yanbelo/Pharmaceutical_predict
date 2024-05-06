# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import os

# Load pre-trained models and encoders from saved pickle files
def load_models():
    models = {}
    try:
        with open('rf_target.pkl', 'rb') as f:
            models['rf_target'] = pickle.load(f)

        with open('rf_formula.pkl', 'rb') as f:
            models['rf_formula'] = pickle.load(f)

        with open('le_target.pkl', 'rb') as f:
            models['le_target'] = pickle.load(f)

        with open('le_formula.pkl', 'rb') as f:
            models['le_formula'] = pickle.load(f)

    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please ensure all model files are available.")
        st.stop()

    return models['rf_target'], models['rf_formula'], models['le_target'], models['le_formula']

# Load the models and encoders with error handling
rf_target, rf_formula, le_target, le_formula = load_models()

# Streamlit app title
st.title('Pharmaceutical Compound Prediction')

# Sidebar input section
st.sidebar.header('Input Features')

# Function to gather user input features
def user_input_features():
    rt = st.sidebar.number_input('Retention Time (RT)', min_value=0.0, step=0.01, value=0.70)
    area = st.sidebar.number_input('Area', min_value=0, step=1000, value=120000)
    expected_mz = st.sidebar.number_input('Expected m/z', min_value=0.0, step=0.001, value=350.123)
    measured_mz = st.sidebar.number_input('Measured m/z', min_value=0.0, step=0.001, value=350.124)
    delta_mz = st.sidebar.number_input('Delta m/z', min_value=-10.0, max_value=10.0, step=0.001, value=-0.001)
    isotopic_score = st.sidebar.number_input('Isotopic Pattern Score (%)', min_value=0, max_value=100, step=1, value=97)

    # Organize input data into a dictionary and convert it to a DataFrame
    data = {
        'RT': rt,
        'Area': area,
        'Expected m/z': expected_mz,
        'Measured m/z': measured_mz,
        'Delta m/z': delta_mz,
        'Isotopic Pattern Score (%)': isotopic_score
    }

    return pd.DataFrame([data])

# Collect user input (always displayed)
input_df = user_input_features()

# Display input features on the main page
st.write('### Input Features')
st.write(input_df)

# Create a placeholder to display predictions
prediction_placeholder = st.empty()

# Create a Run button for the prediction process
if st.sidebar.button('Run Prediction'):
    # Predict using the loaded models
    predictions_target = rf_target.predict(input_df)
    predictions_formula = rf_formula.predict(input_df)

    # Decode predictions to human-readable labels
    predicted_labels_target = le_target.inverse_transform(predictions_target)
    predicted_labels_formula = le_formula.inverse_transform(predictions_formula)

    # Display predictions
    prediction_placeholder.write('### Predictions')
    for idx, (target, formula) in enumerate(zip(predicted_labels_target, predicted_labels_formula)):
        prediction_placeholder.write(f'Prediction for Sample {idx + 1}: **Target = {target}, Formula = {formula}**')
