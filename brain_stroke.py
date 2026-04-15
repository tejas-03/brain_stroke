import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# Debug: Check available files
# -------------------------------
st.write("Files in directory:", os.listdir())

# -------------------------------
# Load Model Safely
# -------------------------------
try:
    model = joblib.load("stroke_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'stroke_model.pkl' not found. Please upload it.")
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.title("🧠 Stroke Prediction App")

st.write("Enter patient details below:")

# -------------------------------
# User Inputs
# -------------------------------
age = st.number_input("Age", min_value=0, max_value=120, step=1)

hypertension = st.selectbox(
    "Hypertension (0 = No, 1 = Yes)",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease (0 = No, 1 = Yes)",
    [0, 1]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    step=0.1
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    step=0.1
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    # Prepare input data
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])

    # Make prediction
    prediction = model.predict(input_data)

    # Output result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
