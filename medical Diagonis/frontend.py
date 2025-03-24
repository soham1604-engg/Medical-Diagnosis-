import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("multi_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Feature names based on training data
features = ['age', 'sex', 'TSH', 'T3', 'TT4', 'cholesterol', 'blood_pressure', 'heart_rate']

def predict_disease(user_data):
    new_data = pd.DataFrame([user_data])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    disease = target_encoder.inverse_transform(prediction)[0]
    return disease

st.title("Multi-Disease Prediction System")
st.write("Enter patient details below:")

# Input fields
user_data = {}
user_data['age'] = st.number_input("Age", min_value=0, max_value=120, value=30)
user_data['sex'] = st.radio("Sex", ('Male', 'Female'))
user_data['sex'] = 0 if user_data['sex'] == 'Male' else 1
user_data['TSH'] = st.number_input("TSH Level", min_value=0.0, step=0.1)
user_data['T3'] = st.number_input("T3 Level", min_value=0.0, step=0.1)
user_data['TT4'] = st.number_input("TT4 Level", min_value=0.0, step=0.1)
user_data['cholesterol'] = st.number_input("Cholesterol", min_value=0.0, step=0.1)
user_data['blood_pressure'] = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
user_data['heart_rate'] = st.number_input("Heart Rate", min_value=0.0, step=1.0)

if st.button("Predict"):
    disease = predict_disease(user_data)
    st.success(f"Predicted Disease: {disease}")