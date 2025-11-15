import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("KNN_heart.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load expected column names
expected_columns = joblib.load("columns.pkl")  # Must be the list of columns

st.title("Heart Disease Prediction")
st.markdown("Provide a following details.")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_Pain = st.selectbox("Chest Pain Type", [
                          "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
resting_BP = st.number_input(
    "Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input(
    "Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [
    "True", "False"])
resting_ecg = st.selectbox("Resting ECG", [
    "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input(
    "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", [
    "Yes", "No"])
oldpeak = st.number_input(
    "ST Depression Induced by Exercise Relative to Rest", min_value=0.0,  max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment",
                        ["Upsloping", "Flat", "Downsloping"])


if st.button("Predict"):
    # One-hot encoding categorical variables
    raw_input = {
        'Age': age,
        'Resting Blood Pressure': resting_BP,
        'Cholesterol': cholesterol,
        'Fasting_blood_sugar': 1 if fasting_blood_sugar == "True" else 0,
        'Max_Heart_Rate': max_heart_rate,
        'Oldpeak': oldpeak,
        # One-hot encoding
        'Sex_Male': 1 if sex == "Male" else 0,
        'Sex_Female': 1 if sex == "Female" else 0,
        'Chest Pain Type_Typical Angina': 1 if chest_Pain == "Typical Angina" else 0,
        'Chest Pain Type_Atypical Angina': 1 if chest_Pain == "Atypical Angina" else 0,
        'Chest Pain Type_Non-anginal Pain': 1 if chest_Pain == "Non-anginal Pain" else 0,
        'Chest Pain Type_Asymptomatic': 1 if chest_Pain == "Asymptomatic" else 0,
        'Resting_ECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'Resting_ECG_ST-T Wave Abnormality': 1 if resting_ecg == "ST-T Wave Abnormality" else 0,
        'Resting_ECG_Left Ventricular Hypertrophy': 1 if resting_ecg == "Left Ventricular Hypertrophy" else 0,
        'Exercise_Induced_Angina_Yes': 1 if exercise_induced_angina == "Yes" else 0,
        'Exercise_Induced_Angina_No': 1 if exercise_induced_angina == "No" else 0,
        'ST_Slope_Upsloping': 1 if st_slope == "Upsloping" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Downsloping': 1 if st_slope == "Downsloping" else 0
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error(
            "The model predicts that you may have ⚠️ high risk of heart disease. Please consult a doctor.")
    else:
        st.success(
            "The model predicts that you have ✅ low risk of heart disease. Maintain a healthy lifestyle!")
