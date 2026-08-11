import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model and preprocessing objects
# --------------------------------------------------

model = joblib.load("diabetes_logistic_model.pkl")
imputer = joblib.load("diabetes_imputer.pkl")
scaler = joblib.load("diabetes_scaler.pkl")


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction App")

st.write(
    "Enter the patient's information below to predict "
    "the probability of diabetes."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=250,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=150,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Diabetes"):

    # Create DataFrame using the same feature order
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Replace invalid zero values with NaN
    columns_with_invalid_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[columns_with_invalid_zero] = (
        input_data[columns_with_invalid_zero].replace(0, float("nan"))
    )

    # Ensure the data type matches the training data
    input_data = input_data.astype(float)

    # Apply the same preprocessing used during training
    input_imputed = imputer.transform(input_data)

    input_scaled = scaler.transform(input_imputed)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get diabetes probability
    probability = model.predict_proba(input_scaled)[0][1]


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetes")
    else:
        st.success("✅ Prediction: No Diabetes")

    st.info(
        f"Estimated Probability of Diabetes: {probability:.2%}"
    )
