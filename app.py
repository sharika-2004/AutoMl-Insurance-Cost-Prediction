import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Load trained model
model = load_model("insurance_automl_model")

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

st.title("Insurance Cost Prediction App")
st.write("Enter customer details to predict insurance charges")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

if st.button("Predict Insurance Charges"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = predict_model(model, data=input_data)
    predicted_cost = prediction["prediction_label"][0]

    st.success(f"Estimated Insurance Charges: ₹ {predicted_cost:,.2f}")