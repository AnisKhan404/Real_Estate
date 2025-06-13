import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

# Set page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="centered")
st.title("🏠  Real Estate Price Prediction")

# Load CSV data
df = pd.read_csv("Cleaned_Real_Estate.csv")
st.header("🏠 Real Estate Data")
st.dataframe(df)

# Show map
st.header("📍 Property Locations Map")
with open("Real_Estate_Locations.html", 'r', encoding='utf-8') as f:
    map_html = f.read()
components.html(map_html, height=500, scrolling=True)

# Load trained model and scaler
with open("Real_Estate.pkl", "rb") as f:
    model = pickle.load(f)
    scaler = joblib.load("Scaler.pkl")  

# Input form for prediction
st.header("📊 Predict Property Price")

with st.form("prediction_form"):
    house_age = st.number_input("House Age (in years)", min_value=0.0, max_value=100.0, step=0.5)
    distance = st.number_input("Distance to Nearest MRT Station (in meters)", min_value=50.0, step=10.0)
    convenience_stores = st.number_input("Number of Convenience Stores", min_value=0, step=1)
    latitude = st.number_input("Latitude", min_value=24.9673, format="%.6f")
    longitude = st.number_input("Longitude", min_value=121.5149, format="%.6f")
    
    submitted = st.form_submit_button("Predict Price")

    if submitted:
        st.write("Form Submitted")  # Debug line
        input_data = np.array([[house_age, distance, convenience_stores, latitude, longitude]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted House Price (per unit area): **NT${prediction:.2f}**")
