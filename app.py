import streamlit as st
import numpy as np
import joblib

# Load trained model & preprocessing tools
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# UI Title
st.title("💳 Credit Card Fraud Detector")

st.write("Enter transaction details below to predict if it's fraud or not.")

# Input fields
category = st.selectbox("Category", ["grocery_pos", "gas_transport", "shopping_pos", "kids_pets", "entertainment"])
amt = st.number_input("Transaction Amount", min_value=0.01, max_value=10000.0, step=0.01)
lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01)
long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01)
city_pop = st.number_input("City Population", min_value=0, max_value=1_000_000, step=1)
hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)

# Additional features (fill with default values)
merchant_encoded = -1  # Placeholder for missing merchant encoding
category_encoded = encoders["category"].transform([category])[0] if category in encoders["category"].classes_ else -1
zip_encoded = -1  # Placeholder for zip encoding
state_encoded = -1  # Placeholder for state encoding
city_encoded = -1  # Placeholder for city encoding
gender_encoded = 0  # Assuming default gender encoding
unix_time = hour * 3600  # Approximate Unix time based on the hour

# Prepare input data (must match the 13-feature format used during training)
features = np.array([[category_encoded, amt, lat, long, city_pop, hour, day, merchant_encoded, zip_encoded, state_encoded, city_encoded, gender_encoded, unix_time]])
features_scaled = scaler.transform(features)

# Predict
if st.button("Check Fraud"):
    prediction = model.predict(features_scaled)[0]
    result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Legitimate Transaction"
    st.subheader(result)
