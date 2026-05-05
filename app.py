import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# =========================
# Paths
# =========================
MODEL_PATH = "models/exoplanet_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# =========================
# Load Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, label_encoder = load_artifacts()

# =========================
# UI
# =========================
st.title("🪐 Exoplanet Classification")

st.write("Enter feature values to predict exoplanet class.")

# Example: 11 features
features = []
for i in range(11):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)

    label = label_encoder.inverse_transform(predicted_class)

    st.success(f"Prediction: {label[0]}")
