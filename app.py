import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🪐",
    layout="centered"
)

st.title("🪐 Deep Learning Exoplanet Classification")

# =========================
# PATHS (FIXED)
# =========================

MODEL_PATH = "models/exoplanet_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# =========================
# LOAD ARTIFACTS
# =========================

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()


# =========================
# INPUT
# =========================

st.subheader("Enter Features")

features = []

for i in range(11):
    features.append(st.number_input(f"Feature {i+1}", value=0.0))

features = np.array([features])


# =========================
# PREDICT
# =========================

if st.button("Predict"):
    scaled = scaler.transform(features)
    pred = model.predict(scaled)

    class_id = np.argmax(pred, axis=1)
    result = label_encoder.inverse_transform(class_id)

    st.success(f"Prediction: {result[0]}")
