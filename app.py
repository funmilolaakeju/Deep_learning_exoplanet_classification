import os
import numpy as np
import streamlit as st
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🪐",
    layout="centered"
)

st.title("Deep Learning Exoplanet Classification")

st.write(
    "This app predicts whether a Kepler Object of Interest is "
    "CANDIDATE, CONFIRMED, or FALSE POSITIVE."
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_model_fixed.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ============================================================
# INPUTS (MUST MATCH TRAINING FEATURE ORDER)
# ============================================================

st.subheader("Enter Scientific Features")

features = np.array([[
    st.number_input("Orbital Period", value=10.0),
    st.number_input("Impact Parameter", value=0.5),
    st.number_input("Transit Duration", value=5.0),
    st.number_input("Transit Depth", value=500.0),
    st.number_input("Planet Radius", value=2.0),
    st.number_input("Equilibrium Temperature", value=700.0),
    st.number_input("Insolation Flux", value=100.0),
    st.number_input("Signal-to-Noise Ratio", value=20.0),
    st.number_input("Stellar Effective Temperature", value=5500.0),
    st.number_input("Stellar Surface Gravity", value=4.4),
    st.number_input("Stellar Radius", value=1.0),
]])


# ============================================================
# PREPROCESS
# ============================================================

features_scaled = scaler.transform(features)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict"):

    probabilities = model.predict(features_scaled)

    predicted_index = np.argmax(probabilities)
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"Prediction: {predicted_class}")

    st.subheader("Class Probabilities")

    for class_name, prob in zip(label_encoder.classes_, probabilities[0]):
        st.write(f"{class_name}: {prob:.4f}")