import os
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
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
    "Predict whether a Kepler Object of Interest is "
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

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH, compile=False)

    scaler = joblib.load(SCALER_PATH)

    label_encoder = joblib.load(ENCODER_PATH)

    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()


# ============================================================
# INPUT FEATURES (MUST MATCH TRAINING ORDER)
# ============================================================

st.subheader("Enter Scientific Features")

koi_period = st.number_input("Orbital Period", value=10.0)
koi_impact = st.number_input("Impact Parameter", value=0.5)
koi_duration = st.number_input("Transit Duration", value=5.0)
koi_depth = st.number_input("Transit Depth", value=500.0)
koi_prad = st.number_input("Planet Radius", value=2.0)
koi_teq = st.number_input("Equilibrium Temperature", value=700.0)
koi_insol = st.number_input("Insolation Flux", value=100.0)
koi_model_snr = st.number_input("Signal-to-Noise Ratio", value=20.0)
koi_steff = st.number_input("Stellar Effective Temperature", value=5500.0)
koi_slogg = st.number_input("Stellar Surface Gravity", value=4.4)
koi_srad = st.number_input("Stellar Radius", value=1.0)


# ============================================================
# FEATURE ARRAY
# ============================================================

features = np.array([[
    koi_period,
    koi_impact,
    koi_duration,
    koi_depth,
    koi_prad,
    koi_teq,
    koi_insol,
    koi_model_snr,
    koi_steff,
    koi_slogg,
    koi_srad
]])


# ============================================================
# SCALE INPUT
# ============================================================

features_scaled = scaler.transform(features)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict"):

    probabilities = model.predict(features_scaled)

    predicted_index = np.argmax(probabilities, axis=1)[0]

    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"Prediction: {predicted_label}")

    st.subheader("Class Probabilities")

    for cls, prob in zip(label_encoder.classes_, probabilities[0]):
        st.write(f"{cls}: {prob:.4f}")
