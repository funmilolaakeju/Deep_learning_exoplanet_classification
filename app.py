import os
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# UI
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
# SAFE LOAD (FIX FOR YOUR ERROR)
# ============================================================

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()


# ============================================================
# INPUTS
# ============================================================

st.subheader("Enter Scientific Features")

features = np.array([[
    st.number_input("Orbital Period", 10.0),
    st.number_input("Impact Parameter", 0.5),
    st.number_input("Transit Duration", 5.0),
    st.number_input("Transit Depth", 500.0),
    st.number_input("Planet Radius", 2.0),
    st.number_input("Equilibrium Temperature", 700.0),
    st.number_input("Insolation Flux", 100.0),
    st.number_input("Signal-to-Noise Ratio", 20.0),
    st.number_input("Stellar Temperature", 5500.0),
    st.number_input("Stellar Gravity", 4.4),
    st.number_input("Stellar Radius", 1.0),
]])


# ============================================================
# SCALE
# ============================================================

features = scaler.transform(features)


# ============================================================
# PREDICT
# ============================================================

if st.button("Predict"):

    probs = model.predict(features)
    pred = np.argmax(probs)

    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"Prediction: {label}")

    st.subheader("Probabilities")

    for cls, p in zip(label_encoder.classes_, probs[0]):
        st.write(f"{cls}: {p:.4f}")
