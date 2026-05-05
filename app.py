import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# =========================
# PATHS
# =========================
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    num_classes = len(encoder.classes_)

    # Rebuilt architecture (must match training)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

# =========================
# UI
# =========================
st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐")

st.title("🪐 Deep Learning Exoplanet Classification")
st.write("Predict whether a Kepler Object of Interest is CANDIDATE, CONFIRMED, or FALSE POSITIVE.")

st.subheader("Enter Scientific Features")

# =========================
# FEATURE INPUTS (REAL NAMES)
# =========================
orbital_period = st.number_input("Orbital Period", value=10.0)
impact_parameter = st.number_input("Impact Parameter", value=0.5)
transit_duration = st.number_input("Transit Duration", value=5.0)
transit_depth = st.number_input("Transit Depth", value=500.0)
planet_radius = st.number_input("Planet Radius", value=2.0)
equilibrium_temp = st.number_input("Equilibrium Temperature", value=700.0)
insolation_flux = st.number_input("Insolation Flux", value=100.0)
snr = st.number_input("Signal-to-Noise Ratio", value=20.0)
stellar_temp = st.number_input("Stellar Effective Temperature", value=5500.0)
stellar_gravity = st.number_input("Stellar Surface Gravity", value=4.4)
stellar_radius = st.number_input("Stellar Radius", value=1.0)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    X = np.array([[
        orbital_period,
        impact_parameter,
        transit_duration,
        transit_depth,
        planet_radius,
        equilibrium_temp,
        insolation_flux,
        snr,
        stellar_temp,
        stellar_gravity,
        stellar_radius
    ]])

    X = scaler.transform(X)

    pred = model.predict(X)

    class_id = np.argmax(pred)
    label = encoder.inverse_transform([class_id])[0]

    st.success(f"Prediction: {label}")
