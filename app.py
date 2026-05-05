import streamlit as st
import numpy as np
import joblib
import json
import tensorflow as tf

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🪐",
    layout="centered"
)

st.title("🪐 Deep Learning Exoplanet Classification")


# =========================
# PATHS
# =========================
ARCH_PATH = "models/model_architecture.json"
WEIGHTS_PATH = "models/exoplanet_weights.h5"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():

    # Load model architecture
    with open(ARCH_PATH, "r") as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)

    # Load weights
    model.load_weights(WEIGHTS_PATH)

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()


# =========================
# INPUT SECTION
# =========================
st.subheader("Enter Input Features")

f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

features = np.array([[f1, f2, f3, f4]])


# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    class_id = np.argmax(prediction, axis=1)

    result = label_encoder.inverse_transform(class_id)

    st.success(f"Prediction: {result[0]}")
