import streamlit as st
import numpy as np
import joblib
import json
import tensorflow as tf

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
# PATHS
# =========================
MODEL_DIR = "models"

ARCH_PATH = f"{MODEL_DIR}/model_architecture.json"
WEIGHTS_PATH = f"{MODEL_DIR}/exoplanet_weights.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():

    with open(ARCH_PATH, "r") as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(WEIGHTS_PATH)

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()


# =========================
# INPUTS (FIXED: 11 FEATURES)
# =========================
st.subheader("Enter Input Features")

f1 = st.number_input("koi_period", value=0.0)
f2 = st.number_input("koi_impact", value=0.0)
f3 = st.number_input("koi_duration", value=0.0)
f4 = st.number_input("koi_depth", value=0.0)
f5 = st.number_input("koi_prad", value=0.0)
f6 = st.number_input("koi_teq", value=0.0)
f7 = st.number_input("koi_insol", value=0.0)
f8 = st.number_input("koi_model_snr", value=0.0)
f9 = st.number_input("koi_steff", value=0.0)
f10 = st.number_input("koi_slogg", value=0.0)
f11 = st.number_input("koi_srad", value=0.0)

features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]])


# =========================
# PREDICT
# =========================
if st.button("Predict"):

    scaled = scaler.transform(features)

    pred = model.predict(scaled)

    class_id = np.argmax(pred, axis=1)

    result = label_encoder.inverse_transform(class_id)

    st.success(f"Prediction: {result[0]}")
