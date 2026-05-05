import os
import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

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
# FIXED PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "exoplanet_model_fixed.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

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
