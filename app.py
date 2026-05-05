Rebuild the architecture (takes 1–2 minutes)--import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# =========================
# PATHS
# =========================
MODEL_PATH = "models/exoplanet_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

# =========================
# UI
# =========================
st.title("🪐 Exoplanet Classifier")

st.write("Enter feature values:")

inputs = []
for i in range(11):
    inputs.append(st.number_input(f"Feature {i+1}", value=0.0))

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    X = scaler.transform(X)

    pred = model.predict(X)
    class_id = np.argmax(pred)

    label = encoder.inverse_transform([class_id])[0]

    st.success(f"Prediction: {label}")
