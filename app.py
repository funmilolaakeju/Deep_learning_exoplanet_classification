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

    # =========================
    # REBUILT MODEL ARCHITECTURE
    # =========================
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
