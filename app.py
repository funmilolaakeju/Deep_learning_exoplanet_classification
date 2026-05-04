import os
import numpy as np
import streamlit as st
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def build_mlp_model(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        Dropout(0.35),

        Dense(128, activation="relu"),
        Dropout(0.30),

        Dense(64, activation="relu"),
        Dropout(0.20),

        Dense(num_classes, activation="softmax"),
    ])

    return model


model = build_mlp_model(input_dim=11, num_classes=3)

model.load_weights(
    os.path.join(
        MODEL_DIR,
        "exoplanet_mlp_weights.weights.h5"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

label_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


st.subheader("Enter Scientific Features")

koi_period = st.number_input("Orbital Period", value=10.0)
koi_impact = st.number_input("Impact Parameter", value=0.5)
koi_duration = st.number_input("Transit Duration", value=5.0)
koi_depth = st.number_input("Transit Depth", value=500.0)
koi_prad = st.number_input("Planet Radius", value=2.0)
koi_teq = st.number_input("Equilibrium Temperature", value=700.0)
koi_insol = st.number_input("Insolation Flux", value=100.0)
koi_model_snr = st.number_input("Model Signal-to-Noise Ratio", value=20.0)
koi_steff = st.number_input("Stellar Effective Temperature", value=5500.0)
koi_slogg = st.number_input("Stellar Surface Gravity", value=4.4)
koi_srad = st.number_input("Stellar Radius", value=1.0)


features = np.array([
    [
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
        koi_srad,
    ]
])


features_scaled = scaler.transform(features)


if st.button("Predict"):
    probabilities = model.predict(features_scaled)

    predicted_index = np.argmax(probabilities)

    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    st.success(f"Prediction: {predicted_class}")

    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):
        st.write(f"{class_name}: {probability:.4f}")