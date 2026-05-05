import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# =========================
# PATHS
# =========================
MODEL_PATH = "models/exoplanet_model.keras"   # may contain full model or weights
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    # Load scaler & encoder first
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    num_classes = len(encoder.classes_)

    # ✅ Rebuild model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # ⚠️ Try loading weights (will fail silently if incompatible)
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print("⚠️ Could not load weights:", e)

    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🪐",
    layout="centered"
)

st.title("🪐 Exoplanet Classifier")
st.write("Enter feature values:")

inputs = []
for i in range(11):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    try:
        X = np.array(inputs).reshape(1, -1)
        X = scaler.transform(X)

        pred = model.predict(X)

        # Handle both binary & multi-class safely
        if pred.shape[1] == 1:
            class_id = int(pred[0] > 0.5)
        else:
            class_id = np.argmax(pred)

        label = encoder.inverse_transform([class_id])[0]

        st.success(f"Prediction: {label}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
