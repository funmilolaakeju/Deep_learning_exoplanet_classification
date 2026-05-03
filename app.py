import os
import joblib
import tensorflow as tf

MODEL_DIR = "models"

# Load model safely
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "exoplanet_mlp_model.keras"),
    compile=False,
    safe_mode=False
)

# Load preprocessors
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
