import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Paths
# =========================
MODEL_PATH = "models/exoplanet_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

os.makedirs("models", exist_ok=True)

# =========================
# Load Dataset
# =========================
def load_data():
    df = pd.read_csv("data/exoplanet_data.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y

# =========================
# Preprocessing
# =========================
def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded, scaler, encoder

# =========================
# Build Model
# =========================
def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.35),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =========================
# Training
# =========================
def train():
    print("Loading data...")
    X, y = load_data()

    print("Preprocessing...")
    X, y, scaler, encoder = preprocess(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building model...")
    model = build_model(X.shape[1])

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32
    )

    print("Saving EVERYTHING (correct way)...")

    # ✅ SAVE FULL MODEL (NO JSON, NO WEIGHTS ONLY)
    model.save(MODEL_PATH)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("✅ DONE — No more loading issues.")

# =========================
# Run
# =========================
if __name__ == "__main__":
    train()
