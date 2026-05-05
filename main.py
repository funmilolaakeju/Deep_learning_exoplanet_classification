import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from urllib.parse import quote
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report


# =========================
# CONFIG
# =========================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_COLUMNS = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

TARGET_COLUMN = "koi_disposition"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =========================
# DATA
# =========================

def load_data():
    query = f"""
    SELECT {', '.join([TARGET_COLUMN] + FEATURE_COLUMNS)}
    FROM cumulative
    WHERE koi_disposition IS NOT NULL
    """

    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query={quote(query)}&format=csv"
    )

    df = pd.read_csv(url).dropna()
    return df


# =========================
# PREPROCESS
# =========================

def preprocess(df):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, encoder


# =========================
# MODEL
# =========================

def build_model(input_dim, num_classes):

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation="relu"),
        Dropout(0.35),

        Dense(128, activation="relu"),
        Dropout(0.30),

        Dense(64, activation="relu"),
        Dropout(0.20),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN
# =========================

def train():

    df = load_data()
    X, y, scaler, encoder = preprocess(df)

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    model = build_model(X.shape[1], len(np.unique(y)))

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    preds = np.argmax(model.predict(X_test), axis=1)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # =========================
    # SAVE (FINAL SAFE FORMAT)
    # =========================

    with open(MODEL_DIR / "model_architecture.json", "w") as f:
        f.write(model.to_json())

    model.save_weights(MODEL_DIR / "exoplanet_weights.h5")

    print("\n✅ TRAINING COMPLETE (SAFE FORMAT SAVED)")


if __name__ == "__main__":
    train()
