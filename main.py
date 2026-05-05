import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/exoplanet.csv")

X = df.drop("target", axis=1).values
y = df["target"].values

# =========================
# LABEL ENCODING
# =========================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# BUILD MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.35),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# =========================
# SAVE EVERYTHING (FIXED PART)
# =========================
import os
os.makedirs("models", exist_ok=True)

# ✅ IMPORTANT FIX: modern format only
model.save("models/exoplanet_model.keras", include_optimizer=False)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("ALL ARTIFACTS SAVED CORRECTLY")
