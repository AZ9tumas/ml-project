"""
6_train_lstm.py — Train LSTM model for vibe forecasting

Builds sequences from the chronologically-ordered scrobble history and
trains an LSTM neural network to predict the next 'vibe' (cluster label)
given the previous SEQ_LEN listening events + temporal features.

Outputs:
  - models/lstm_vibe.h5
  - models/label_encoder.pkl
  - plots/training_history.png

Usage:
    python 6_train_lstm.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Sequence length: how many past events to use for prediction
SEQ_LEN = 20


def build_sequences(df, seq_len):
    """Build (X, y) sequence pairs for LSTM training.

    X: sequences of [vibe_id, hour, day_of_week, is_weekend]
    y: the next vibe_id after each sequence
    """
    X, y = [], []
    feats = df[["vibe_id", "hour", "day_of_week", "is_weekend"]].values
    for i in range(len(feats) - seq_len):
        X.append(feats[i : i + seq_len])
        y.append(feats[i + seq_len, 0])  # Next vibe_id
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("data/clustered_dataset.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Encode vibe labels as integers
    le = LabelEncoder()
    df["vibe_id"] = le.fit_transform(df["vibe_label"])
    NUM_CLASSES = df["vibe_id"].nunique()
    print(f"Number of vibe classes: {NUM_CLASSES}")
    print(f"Classes: {list(le.classes_)}")

    # Build sequences
    X, y = build_sequences(df, SEQ_LEN)
    print(f"Sequences built: X={X.shape}, y={y.shape}")

    # Time-series split (no shuffle — preserve temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Build LSTM model
    model = Sequential(
        [
            LSTM(64, input_shape=(SEQ_LEN, X.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Train
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.set_xlabel("Epoch")

    ax2.plot(history.history["accuracy"], label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("plots/training_history.png", dpi=150)
    plt.close()
    print("Saved plots/training_history.png")

    # Evaluate
    y_pred = model.predict(X_test).argmax(axis=1)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0
        )
    )

    # Save model and encoder
    model.save("models/lstm_vibe.h5")
    joblib.dump(le, "models/label_encoder.pkl")
    print("\nModel saved to models/lstm_vibe.h5")
    print("Label encoder saved to models/label_encoder.pkl")


if __name__ == "__main__":
    main()
