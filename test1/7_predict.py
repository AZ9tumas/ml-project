"""
7_predict.py — Vibe Forecasting: Predict current listening mood

Uses the trained LSTM model to predict what 'vibe' of music you're
most likely to want right now, based on your recent listening history
and the current time.

Usage:
    python 7_predict.py
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model

SEQ_LEN = 20


def predict_now():
    """Predict the most likely vibe for the current moment."""
    # Load models and data
    kmeans = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")
    model = load_model("models/lstm_vibe.h5")
    df = pd.read_csv("data/clustered_dataset.csv", parse_dates=["datetime"])

    now = datetime.now()
    hour = now.hour
    dow = now.weekday()
    is_weekend = int(dow in [5, 6])

    # Get last SEQ_LEN vibes from history
    recent = df.sort_values("datetime").tail(SEQ_LEN)
    recent_vibe_ids = le.transform(recent["vibe_label"].values)

    seq = []
    for i, vid in enumerate(recent_vibe_ids):
        seq.append(
            [
                vid,
                recent.iloc[i]["hour"],
                recent.iloc[i]["day_of_week"],
                recent.iloc[i]["is_weekend"],
            ]
        )

    X = np.array([seq], dtype=np.float32)  # shape: (1, SEQ_LEN, 4)
    probs = model.predict(X, verbose=0)[0]
    top_vibe = le.inverse_transform([np.argmax(probs)])[0]
    confidence = probs.max() * 100

    print(f"\n{'='*50}")
    print(f"  VIBE FORECASTER")
    print(f"{'='*50}")
    print(f"\n  It's {now.strftime('%A')} at {now.strftime('%I:%M %p')}")
    print(
        f"  You are {confidence:.0f}% likely to want '{top_vibe}' music\n"
    )

    print("  Vibe Probabilities:")
    print(f"  {'─'*40}")
    for vibe, prob in sorted(
        zip(le.classes_, probs), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 30)
        marker = " ◄" if vibe == top_vibe else ""
        print(f"  {vibe:<15} {bar} {prob*100:.1f}%{marker}")

    print(f"\n{'='*50}")
    return top_vibe, probs


if __name__ == "__main__":
    predict_now()
