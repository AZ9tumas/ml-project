"""
3_build_dataset.py — Feature Engineering: Merge scrobbles with audio features

Merges the scrobble history with Spotify audio features and adds
temporal features (hour, day_of_week, month, is_weekend).
Outputs data/master_dataset.csv.

Usage:
    python 3_build_dataset.py
"""

import pandas as pd
import os

# Audio features used for vibe clustering
FEATURE_COLS = [
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "tempo",
    "loudness",
]


def main():
    os.makedirs("data", exist_ok=True)

    scrobbles = pd.read_csv("data/scrobbles.csv", parse_dates=["datetime"])
    features = pd.read_csv("data/audio_features.csv")

    # Merge on artist + track name
    df = scrobbles.merge(features, on=["artist", "track"], how="inner")
    print(f"Merged dataset: {len(df)} rows ({len(scrobbles)} scrobbles matched)")

    # Add temporal features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Drop rows missing any key audio features
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing audio features")

    df.to_csv("data/master_dataset.csv", index=False)
    print(f"Master dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Unique artists: {df['artist'].nunique()}")
    print(f"Unique tracks: {df['track'].nunique()}")


if __name__ == "__main__":
    main()
