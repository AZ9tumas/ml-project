"""
2b_synthetic_audio_features.py — Generate synthetic audio features for testing

When Spotify API credentials are not available, this script generates
realistic synthetic audio features for each unique track in the scrobble
history. This allows the rest of the pipeline (clustering, LSTM, dashboard)
to run end-to-end for testing/demo purposes.

The synthetic features are seeded by hashing (artist + track) so they are
deterministic — the same song always gets the same features.

Usage:
    python 2b_synthetic_audio_features.py
"""

import pandas as pd
import numpy as np
import hashlib
import os


def generate_features_for_track(artist, track):
    """Generate deterministic synthetic audio features from artist+track name."""
    # Use a hash so the same song always gets the same features
    seed_str = f"{artist}::{track}"
    hash_bytes = hashlib.sha256(seed_str.encode()).digest()
    # Use first 8 bytes as a seed for numpy
    seed = int.from_bytes(hash_bytes[:8], "big") % (2**31)
    rng = np.random.RandomState(seed)

    return {
        "artist": artist,
        "track": track,
        "valence": rng.beta(2, 2),          # 0-1, centered
        "energy": rng.beta(2, 2),            # 0-1, centered
        "danceability": rng.beta(2.5, 2),    # 0-1, slightly skewed high
        "acousticness": rng.beta(1.5, 3),    # 0-1, skewed low
        "instrumentalness": rng.beta(1, 5),  # 0-1, mostly low
        "tempo": rng.normal(120, 25),        # BPM, ~70-170
        "loudness": rng.normal(-8, 4),       # dB, typically -2 to -14
        "speechiness": rng.beta(1, 5),       # 0-1, mostly low
        "liveness": rng.beta(1.5, 5),        # 0-1, mostly low
        "key": rng.randint(0, 12),
        "mode": rng.randint(0, 2),
        "time_signature": rng.choice([3, 4, 5], p=[0.1, 0.8, 0.1]),
        "duration_ms": int(rng.normal(220000, 60000)),
    }


def main():
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv("data/scrobbles.csv")
    unique_tracks = df[["artist", "track"]].drop_duplicates()
    print(f"Generating synthetic audio features for {len(unique_tracks)} unique tracks...")

    features = []
    for _, row in unique_tracks.iterrows():
        features.append(generate_features_for_track(row["artist"], row["track"]))

    feat_df = pd.DataFrame(features)

    # Clamp values to valid ranges
    for col in ["valence", "energy", "danceability", "acousticness",
                "instrumentalness", "speechiness", "liveness"]:
        feat_df[col] = feat_df[col].clip(0, 1)
    feat_df["tempo"] = feat_df["tempo"].clip(40, 220)
    feat_df["loudness"] = feat_df["loudness"].clip(-20, 0)
    feat_df["duration_ms"] = feat_df["duration_ms"].clip(60000, 600000)

    feat_df.to_csv("data/audio_features.csv", index=False)
    print(f"Saved synthetic audio features for {len(feat_df)} tracks to data/audio_features.csv")
    print("NOTE: These are synthetic features for testing. Use 2_fetch_audio_features.py with real Spotify credentials for production.")


if __name__ == "__main__":
    main()
