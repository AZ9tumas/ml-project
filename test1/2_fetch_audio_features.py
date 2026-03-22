"""
2_fetch_audio_features.py — Enrich scrobbles with Spotify audio features

Searches Spotify for each unique (artist, track) pair from the scrobble
history, then fetches audio features (valence, energy, danceability, etc.)
and saves them to data/audio_features.csv.

Usage:
    1. Ensure .env has SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET
    2. Run: python 2_fetch_audio_features.py
"""

import spotipy
import pandas as pd
import time
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())


def get_features(artist, track):
    """Search Spotify for a track and return its audio features."""
    query = f"track:{track} artist:{artist}"
    try:
        results = sp.search(q=query, type="track", limit=1)
        items = results["tracks"]["items"]
        if not items:
            return None
        track_id = items[0]["id"]
        feats = sp.audio_features([track_id])[0]
        if feats:
            feats["artist"] = artist
            feats["track"] = track
        return feats
    except Exception as e:
        print(f"  Error fetching '{track}' by '{artist}': {e}")
        return None


def main():
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv("data/scrobbles.csv")
    unique_tracks = df[["artist", "track"]].drop_duplicates()
    print(f"Fetching audio features for {len(unique_tracks)} unique tracks...")

    features = []
    for i, row in unique_tracks.iterrows():
        f = get_features(row["artist"], row["track"])
        if f:
            features.append(f)
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(unique_tracks)} done ({len(features)} found)")
        time.sleep(0.1)  # Respect Spotify rate limits

    feat_df = pd.DataFrame(features)
    feat_df.to_csv("data/audio_features.csv", index=False)
    print(
        f"Saved audio features for {len(feat_df)} tracks to data/audio_features.csv"
    )
    print(
        f"Match rate: {len(feat_df)}/{len(unique_tracks)} "
        f"({100*len(feat_df)/len(unique_tracks):.1f}%)"
    )


if __name__ == "__main__":
    main()
