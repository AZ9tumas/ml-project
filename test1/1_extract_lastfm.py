"""
1_extract_lastfm.py — Pull full scrobble history from Last.fm API

This script fetches a user's recent tracks (scrobble history) from the
Last.fm API and saves them to data/scrobbles.csv.

Usage:
    1. Copy .env.template to .env and fill in your Last.fm credentials
    2. Run: python 1_extract_lastfm.py
"""

import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LASTFM_API_KEY")
USER = os.getenv("LASTFM_USERNAME")


def get_scrobbles(user, api_key, pages=10):
    """Fetch scrobble history from Last.fm API, page by page."""
    all_tracks = []
    for page in range(1, pages + 1):
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "user.getrecenttracks",
            "user": user,
            "api_key": api_key,
            "format": "json",
            "limit": 200,
            "page": page,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        tracks = data["recenttracks"]["track"]

        for t in tracks:
            # Skip currently playing track (has no timestamp)
            if "@attr" in t and t["@attr"].get("nowplaying"):
                continue
            all_tracks.append(
                {
                    "artist": t["artist"]["#text"],
                    "track": t["name"],
                    "album": t["album"]["#text"],
                    "timestamp": int(t["date"]["uts"]),
                }
            )
        time.sleep(0.25)  # Respect rate limits
        print(f"Page {page} done — {len(all_tracks)} tracks so far")
    return pd.DataFrame(all_tracks)


def main():
    if not API_KEY or not USER:
        print("ERROR: Set LASTFM_API_KEY and LASTFM_USERNAME in your .env file")
        return

    os.makedirs("data", exist_ok=True)

    # 50 pages x 200 tracks/page = up to 10,000 scrobbles
    df = get_scrobbles(USER, API_KEY, pages=50)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.to_csv("data/scrobbles.csv", index=False)
    print(f"Saved {len(df)} scrobbles to data/scrobbles.csv")


if __name__ == "__main__":
    main()
