"""
1b_convert_existing_data.py — Convert existing fm_data.csv to the scrobbles format

If you already have Last.fm data exported as fm_data.csv (with columns:
Username, Artist, Track, Album, Date, Time), this script converts it
to the standard scrobbles.csv format used by the rest of the pipeline.

This is an alternative to 1_extract_lastfm.py — use one or the other.

Usage:
    python 1b_convert_existing_data.py
"""

import pandas as pd
import os


def main():
    os.makedirs("data", exist_ok=True)

    # Read existing fm_data.csv (has an unnamed index column)
    df = pd.read_csv("fm_data.csv", index_col=0)

    # Combine Date and Time into a single datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(),
        format="%d %b %Y %H:%M",
    )

    # Convert to unix timestamp for consistency
    df["timestamp"] = df["datetime"].astype("int64") // 10**9

    # Rename to standard column names (lowercase)
    result = df.rename(
        columns={"Artist": "artist", "Track": "track", "Album": "album"}
    )[["artist", "track", "album", "timestamp", "datetime"]]

    result = result.sort_values("datetime").reset_index(drop=True)
    result.to_csv("data/scrobbles.csv", index=False)
    print(f"Converted {len(result)} scrobbles to data/scrobbles.csv")


if __name__ == "__main__":
    main()
