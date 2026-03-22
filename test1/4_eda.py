"""
4_eda.py — Exploratory Data Analysis

Generates visualizations of listening patterns:
  - Audio feature correlation heatmap
  - Average energy/valence by hour of day
  - Energy heatmap by day-of-week x hour

Outputs saved to plots/ directory.

Usage:
    python 4_eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURE_COLS = [
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
]


def main():
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("data/master_dataset.csv")
    print(f"Loaded {len(df)} rows for EDA")

    # 1. Correlation heatmap of audio features
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[FEATURE_COLS].corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0
    )
    plt.title("Audio Feature Correlations")
    plt.tight_layout()
    plt.savefig("plots/correlation.png", dpi=150)
    plt.close()
    print("Saved plots/correlation.png")

    # 2. Average energy and valence by hour of day
    hourly = df.groupby("hour")[["energy", "valence"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    hourly.plot(ax=ax, marker="o")
    ax.set_title("Average Energy & Valence by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Feature Value")
    ax.set_xticks(range(24))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/hourly_trends.png", dpi=150)
    plt.close()
    print("Saved plots/hourly_trends.png")

    # 3. Energy heatmap: day-of-week x hour
    pivot = df.groupby(["day_of_week", "hour"])["energy"].mean().unstack()
    plt.figure(figsize=(12, 4))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(pivot, cmap="YlOrRd", xticklabels=True, yticklabels=days)
    plt.title("Average Energy by Day of Week × Hour")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    plt.tight_layout()
    plt.savefig("plots/energy_heatmap.png", dpi=150)
    plt.close()
    print("Saved plots/energy_heatmap.png")

    # 4. Distribution of audio features
    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(16, 3))
    for ax, col in zip(axes, FEATURE_COLS):
        df[col].hist(ax=ax, bins=30, alpha=0.7, edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel("")
    plt.suptitle("Audio Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=150)
    plt.close()
    print("Saved plots/feature_distributions.png")

    # 5. Top 15 most-listened artists
    top_artists = df["artist"].value_counts().head(15)
    plt.figure(figsize=(10, 5))
    top_artists.plot(kind="barh")
    plt.title("Top 15 Most-Listened Artists")
    plt.xlabel("Number of Scrobbles")
    plt.tight_layout()
    plt.savefig("plots/top_artists.png", dpi=150)
    plt.close()
    print("Saved plots/top_artists.png")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total scrobbles: {len(df)}")
    print(f"Unique artists: {df['artist'].nunique()}")
    print(f"Unique tracks: {df['track'].nunique()}")
    print(f"\nAudio Feature Means:")
    print(df[FEATURE_COLS].mean().to_string())


if __name__ == "__main__":
    main()
