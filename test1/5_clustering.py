"""
5_clustering.py — Vibe Clustering using K-Means

Clusters scrobbles into 'vibe' groups based on audio features using K-Means.
Includes elbow method for choosing k, PCA visualization, and cluster profiling.

Outputs:
  - data/clustered_dataset.csv
  - models/kmeans.pkl, models/scaler.pkl
  - plots/elbow.png, plots/clusters_pca.png, plots/cluster_profiles.png

Usage:
    python 5_clustering.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("data/master_dataset.csv")
    print(f"Loaded {len(df)} rows for clustering")

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow Method to find optimal k ---
    inertias = []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(K_range), inertias, marker="o", linewidth=2)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method — Choose Optimal k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/elbow.png", dpi=150)
    plt.close()
    print("Saved plots/elbow.png")

    # --- Fit K-Means with chosen k ---
    # Start with k=5, adjust after inspecting the elbow plot
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    df["vibe_cluster"] = kmeans.fit_predict(X_scaled)

    # --- Inspect cluster centroids ---
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURE_COLS
    )
    print("\nCluster Centroids (original scale):")
    print(centroids.round(3).to_string())

    # --- Name clusters based on centroid profiles ---
    # Auto-naming heuristic based on feature values
    vibe_names = {}
    for i in range(K):
        c = centroids.iloc[i]
        if c["energy"] > centroids["energy"].median() and c["valence"] > centroids["valence"].median():
            if c["danceability"] > centroids["danceability"].median():
                vibe_names[i] = "Party Mode"
            else:
                vibe_names[i] = "Peak Energy"
        elif c["acousticness"] > centroids["acousticness"].median() and c["energy"] < centroids["energy"].median():
            vibe_names[i] = "Chill Vibes"
        elif c["instrumentalness"] > centroids["instrumentalness"].median():
            vibe_names[i] = "Deep Focus"
        elif c["valence"] < centroids["valence"].median() and c["energy"] < centroids["energy"].median():
            vibe_names[i] = "Wind-down"
        else:
            vibe_names[i] = f"Vibe {i}"

    # De-duplicate names if the heuristic assigns the same name
    seen = set()
    for i in sorted(vibe_names):
        if vibe_names[i] in seen:
            vibe_names[i] = f"{vibe_names[i]} {i}"
        seen.add(vibe_names[i])

    print(f"\nVibe Names: {vibe_names}")
    df["vibe_label"] = df["vibe_cluster"].map(vibe_names)

    # --- PCA Visualization ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"] = X_pca[:, 0], X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    for label in df["vibe_label"].unique():
        sub = df[df["vibe_label"] == label]
        plt.scatter(sub["pc1"], sub["pc2"], label=label, alpha=0.4, s=10)
    plt.legend(fontsize=10)
    plt.title("Vibe Clusters (PCA Projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("plots/clusters_pca.png", dpi=150)
    plt.close()
    print("Saved plots/clusters_pca.png")

    # --- Cluster Profile Radar Chart ---
    norm_centroids = (centroids - centroids.min()) / (centroids.max() - centroids.min())
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        vals = norm_centroids.iloc[i].values
        ax.bar(range(len(FEATURE_COLS)), vals, color=f"C{i}", alpha=0.7)
        ax.set_xticks(range(len(FEATURE_COLS)))
        ax.set_xticklabels([c[:5] for c in FEATURE_COLS], rotation=45, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(vibe_names[i], fontsize=10)
    plt.suptitle("Cluster Profiles (Normalized)", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/cluster_profiles.png", dpi=150)
    plt.close()
    print("Saved plots/cluster_profiles.png")

    # --- Save outputs ---
    df.to_csv("data/clustered_dataset.csv", index=False)
    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"\nSaved clustered dataset: {len(df)} rows")
    print("Saved models/kmeans.pkl and models/scaler.pkl")
    print("\nCluster distribution:")
    print(df["vibe_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
