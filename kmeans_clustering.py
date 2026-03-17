import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load SVD output
df = pd.read_csv("svd_output.csv")

# 2. Select features
features = ['svd_1', 'svd_2', 'svd_3']
X = df[features]

# 3. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Map clusters to vibes
vibe_map = {
    0: "Focus State",
    1: "Peak State",
    2: "Wind-down State"
}

df['vibe'] = df['cluster'].map(vibe_map)

# 6. Save result
df.to_csv("clustered_vibes.csv", index=False)

print("K-Means clustering complete ✅")
print(df.head())