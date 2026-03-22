import pandas as pd
from sklearn.decomposition import TruncatedSVD

# 1. Load dataset
df = pd.read_csv("fm_data.csv")


# 2. Create User-Track Matrix
user_track_matrix = df.pivot_table(
    index='Track',
    columns='Username',
    aggfunc='size',
    fill_value=0
)

# 3. Apply SVD
svd = TruncatedSVD(n_components=3, random_state=42)
X_svd = svd.fit_transform(user_track_matrix)

# 4. Save SVD output
svd_df = pd.DataFrame(X_svd, columns=['svd_1', 'svd_2', 'svd_3'])
svd_df['Track'] = user_track_matrix.index

svd_df.to_csv("svd_output.csv", index=False)

print("SVD processing complete ✅")
print(svd_df.head())