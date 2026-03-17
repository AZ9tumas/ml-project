import numpy as np

def create_sequences(df, seq_len=5):
    X, y = [], []

    data = df[['vibe_encoded','hour','day']].values

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])

    return np.array(X), np.array(y)