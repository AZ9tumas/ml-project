import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_merge():
    df = pd.read_csv("fm_data.csv")
    vibes = pd.read_csv("clustered_vibes.csv")

    # Merge vibes
    df = df.merge(vibes[['Track','vibe']], on='Track', how='left')

    # Fix datetime (IMPORTANT: space before Time)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.strip())
    df = df.sort_values('datetime')

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek

    # Encode vibe
    le = LabelEncoder()
    df['vibe_encoded'] = le.fit_transform(df['vibe'])

    return df, le