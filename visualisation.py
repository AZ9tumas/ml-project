import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 🔹 Load & prepare data
def load_data():
    df = pd.read_csv("fm_data.csv")
    vibes = pd.read_csv("clustered_vibes.csv")

    # Merge
    df = df.merge(vibes[['Track','vibe']], on='Track', how='left')

    # Fix datetime
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.strip(),
        format="%d %b %Y %H:%M"
    )

    df = df.sort_values('datetime')

    # Features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek

    # Encode vibe
    df['vibe_code'] = df['vibe'].astype('category').cat.codes

    return df


# 🔹 1. Smoothed Graph
def plot_smoothed(df):
    df['vibe_smooth'] = df['vibe_code'].rolling(window=50).mean()

    plt.figure()
    plt.plot(df['datetime'], df['vibe_smooth'])
    plt.title("Smoothed Vibe Trend")
    plt.xlabel("Time")
    plt.ylabel("Vibe (Smoothed)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 🔹 2. Hourly Trend (VERY CLEAN)
def plot_hourly(df):
    hourly = df.groupby('hour')['vibe_code'].mean()

    plt.figure()
    hourly.plot()
    plt.title("Average Vibe by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Vibe")
    plt.show()


# 🔹 3. Daily Trend
def plot_daily(df):
    daily = df.resample('D', on='datetime')['vibe_code'].mean()

    plt.figure()
    daily.plot()
    plt.title("Daily Vibe Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Vibe")
    plt.show()


# 🔥 4. Heatmap (BEST GRAPH)
def plot_heatmap(df):
    pivot = df.pivot_table(
        index='day',
        columns='hour',
        values='vibe_code',
        aggfunc='mean'
    )

    plt.figure()
    sns.heatmap(pivot)
    plt.title("Vibe Pattern (Day vs Hour)")
    plt.xlabel("Hour")
    plt.ylabel("Day (0=Mon)")
    plt.show()


# ▶️ Run all
if __name__ == "__main__":
    df = load_data()

    print("Data Loaded ✅")
    print(df.head())

    plot_smoothed(df)
    plot_hourly(df)
    plot_daily(df)
    plot_heatmap(df)