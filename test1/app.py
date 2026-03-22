"""
app.py — Streamlit Dashboard for Vibe Forecaster

Interactive dashboard that shows:
  - Current vibe prediction with confidence
  - Vibe probability bar chart
  - Weekly vibe heatmap (day x hour)
  - Cluster profile explorer
  - Listening history timeline

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Vibe Forecaster", layout="wide")

SEQ_LEN = 20


@st.cache_resource
def load_models():
    """Load all saved models (cached for performance)."""
    return (
        joblib.load("models/kmeans.pkl"),
        joblib.load("models/scaler.pkl"),
        joblib.load("models/label_encoder.pkl"),
        load_model("models/lstm_vibe.h5"),
    )


@st.cache_data
def load_data():
    """Load the clustered dataset."""
    return pd.read_csv("data/clustered_dataset.csv", parse_dates=["datetime"])


# ---- Load resources ----
try:
    kmeans, scaler, le, model = load_models()
    df = load_data()
except Exception as e:
    st.error(
        f"Error loading models/data: {e}\n\n"
        "Make sure you've run scripts 1-6 first to generate the data and models."
    )
    st.stop()

st.title("🎵 Vibe Forecaster")
st.markdown("*Predicting your music mood based on listening history*")

# ---- Sidebar ----
st.sidebar.header("About")
st.sidebar.markdown(
    f"""
- **Total scrobbles:** {len(df):,}
- **Unique artists:** {df['artist'].nunique():,}
- **Unique tracks:** {df['track'].nunique():,}
- **Date range:** {df['datetime'].min().strftime('%b %Y')} — {df['datetime'].max().strftime('%b %Y')}
- **Vibe clusters:** {df['vibe_label'].nunique()}
"""
)

# ---- Current Prediction ----
st.header("Current Vibe Prediction")
col1, col2 = st.columns([1, 2])

with col1:
    now = datetime.now()
    recent = df.sort_values("datetime").tail(SEQ_LEN)
    recent_ids = le.transform(recent["vibe_label"].values)
    seq = [
        [
            vid,
            recent.iloc[i]["hour"],
            recent.iloc[i]["day_of_week"],
            recent.iloc[i]["is_weekend"],
        ]
        for i, vid in enumerate(recent_ids)
    ]
    probs = model.predict(np.array([seq], dtype=np.float32), verbose=0)[0]
    top_vibe = le.inverse_transform([np.argmax(probs)])[0]
    confidence = probs.max() * 100

    st.metric("Predicted Vibe", top_vibe, f"{confidence:.0f}% confidence")
    st.caption(now.strftime("%A, %I:%M %p"))

    # Show all probabilities as text
    st.markdown("**All vibe probabilities:**")
    for vibe, prob in sorted(
        zip(le.classes_, probs), key=lambda x: x[1], reverse=True
    ):
        st.text(f"  {vibe}: {prob*100:.1f}%")

with col2:
    st.subheader("Vibe Probabilities")
    prob_df = pd.DataFrame({"Vibe": le.classes_, "Probability": probs * 100})
    fig = px.bar(
        prob_df.sort_values("Probability", ascending=True),
        x="Probability",
        y="Vibe",
        orientation="h",
        color="Probability",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

# ---- Weekly Heatmap ----
st.header("Your Vibe Through the Week")
pivot = (
    df.groupby(["day_of_week", "hour"])["vibe_cluster"]
    .agg(lambda x: x.value_counts().index[0])
    .unstack()
)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig2 = px.imshow(
    pivot,
    labels={"x": "Hour", "y": "Day", "color": "Cluster"},
    y=days,
    color_continuous_scale="Plasma",
    title="Most Common Vibe Cluster by Day & Hour",
)
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

# ---- Cluster Explorer ----
st.header("Vibe Cluster Profiles")
feature_cols = [
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
]
cluster_stats = (
    df.groupby("vibe_label")[feature_cols].mean().reset_index()
)
fig3 = px.bar(
    cluster_stats.melt(id_vars="vibe_label"),
    x="variable",
    y="value",
    color="vibe_label",
    barmode="group",
    labels={"variable": "Audio Feature", "value": "Mean Value", "vibe_label": "Vibe"},
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

# ---- Cluster Distribution ----
st.header("Listening Distribution by Vibe")
col3, col4 = st.columns(2)

with col3:
    vibe_counts = df["vibe_label"].value_counts().reset_index()
    vibe_counts.columns = ["Vibe", "Count"]
    fig4 = px.pie(vibe_counts, values="Count", names="Vibe", title="Vibe Distribution")
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    # Listening activity by hour
    hourly_counts = df.groupby("hour").size().reset_index(name="Scrobbles")
    fig5 = px.bar(
        hourly_counts,
        x="hour",
        y="Scrobbles",
        title="Listening Activity by Hour",
        labels={"hour": "Hour of Day"},
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.caption("Vibe Forecaster — Built with Last.fm data, Spotify audio features, K-Means clustering, and LSTM prediction")
