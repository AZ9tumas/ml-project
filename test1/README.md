# Vibe Forecaster — Test Implementation

A music mood prediction system that analyzes Last.fm listening history, enriches it with Spotify audio features, clusters songs into "vibe" groups using K-Means, and trains an LSTM neural network to forecast your music mood.

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your Spotify and Last.fm API credentials
```

### 2. Run the Pipeline (in order)

**Option A — Fetch fresh data from Last.fm:**
```bash
python 1_extract_lastfm.py
```

**Option B — Use existing fm_data.csv:**
```bash
python 1b_convert_existing_data.py
```

**Then continue with:**
```bash
python 2_fetch_audio_features.py   # Enrich with Spotify features
python 3_build_dataset.py          # Merge & feature engineering
python 4_eda.py                    # Generate exploration plots
python 5_clustering.py             # K-Means vibe clustering
python 6_train_lstm.py             # Train LSTM forecaster
```

### 3. Predict
```bash
python 7_predict.py                # CLI prediction
streamlit run app.py               # Interactive dashboard
```

## Pipeline Overview

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `requirements.txt`, `.env` | Environment setup |
| 1 | `1_extract_lastfm.py` | Pull scrobble history from Last.fm |
| 1b | `1b_convert_existing_data.py` | Convert existing fm_data.csv |
| 1 | `2_fetch_audio_features.py` | Enrich with Spotify audio features |
| 2 | `3_build_dataset.py` | Merge + temporal feature engineering |
| 3 | `4_eda.py` | Exploratory data analysis & plots |
| 4 | `5_clustering.py` | K-Means vibe clustering (k=5) |
| 5 | `6_train_lstm.py` | LSTM sequence model training |
| 6 | `7_predict.py` | CLI vibe prediction |
| 7 | `app.py` | Streamlit interactive dashboard |

## Audio Features Used

- **valence**: Musical positiveness (0–1)
- **energy**: Intensity and activity (0–1)
- **danceability**: Dance suitability (0–1)
- **acousticness**: Acoustic confidence (0–1)
- **instrumentalness**: Vocal absence prediction (0–1)
- **tempo**: Estimated BPM
- **loudness**: Overall loudness (dB)

## Vibe Clusters

The K-Means algorithm groups songs into 5 interpretable vibes:
- **Party Mode**: High energy, high valence, high danceability
- **Peak Energy**: High energy, high valence
- **Chill Vibes**: High acousticness, low energy
- **Deep Focus**: High instrumentalness
- **Wind-down**: Low valence, low energy

## Documentation

Full implementation report available at `docs/implementation_report.tex`.

## Tips

- **Rate limits**: Scripts include built-in delays for API calls
- **Elbow method**: Check `plots/elbow.png` and adjust k in `5_clustering.py` if needed
- **Data size**: LSTM works best with 5,000+ scrobbles; reduce `SEQ_LEN` if data is sparse
- **Cluster inspection**: After clustering, review centroids to verify vibe names make sense
