#!/usr/bin/env python3
"""
run_pipeline.py — Execute the full Vibe Forecaster pipeline end-to-end

Runs all steps in sequence using existing data and synthetic audio features
(no API credentials needed). For production use, replace steps 1 and 2
with the API-based scripts.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os

STEPS = [
    ("Phase 1: Convert existing scrobble data", "1b_convert_existing_data.py"),
    ("Phase 1: Generate synthetic audio features", "2b_synthetic_audio_features.py"),
    ("Phase 2: Build master dataset", "3_build_dataset.py"),
    ("Phase 3: Exploratory Data Analysis", "4_eda.py"),
    ("Phase 4: Vibe Clustering (K-Means)", "5_clustering.py"),
    ("Phase 5: Train LSTM Model", "6_train_lstm.py"),
    ("Phase 6: Run Prediction", "7_predict.py"),
]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  VIBE FORECASTER — Full Pipeline Execution")
    print("=" * 60)

    for i, (desc, script) in enumerate(STEPS, 1):
        print(f"\n{'─' * 60}")
        print(f"  Step {i}/{len(STEPS)}: {desc}")
        print(f"  Running: {script}")
        print(f"{'─' * 60}\n")

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED at step {i}: {script} (exit code {result.returncode})")
            print("Fix the error above and re-run.")
            sys.exit(1)

        print(f"\n✅ Step {i} complete: {desc}")

    print(f"\n{'=' * 60}")
    print("  ✅ ALL STEPS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    for folder in ["data", "models", "plots"]:
        if os.path.isdir(folder):
            files = os.listdir(folder)
            print(f"\n  {folder}/")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(folder, f))
                print(f"    {f:40s} {size:>10,} bytes")

    print("\nNext steps:")
    print("  - View plots in the plots/ directory")
    print("  - Run 'streamlit run app.py' for the interactive dashboard")


if __name__ == "__main__":
    main()
