#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying Jupyter kernel..."
python -m ipykernel install --user --name python3 --display-name "Python 3"

echo "Post-create setup complete."
