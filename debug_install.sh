#!/bin/bash
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Running pip install..."
pip install -r requirements.txt --verbose
echo "Deactivating virtual environment..."
deactivate
