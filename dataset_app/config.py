import os
from pathlib import Path

# Base configuration
BASE_DIR = Path("f:/ai/train_dataset")
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
