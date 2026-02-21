import os
from pathlib import Path

# Base configuration
BASE_DIR = Path("f:/ai/train_dataset")
DATASETS_DIR = BASE_DIR / "datasets"

# Ensure datasets directory exists
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
