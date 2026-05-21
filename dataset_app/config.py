import os
from pathlib import Path

# Resolve BASE_DIR from this file's location so the app works under both
# Windows ("f:/ai/train_dataset") and WSL ("/mnt/f/ai/train_dataset") without
# editing config per host. An explicit DATASETS_BASE_DIR env var still wins.
_env_base = os.environ.get("DATASETS_BASE_DIR")
BASE_DIR = Path(_env_base) if _env_base else Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
