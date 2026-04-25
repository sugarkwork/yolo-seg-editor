# YOLO Segmentation Dataset Maker

A FastAPI web application for creating, managing, and editing YOLO segmentation datasets.

## Features

- **Dataset Management:** View and create YOLO segmentation datasets (using `data.yaml`).
- **Image Upload:** Drag/drop or pick images from the dashboard. Files are saved into `train/images/` and renamed to `{sha1[:16]}.{ext}` on save; identical-content uploads are skipped automatically.
- **Auto-Split:** Randomly shuffle and distribute images and their labels into train/valid/test splits.
- **Move Between Splits:** Change a single image's split (with its label) directly from the gallery.
- **Filename Normalization & Dedupe:** Press **Reload** on the dataset page to rename every image and its label to `{sha1[:16]}.{ext}`, drop any content duplicates (within or across splits), clean up orphan label files (label without an image), and update `auto_check.json` keys accordingly. ASCII-only filenames keep the dataset portable across filesystems and training environments.
- **Image Editor:**
  - Polygon drawing tool for segmentation.
  - Interactive manipulation (drag points, undo/redo, zoom/pan).
  - Class management (add, rename, merge, delete classes).
  - Delete unwanted images directly from the editor.
- **Auto-Segmentation:** Use pre-trained YOLO models residing in the `models/` directory to predict segmentation masks and convert them to editable polygons in the browser. Optional OpenCV denoising can be applied before inference.
- **Auto-Check:** Run a model across the dataset and score every image by `1 - IoU(GT, Pred)`. Scores are persisted in `auto_check.json` and can be used in the gallery as a sort key (highest-diff first) to surface labels worth reviewing.

## Directory Structure

- `dataset_app/`: Application source code, including the FastAPI backend (`main.py`, `config.py`), static assets (`static/`), and HTML templates (`templates/`).
- `datasets/`: Storage directory for YOLO format datasets (e.g., `dogcat/train/images`, `dogcat/train/labels`).
- `models/`: Storage directory for `.pt` YOLO models used by the auto-segmentation feature.

## Setup & Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required directories created in the project root:
   ```bash
   mkdir datasets models
   ```

## Running the Application

1. Start the FastAPI server using Uvicorn:
   ```bash
   cd dataset_app
   uvicorn main:app --reload --port 8000
   ```

2. Access the dashboard in your web browser:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Git LFS for Models

This project uses Git LFS (Large File Storage) to track `.pt` model files located in the `models/` directory.

To ensure models are retrieved and pushed correctly:
```bash
git lfs install
git lfs pull
```
