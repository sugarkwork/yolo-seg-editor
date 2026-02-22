# YOLO Segmentation Dataset Maker

A web application built with FastAPI and HTMX for creating, managing, and editing YOLO segmentation datasets.

## Features

- **Dataset Management:** View and create YOLO segmentation datasets (using `data.yaml`).
- **Image Upload:** Upload images directly to train/valid/test splits via the dashboard.
- **Auto-Split:** Randomly shuffle and distribute images and their labels into train/valid/test splits.
- **Image Editor:**
  - Polygon drawing tool for segmentation.
  - Interactive manipulation (drag points, undo/redo, zoom/pan).
  - Class management (add, rename, merge, delete classes).
  - Delete unwanted images directly from the editor.
- **Auto-Segmentation:** Use pre-trained YOLO models residing in the `models/` directory to automatically predict segmentation masks and convert them to editable polygons in the browser.

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
