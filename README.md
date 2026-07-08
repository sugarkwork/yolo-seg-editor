# YOLO Segmentation Dataset Maker

FastAPI-based web application for creating, cleaning, labeling, comparing, and evaluating YOLO segmentation datasets.

The app is built around a safe dataset workflow: newly uploaded images land in `pending/`, and only labeled or explicitly negative samples are moved into `train/`, `valid/`, or `test/`.

## Current Workflow

1. Create or open a dataset from the dashboard.
2. Upload images. New files are hashed, deduplicated, and stored under `pending/images/`.
3. Label images in the editor, or mark images as Negative Sample when no detection target is present.
4. Move completed pending images into `test/` with **Pending Labeled/Negative -> Test**, or move individual images between splits from the gallery/editor.
5. Use Auto Split to reshuffle only the training splits (`train`, `valid`, `test`, `val`). `pending` is intentionally excluded.
6. Run Auto Check or Benchmark against models in `models/` to find labels that need review and compare model quality.

## Features

- Dataset dashboard for datasets that contain a `data.yaml`.
- YOLO segmentation editor with polygon drawing, point editing, undo/redo, zoom, pan, class selection, and image navigation.
- Class management from the dataset page and editor: add, rename, merge, and delete classes while updating labels.
- Pending split support:
  - Uploaded images go to `pending/images/`.
  - `pending` is excluded from Auto Split, Benchmark, and the gallery's default All Splits view.
  - **Unlabeled -> Pending** moves empty or missing-label training images out of training splits.
  - **Pending Labeled/Negative -> Test** moves completed pending samples into `test/`.
- Negative samples stored in `negative_samples.json`; these are intentionally unlabeled images and are skipped by the "Unlabeled (needs work)" filter and Save & Next.
- Filename normalization and dedupe via **Reload**:
  - Renames images and labels to `{sha1[:16]}.{ext}`.
  - Removes duplicate image content across splits.
  - Cleans orphan label files.
  - Updates `auto_check.json` and `negative_samples.json` keys after renames.
- Label cleanup:
  - Clip one dataset or all datasets with Sutherland-Hodgman clipping so polygons stay within the image frame.
  - Gallery metadata tracks polygon counts, point counts, and near-edge scores.
- Auto-segmentation with YOLO `.pt` models in `models/`, including optional OpenCV denoise.
- Polygon tools in the editor:
  - Shrink or expand polygons by a percentage of the bounding-box short side.
  - Snap polygons to image edges with GrabCut refinement.
- Auto Check:
  - Runs one model across training splits.
  - Scores each image by `1 - IoU(GT, prediction)`.
  - Stores scores in `auto_check.json` for gallery sorting.
  - Single-image diff check is available from the editor.
- Benchmark:
  - Ranks one or more models on `test`, `valid`, `train`, or all training splits.
  - Uses coverage-first censoring scoring: missed or under-covered GT regions are penalized heavily, while prediction overreach up to 5% of GT area is allowed.
  - Reports composite score, mask coverage score, class recall, critical-class coverage score, raw IoU, overreach, and false-positive penalty indicators.
  - Also evaluates best 2-model combinations when multiple models are selected.
  - Stores latest results in `benchmark.json`.
- Dataset comparison page with class definition diffs, file-only diffs, label-signature diffs, side-by-side canvas previews, and edit links.
- ONNX image tagging and tag search:
  - Reload normalizes filenames first, then tags images that are missing from the per-dataset tag database.
  - Uses WD-EVA02-Large-Tagger-v3 through ONNX Runtime, not TensorRT.
  - Stores tags in `tag_search.db` with `images`, `tags`, and `image_tags` tables.
  - Gallery tag search supports positive AND tags, negative excluded tags, and simple people-count conflict expansion such as `1girl` vs `2girls`.
- Optional archive/upload extension through `dataset_app/hooks_local.py` and `dataset_app/.env`.

## Dataset Layout

Datasets live under `datasets/<dataset_name>/`.

```text
datasets/
  my_dataset/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/
    val/
      images/
      labels/
    pending/
      images/
      labels/
    negative_samples.json
    image_meta.json
    auto_check.json
    benchmark.json
    tag_search.db
```

Notes:

- `val/` is supported as a training split for existing YOLO datasets.
- `pending/` is a holding area, not a training split.
- `negative_samples.json`, `image_meta.json`, `auto_check.json`, `benchmark.json`, and `tag_search.db` are app metadata files, not YOLO label files.

## Project Layout

```text
dataset_app/
  main.py                    FastAPI app and API routes
  config.py                  Paths for datasets and models
  hooks_share_upload.py      Optional archive/upload extension
  hooks_local.py.example     Template for enabling local hooks
  static/js/editor.js        Editor behavior
  static/js/compare.js       Dataset comparison behavior
  templates/                 Jinja2 pages
datasets/                    YOLO datasets
models/                      YOLO .pt models used by inference, Auto Check, and Benchmark
requirements.txt             Python dependencies
start.bat                    Windows launcher, port 8322
start.sh                     Linux/WSL launcher, default port 8322
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bat
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The app creates `datasets/` and `models/` automatically from `dataset_app/config.py`. You can override the project base directory with:

```bash
DATASETS_BASE_DIR=/path/to/project python -m uvicorn main:app
```

## Running

Windows:

```bat
start.bat
```

Linux/WSL:

```bash
./start.sh
```

Manual run:

```bash
cd dataset_app
python -m uvicorn main:app --host 0.0.0.0 --port 8322
```

Open:

```text
http://127.0.0.1:8322
```

## Models

Put YOLO segmentation `.pt` files in `models/`.

These models are used by:

- editor Auto-Segment
- editor Check Diff Score
- dataset Auto Check
- dataset Benchmark

If the repository is configured with Git LFS for model files:

```bash
git lfs install
git lfs pull
```

## ONNX Tagging

Image tags are generated by WD-EVA02-Large-Tagger-v3 through ONNX Runtime.

The app resolves the tagger model in this order:

1. `TAGGER_ONNX_PATH` environment variable.
2. `tagger_models/wd-eva02-large-tagger-v3-fp16.onnx` under the project root.
3. Existing ComfyUI paths such as `F:\ai\ComfyUI\custom_nodes\comfyui-onnxtagger\models\wd-eva02-large-tagger-v3-fp16.onnx`.

The tag CSV is resolved from `TAGGER_CSV_PATH`, `tagger_models/`, or existing ComfyUI WD14 tagger folders. If only the CSV is missing, the app downloads `selected_tags.csv` from Hugging Face.

Reload behavior:

```text
Reload -> normalize filenames -> remove duplicates/orphans -> sync tag_search.db -> tag only missing/changed images
```

First tagging of a large dataset can take time. Later Reload runs only process images that are new, changed, or missing from `tag_search.db`.

## Optional Archive Upload Hook

The dataset page can show **Archive & Upload** when local hooks are enabled.

To enable:

```bash
copy dataset_app\hooks_local.py.example dataset_app\hooks_local.py
```

Configure credentials in `dataset_app/.env`:

```text
UPLOAD_BASE_URL=https://example.com
UPLOAD_TOKEN=replace-with-token
UPLOAD_AUTH_SCHEME=Bearer
UPLOAD_PUBLIC=true
UPLOAD_7ZIP_PATH=C:\Program Files\7-Zip\7z.exe
```

If 7-Zip is unavailable, the hook falls back to ZIP archives.

## Editor Controls

- Left click: add point.
- Left-drag point: move point.
- Right click: close polygon or delete point.
- Mouse wheel: zoom.
- Middle drag: pan.
- Ctrl+Z / Ctrl+Y: undo / redo.
- Delete / Backspace: delete selected object or point.
- Esc: cancel drawing.
- N: toggle Negative Sample.

## Notes

- YOLO segmentation label lines are treated as valid when they have a class id plus at least three `(x, y)` points.
- Empty label files are not considered labeled.
- Negative samples are complete samples with no target, and can be moved into `test/`.
- Auto Split only shuffles existing `train`, `valid`, `test`, and `val` images; it does not pull from `pending/`.
