from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import yaml
import shutil
import random
import json
import hashlib
import re
from pathlib import Path

HASH_STEM_PATTERN = re.compile(r"^[0-9a-f]{16}$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Splits that participate in training. "pending" is a holding area for
# unlabeled images and is intentionally excluded from auto-split, benchmark,
# and the gallery's "All Splits" view.
TRAINING_SPLITS = ["train", "valid", "test", "val"]
PENDING_SPLIT = "pending"
ALL_SPLITS = TRAINING_SPLITS + [PENDING_SPLIT]

def compute_image_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
from config import DATASETS_DIR, BASE_DIR, MODELS_DIR

app = FastAPI(title="YOLO Segmentation Dataset Editor")

# Setup templates and static files
# We will create static and templates directories later
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount an endpoint to serve raw datasets images
app.mount("/datasets", StaticFiles(directory=str(DATASETS_DIR)), name="datasets")

templates = Jinja2Templates(directory="templates")

def get_yaml_path(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name / "data.yaml"

EDGE_PROXIMITY_THRESHOLD = 0.02

def _polygon_edge_score(coords, threshold: float = EDGE_PROXIMITY_THRESHOLD) -> float:
    """Score how 'almost-off-screen' a polygon is.

    1.0 means a vertex sits just inside the frame and the polygon never
    actually touches the edge — suspicious, the user likely forgot to drag
    that point past the boundary. 0.0 means either every vertex is well
    inside, or some vertex already reaches/exceeds the edge.
    """
    min_d_inside = 1.0
    for i in range(0, len(coords), 2):
        try:
            x = float(coords[i])
            y = float(coords[i + 1])
        except (ValueError, IndexError):
            continue
        if x <= 0.0 or x >= 1.0 or y <= 0.0 or y >= 1.0:
            return 0.0
        d = min(x, 1.0 - x, y, 1.0 - y)
        if d < min_d_inside:
            min_d_inside = d
    if min_d_inside >= threshold:
        return 0.0
    return round(1.0 - (min_d_inside / threshold), 4)

def _compute_label_polygon_stats(label_path: Path):
    total = 0
    point_counts = []
    edge_scores = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 7 and len(parts) % 2 == 1:
                    total += 1
                    point_counts.append((len(parts) - 1) // 2)
                    edge_scores.append(_polygon_edge_score(parts[1:]))
    return {
        "total_polygons": total,
        "max_polygon_points": max(point_counts) if point_counts else 0,
        "min_polygon_points": min(point_counts) if point_counts else 0,
        "edge_score": max(edge_scores) if edge_scores else 0.0,
    }

def _negative_samples_file(dataset_name: str) -> Path:
    """Per-dataset list of images intentionally left unlabeled (true negatives).

    Stored at the dataset root, not under any split folder, so YOLO training
    tooling never sees it. Keyed by image filename (the hash-stem.ext form
    after normalize_filenames runs).
    """
    return DATASETS_DIR / dataset_name / "negative_samples.json"


def load_negative_samples(dataset_name: str) -> set:
    p = _negative_samples_file(dataset_name)
    if not p.exists():
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_negative_samples(dataset_name: str, names) -> None:
    p = _negative_samples_file(dataset_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sorted(names), f)


META_SCHEMA_VERSION = 2

def get_image_meta(dataset_name: str, force_rebuild: bool = False) -> dict:
    """Return per-image metadata (created/modified/polygon stats).

    Cached in image_meta.json keyed by image filename. Entries are
    invalidated when image or label mtime changes, so we never re-read a
    label file that hasn't moved since last scan.
    """
    meta_file = DATASETS_DIR / dataset_name / "image_meta.json"
    meta = {}
    if meta_file.exists() and not force_rebuild:
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    updated = False
    current_keys = set()

    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue
        for img_file in images_dir.glob("*.*"):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            current_keys.add(img_file.name)

            img_stat = img_file.stat()
            img_mtime = img_stat.st_mtime

            lbl_file = labels_dir / (img_file.stem + ".txt")
            lbl_mtime = lbl_file.stat().st_mtime if lbl_file.exists() else 0.0

            existing = meta.get(img_file.name)
            if (not force_rebuild
                    and existing
                    and existing.get("v") == META_SCHEMA_VERSION
                    and existing.get("img_mtime") == img_mtime
                    and existing.get("lbl_mtime") == lbl_mtime):
                continue

            created = getattr(img_stat, "st_birthtime", img_stat.st_ctime)
            poly_stats = _compute_label_polygon_stats(lbl_file)

            meta[img_file.name] = {
                "v": META_SCHEMA_VERSION,
                "created": created,
                "modified": img_mtime,
                "img_mtime": img_mtime,
                "lbl_mtime": lbl_mtime,
                **poly_stats,
            }
            updated = True

    stale = [k for k in meta if k not in current_keys]
    for k in stale:
        meta.pop(k)
        updated = True

    if updated:
        try:
            with open(meta_file, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    return meta

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    # List all directories in datasets
    datasets = []
    if DATASETS_DIR.exists():
        for item in DATASETS_DIR.iterdir():
            if item.is_dir() and (item / "data.yaml").exists():
                datasets.append(item.name)
    
    return templates.TemplateResponse(
        request=request, name="index.html", context={"datasets": datasets}
    )

@app.get("/dataset/{dataset_name}", response_class=HTMLResponse)
async def read_dataset(request: Request, dataset_name: str):
    yaml_path = get_yaml_path(dataset_name)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found or missing data.yaml")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    classes = data.get("names", [])
    
    scores_file = DATASETS_DIR / dataset_name / "auto_check.json"
    diff_scores = {}
    if scores_file.exists():
        try:
            with open(scores_file, "r") as f:
                diff_scores = json.load(f)
        except Exception:
            pass

    image_meta = get_image_meta(dataset_name)
    negative_samples = load_negative_samples(dataset_name)

    # Let's find images (looking into train/images for now, or val/images, test/images)
    images = []
    stem_counts = {}
    all_img_files = []
    for split in ALL_SPLITS:
        split_dir = DATASETS_DIR / dataset_name / split / "images"
        if split_dir.exists():
            for img_file in split_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                     stem_counts[img_file.stem] = stem_counts.get(img_file.stem, 0) + 1
                     all_img_files.append((split, img_file))

    for split, img_file in all_img_files:
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        label_file = labels_dir / (img_file.stem + ".txt")

        classes_present = set()
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes_present.add(int(parts[0]))
                        except ValueError:
                            pass

        m = image_meta.get(img_file.name, {})
        images.append({
            "name": img_file.name,
            "path": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
            "label_path": f"/datasets/{dataset_name}/{split}/labels/{label_file.name}",
            "has_label": label_file.exists(),
            "is_negative": img_file.name in negative_samples,
            "split": split,
            "classes_present": list(classes_present),
            "is_duplicate": stem_counts[img_file.stem] > 1,
            "diff_score": diff_scores.get(img_file.name, 0.0),
            "created": m.get("created", 0.0),
            "modified": m.get("modified", 0.0),
            "total_polygons": m.get("total_polygons", 0),
            "max_polygon_points": m.get("max_polygon_points", 0),
            "min_polygon_points": m.get("min_polygon_points", 0),
            "edge_score": m.get("edge_score", 0.0),
        })
    
    return templates.TemplateResponse(
        request=request, name="dataset.html", context={
            "dataset_name": dataset_name,
            "classes": classes,
            "images": images,
            "hooks_local_enabled": HOOKS_LOCAL_ENABLED,
        }
    )

@app.get("/editor/{dataset_name}", response_class=HTMLResponse)
async def read_editor(request: Request, dataset_name: str, img: str, lbl: str):
    yaml_path = get_yaml_path(dataset_name)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found or missing data.yaml")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    classes = data.get("names", [])
    
    return templates.TemplateResponse(
        request=request, name="editor.html", context={
            "dataset_name": dataset_name,
            "image_url": img,
            "label_url": lbl,
            "classes": classes
        }
    )

from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float

class Polygon(BaseModel):
    classId: int
    points: List[Point]

class SaveLabelsRequest(BaseModel):
    dataset_name: str
    label_path: str
    polygons: List[Polygon]

import urllib.parse

@app.get("/api/labels")
async def api_get_labels(dataset: str, label_path: str):
    # Construct full physical path from request path
    # label_path is something like /datasets/dogcat/train/labels/000.txt
    decoded_path = urllib.parse.unquote(label_path)
    # the frontend requests it as /datasets/... so lstrip the first slash to append to BASE_DIR correctly
    # actually, since BASE_DIR is f:/ai/train_dataset, and the path is /datasets/dogcat...
    # stripping the absolute leading slash makes it a relative path to concatenate.
    physical_path = BASE_DIR / decoded_path.lstrip("/")
    
    polygons = []
    if physical_path.exists():
        with open(physical_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7 and len(parts) % 2 == 1: # class_id x1 y1 x2 y2 x3 y3 ...
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    points = [{"x": coords[i], "y": coords[i+1]} for i in range(0, len(coords), 2)]
                    polygons.append({"classId": class_id, "points": points})
                    
    return {"polygons": polygons}

@app.post("/api/save_labels")
async def api_save_labels(req: SaveLabelsRequest):
    physical_path = BASE_DIR / req.label_path.lstrip("/")
    
    # Ensure parent dir exists
    physical_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(physical_path, "w", encoding="utf-8") as f:
        for poly in req.polygons:
            coords_str = " ".join([f"{pt.x:.6f} {pt.y:.6f}" for pt in poly.points])
            f.write(f"{poly.classId} {coords_str}\n")
            
    return {"status": "ok"}

class ModifyClassRequest(BaseModel):
    action: str # "add", "rename", "delete", "merge"
    dataset_name: str
    class_name: str = "" # Used for add/rename
    class_id: int = -1 # Used for rename/delete/merge (the one being modified/deleted)
    target_class_id: int = -1 # Used only for merge

def update_label_files(dataset_name: str, deleted_id: int, merge_target_id: int = -1):
    # If deleted_id matches, it becomes merge_target_id (if valid), else deleted.
    # Any ID > deleted_id decreases by 1.
    for split in ALL_SPLITS:
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not labels_dir.exists(): continue
        for txt_file in labels_dir.glob("*.txt"):
            lines = []
            modified = False
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    c_id = int(parts[0])
                    if c_id == deleted_id:
                        if merge_target_id != -1:
                            # It becomes the target
                            new_id = merge_target_id
                            # If target ID is originally > deleted_id, its new ID will be target_id - 1
                            if merge_target_id > deleted_id:
                                new_id -= 1
                            lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                            modified = True
                        else:
                            # Just deleted
                            modified = True
                            continue
                    else:
                        new_id = c_id
                        if c_id > deleted_id:
                            new_id -= 1
                        if new_id != c_id:
                            modified = True
                        lines.append(f"{new_id} {' '.join(parts[1:])}\n")
            
            if modified:
                with open(txt_file, "w") as f:
                    f.writelines(lines)

@app.post("/api/class_manage")
async def api_class_manage(req: ModifyClassRequest):
    yaml_path = get_yaml_path(req.dataset_name)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="yaml not found")
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    names = data.get("names", [])
    
    if req.action == "add":
        if req.class_name not in names:
            names.append(req.class_name)
    elif req.action == "rename":
        if 0 <= req.class_id < len(names):
            names[req.class_id] = req.class_name
    elif req.action == "delete":
        if 0 <= req.class_id < len(names):
            names.pop(req.class_id)
            update_label_files(req.dataset_name, req.class_id, -1)
    elif req.action == "merge":
        if 0 <= req.class_id < len(names) and 0 <= req.target_class_id < len(names) and req.class_id != req.target_class_id:
            names.pop(req.class_id)
            update_label_files(req.dataset_name, req.class_id, req.target_class_id)
            
    data["names"] = names
    data["nc"] = len(names) # Update number of classes just in case
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
        
    return {"status": "ok", "classes": names}

@app.get("/api/models")
async def api_get_models():
    models = []
    if MODELS_DIR.exists():
        for item in MODELS_DIR.glob("*.pt"):
            models.append(item.name)
    return {"models": models}

class AutoSegmentRequest(BaseModel):
    dataset_name: str
    image_path: str
    model_name: str
    use_denoise: bool = False
    h_lum: float = 10.0
    h_col: float = 10.0
    tw: int = 7
    sw: int = 21

@app.post("/api/auto_segment")
def api_auto_segment(req: AutoSegmentRequest):
    model_path = MODELS_DIR / req.model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    decoded_img_path = urllib.parse.unquote(req.image_path)
    physical_img_path = BASE_DIR / decoded_img_path.lstrip("/")
    
    if not physical_img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Read dataset data.yaml to resolve classes
    yaml_path = get_yaml_path(req.dataset_name)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dataset_classes = data.get("names", [])
    
    try:
        # Dynamically import to avoid slowing down startup if inference isn't used
        from ultralytics import YOLO
        import tempfile
        import cv2
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="ultralytics or cv2 library is not installed.")

    model = YOLO(str(model_path))
    
    # Run prediction
    if getattr(req, 'use_denoise', False):
        # Apply OpenCV Denoising
        img_bgr = cv2.imread(str(physical_img_path))
        if img_bgr is None:
            raise HTTPException(status_code=500, detail="Failed to read image for denoising.")
        
        # dst = fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)
        denoised_img = cv2.fastNlMeansDenoisingColored(
            img_bgr, None, float(req.h_lum), float(req.h_col), int(req.tw), int(req.sw)
        )
        results = model.predict(source=denoised_img, save=False, conf=0.25, retina_masks=True)
    else:
        results = model.predict(source=str(physical_img_path), save=False, conf=0.25, retina_masks=True)
    
    result = results[0]
    
    # Get model class names
    model_classes = result.names
    
    predicted_polygons = []
    
    if result.masks is not None:
        # masks.xyn provides normalized relative coordinates [0-1] which fits perfectly with our schema
        masks_xyn = result.masks.xyn 
        class_ids = result.boxes.cls.tolist()
        
        for mask_xyn, cls_id_float in zip(masks_xyn, class_ids):
            original_cls_id = int(cls_id_float)
            original_cls_name = model_classes.get(original_cls_id, f"class_{original_cls_id}")
            
            # Try to map to dataset class
            mapped_dataset_id = -1
            if original_cls_name in dataset_classes:
                mapped_dataset_id = dataset_classes.index(original_cls_name)
            else:
                # Class from model is unknown to this dataset. Fallback to 0 if possible, or append it.
                # For safety against accidental schema breaking, we will default mapping to 0 for unknown classes
                # and emit a warning.
                mapped_dataset_id = 0
                print(f"Dataset '{req.dataset_name}' missing model class '{original_cls_name}'. Assuming 0.")
                
            points = [{"x": float(x), "y": float(y)} for (x, y) in mask_xyn]
            predicted_polygons.append({
                "classId": mapped_dataset_id,
                "points": points
            })
            
    return {"polygons": predicted_polygons}

class ShrinkPolygonsRequest(BaseModel):
    polygons: List[Polygon]
    shrink_percent: float
    indices: Optional[List[int]] = None

@app.post("/api/shrink_polygons")
async def api_shrink_polygons(req: ShrinkPolygonsRequest):
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="cv2/numpy not installed")

    if req.shrink_percent <= 0:
        return {"polygons": [{"classId": p.classId, "points": [{"x": pt.x, "y": pt.y} for pt in p.points]} for p in req.polygons]}

    target_set = set(req.indices) if req.indices is not None else None
    out = []
    for i, poly in enumerate(req.polygons):
        passthrough = (target_set is not None and i not in target_set) or len(poly.points) < 3
        if passthrough:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        pts_arr = np.array([(pt.x, pt.y) for pt in poly.points], dtype=np.float64)
        min_xy = pts_arr.min(axis=0)
        max_xy = pts_arr.max(axis=0)
        bbox_w = float(max_xy[0] - min_xy[0])
        bbox_h = float(max_xy[1] - min_xy[1])
        min_side = min(bbox_w, bbox_h)
        if min_side <= 0:
            out.append({"classId": poly.classId, "points": []})
            continue

        shrink_px = max(1, int(round(min_side * req.shrink_percent / 100.0)))
        pad = shrink_px + 2
        mask_w = int(np.ceil(bbox_w)) + 2 * pad
        mask_h = int(np.ceil(bbox_h)) + 2 * pad
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        local_pts = np.round(pts_arr - min_xy + pad).astype(np.int32)
        cv2.fillPoly(mask, [local_pts], 1)

        k = 2 * shrink_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(mask, kernel)

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out.append({"classId": poly.classId, "points": []})
            continue

        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.0, True)
        if len(contour) < 3:
            out.append({"classId": poly.classId, "points": []})
            continue

        new_pts = [
            {"x": float(c[0][0] - pad + min_xy[0]), "y": float(c[0][1] - pad + min_xy[1])}
            for c in contour
        ]
        out.append({"classId": poly.classId, "points": new_pts})

    return {"polygons": out}

class SnapPolygonsRequest(BaseModel):
    image_path: str
    polygons: List[Polygon]
    indices: Optional[List[int]] = None
    iterations: int = 3
    margin_px: int = 8
    smooth_px: float = 0.8

@app.post("/api/snap_polygons")
async def api_snap_polygons(req: SnapPolygonsRequest):
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="cv2/numpy not installed")

    decoded_img_path = urllib.parse.unquote(req.image_path)
    physical_img_path = BASE_DIR / decoded_img_path.lstrip("/")
    if not physical_img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img_bgr = cv2.imread(str(physical_img_path))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read image")
    H_img, W_img = img_bgr.shape[:2]

    iterations = max(1, min(int(req.iterations), 10))
    margin = max(1, int(req.margin_px))
    smooth_eps = max(0.0, min(float(req.smooth_px), 5.0))
    target_set = set(req.indices) if req.indices is not None else None

    out = []
    for i, poly in enumerate(req.polygons):
        passthrough = (target_set is not None and i not in target_set) or len(poly.points) < 3
        if passthrough:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        pts = np.array([(pt.x, pt.y) for pt in poly.points], dtype=np.float64)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)

        x0 = max(0, int(np.floor(min_xy[0])) - margin)
        y0 = max(0, int(np.floor(min_xy[1])) - margin)
        x1 = min(W_img, int(np.ceil(max_xy[0])) + margin)
        y1 = min(H_img, int(np.ceil(max_xy[1])) + margin)

        if x1 - x0 < 4 or y1 - y0 < 4:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        crop = img_bgr[y0:y1, x0:x1].copy()
        Hc, Wc = crop.shape[:2]

        init_mask = np.full((Hc, Wc), cv2.GC_PR_BGD, dtype=np.uint8)
        local_pts = np.round(pts - np.array([x0, y0])).astype(np.int32)
        cv2.fillPoly(init_mask, [local_pts], int(cv2.GC_PR_FGD))

        # Seed a definite-foreground core by eroding the polygon — keeps GrabCut
        # anchored when the polygon is mostly correct.
        inner = np.zeros((Hc, Wc), dtype=np.uint8)
        cv2.fillPoly(inner, [local_pts], 1)
        erode_k = max(3, int(min(Hc, Wc) * 0.08))
        if erode_k % 2 == 0:
            erode_k += 1
        inner_eroded = cv2.erode(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))
        init_mask[inner_eroded == 1] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(crop, init_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        except Exception:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        fg_mask = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        contour = max(contours, key=cv2.contourArea)
        if smooth_eps > 0.0:
            contour = cv2.approxPolyDP(contour, smooth_eps, True)
        if len(contour) < 3:
            out.append({"classId": poly.classId, "points": [{"x": pt.x, "y": pt.y} for pt in poly.points]})
            continue

        new_pts = [
            {"x": float(c[0][0] + x0), "y": float(c[0][1] + y0)}
            for c in contour
        ]
        out.append({"classId": poly.classId, "points": new_pts})

    return {"polygons": out}

class AutoCheckRequest(BaseModel):
    model_name: str

@app.post("/api/dataset/{dataset_name}/auto_check")
def api_auto_check(dataset_name: str, req: AutoCheckRequest):
    model_path = MODELS_DIR / req.model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="Missing required libraries for Auto Check")

    model = YOLO(str(model_path))
    scores = {}

    # Check all images in dataset (pending images have no labels, so skip them)
    for split in TRAINING_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists(): continue
        
        for img_file in images_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]: continue
            
            lbl_file = labels_dir / (img_file.stem + ".txt")
            
            # Predict
            results = model.predict(source=str(img_file), save=False, conf=0.25, verbose=False, retina_masks=True)
            res = results[0]
            
            # Create a 640x640 mask for rendering (to calculate IoU fast via rasterization)
            H, W = 640, 640
            gt_mask = np.zeros((H, W), dtype=np.uint8)
            pred_mask = np.zeros((H, W), dtype=np.uint8)
            
            # Fill Ground Truth
            if lbl_file.exists():
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 7 and len(parts) % 2 == 1:
                            coords = [float(x) for x in parts[1:]]
                            pts = np.array([ [int(coords[i]*W), int(coords[i+1]*H)] for i in range(0, len(coords), 2) ], dtype=np.int32)
                            cv2.fillPoly(gt_mask, [pts], 1)
                            
            # Fill Predictions
            if res.masks is not None:
                for mask_xyn in res.masks.xyn:
                    pts = np.array([ [int(x*W), int(y*H)] for x, y in mask_xyn ], dtype=np.int32)
                    cv2.fillPoly(pred_mask, [pts], 1)
                    
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            
            if union == 0:
                iou = 1.0 # both are empty -> perfect match
            else:
                iou = intersection / union
                
            diff = 1.0 - iou
            scores[img_file.name] = round(diff, 4)
            
    scores_file = DATASETS_DIR / dataset_name / "auto_check.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f)
        
    return {"status": "ok", "checked": len(scores)}

class AutoCheckSingleRequest(BaseModel):
    model_name: str
    image_path: str
    use_denoise: bool = False
    h_lum: float = 10.0
    h_col: float = 10.0
    tw: int = 7
    sw: int = 21
    
@app.post("/api/dataset/{dataset_name}/auto_check_single")
def api_auto_check_single(dataset_name: str, req: AutoCheckSingleRequest):
    model_path = MODELS_DIR / req.model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="Missing required libraries for Auto Check")
        
    decoded_img_path = urllib.parse.unquote(req.image_path)
    physical_img_path = BASE_DIR / decoded_img_path.lstrip("/")
    
    if not physical_img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    model = YOLO(str(model_path))
    
    lbl_file = physical_img_path.parent.parent / "labels" / (physical_img_path.stem + ".txt")
    
    if getattr(req, 'use_denoise', False):
        img_bgr = cv2.imread(str(physical_img_path))
        if img_bgr is not None:
            denoised_img = cv2.fastNlMeansDenoisingColored(
                img_bgr, None, float(req.h_lum), float(req.h_col), int(req.tw), int(req.sw)
            )
            results = model.predict(source=denoised_img, save=False, conf=0.25, verbose=False, retina_masks=True)
        else:
            results = model.predict(source=str(physical_img_path), save=False, conf=0.25, verbose=False, retina_masks=True)
    else:
        results = model.predict(source=str(physical_img_path), save=False, conf=0.25, verbose=False, retina_masks=True)
        
    res = results[0]
    
    H, W = 640, 640
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    
    if lbl_file.exists():
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7 and len(parts) % 2 == 1:
                    coords = [float(x) for x in parts[1:]]
                    pts = np.array([ [int(coords[i]*W), int(coords[i+1]*H)] for i in range(0, len(coords), 2) ], dtype=np.int32)
                    cv2.fillPoly(gt_mask, [pts], 1)
                    
    if res.masks is not None:
        for mask_xyn in res.masks.xyn:
            pts = np.array([ [int(x*W), int(y*H)] for x, y in mask_xyn ], dtype=np.int32)
            cv2.fillPoly(pred_mask, [pts], 1)
            
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    iou = 1.0 if union == 0 else intersection / union
    diff = round(1.0 - iou, 4)
    
    scores_file = DATASETS_DIR / dataset_name / "auto_check.json"
    scores = {}
    if scores_file.exists():
        try:
            with open(scores_file, "r") as f:
                scores = json.load(f)
        except Exception:
            pass
            
    scores[physical_img_path.name] = diff
    with open(scores_file, "w") as f:
        json.dump(scores, f)

    return {"status": "ok", "diff_score": diff, "image": physical_img_path.name}

class BenchmarkRequest(BaseModel):
    model_names: List[str]
    split: str = "test"  # "train" | "valid" | "test" | "val" | "all"
    # Composite score weights. Default weighting reflects the censoring use case:
    # missing a class is much worse than imperfect mask shape.
    weight_mask: float = 0.30
    weight_class: float = 0.40
    weight_critical: float = 0.30
    # Class names that should be detected with high spatial accuracy (case-insensitive).
    # Names absent from the dataset's data.yaml are silently dropped.
    critical_classes: List[str] = ["penis", "pussy"]

@app.post("/api/dataset/{dataset_name}/benchmark")
def api_benchmark(dataset_name: str, req: BenchmarkRequest):
    if not req.model_names:
        raise HTTPException(status_code=400, detail="No models specified")

    valid_splits = {"train", "valid", "test", "val", "all"}
    if req.split not in valid_splits:
        raise HTTPException(status_code=400, detail=f"Invalid split: {req.split}")

    splits = list(TRAINING_SPLITS) if req.split == "all" else [req.split]

    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch
        import time
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing required libraries: {e}")

    # Resolve dataset class list and intersect critical classes against it.
    yaml_path = get_yaml_path(dataset_name)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="data.yaml not found")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    dataset_classes = data_yaml.get("names", []) or []
    name_to_id = {n.lower(): i for i, n in enumerate(dataset_classes)}
    critical_ids = []
    critical_names_used = []
    for n in req.critical_classes:
        key = n.lower()
        if key in name_to_id:
            critical_ids.append(name_to_id[key])
            critical_names_used.append(dataset_classes[name_to_id[key]])
    critical_id_set = set(critical_ids)

    # Collect images and corresponding label paths from the requested split(s).
    image_paths = []
    label_paths = []
    for split in splits:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.*")):
            if img_file.suffix.lower() in IMAGE_EXTS:
                image_paths.append(img_file)
                label_paths.append(labels_dir / f"{img_file.stem}.txt")

    if not image_paths:
        raise HTTPException(status_code=404, detail=f"No images found in split: {req.split}")

    H, W = 640, 640

    # Pre-rasterize ground truth: overall (any-class union) + per-critical-class
    # masks, plus the set of class IDs present per image.
    gt_overall = []
    gt_critical_masks = []  # list of dict {cls_id: mask}
    gt_class_sets = []
    for lbl_file in label_paths:
        overall = np.zeros((H, W), dtype=np.uint8)
        crit = {}
        cls_set = set()
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and len(parts) % 2 == 1:
                        try:
                            cls_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        coords = [float(x) for x in parts[1:]]
                        pts = np.array(
                            [[int(coords[i] * W), int(coords[i + 1] * H)] for i in range(0, len(coords), 2)],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(overall, [pts], 1)
                        cls_set.add(cls_id)
                        if cls_id in critical_id_set:
                            if cls_id not in crit:
                                crit[cls_id] = np.zeros((H, W), dtype=np.uint8)
                            cv2.fillPoly(crit[cls_id], [pts], 1)
        gt_overall.append(overall)
        gt_critical_masks.append(crit)
        gt_class_sets.append(cls_set)

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"
    paths_str = [str(p) for p in image_paths]

    w_mask = float(req.weight_mask)
    w_class = float(req.weight_class)
    w_critical = float(req.weight_critical)

    def composite_from_means(mean_mask, mean_class, mean_critical):
        total = 0.0
        weight_used = 0.0
        if mean_mask is not None:
            total += w_mask * mean_mask
            weight_used += w_mask
        if mean_class is not None:
            total += w_class * mean_class
            weight_used += w_class
        if mean_critical is not None:
            total += w_critical * mean_critical
            weight_used += w_critical
        return (total / weight_used * 100.0) if weight_used > 0 else 0.0

    results = []
    for model_name in req.model_names:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            results.append({
                "model": model_name,
                "error": "Model file not found",
                "score": 0.0,
                "mean_iou": 0.0,
                "median_iou": 0.0,
                "class_recall": None,
                "critical_iou": None,
                "image_count": 0,
                "elapsed_sec": 0.0,
            })
            continue

        mask_ious = []
        class_recalls = []
        critical_iou_values = []
        t0 = time.time()
        model = None
        try:
            model = YOLO(str(model_path))
            # Build a translation table from model class IDs to dataset class IDs
            # via name match (case-insensitive). Predictions whose model class has
            # no corresponding dataset class are ignored for class-aware metrics.
            model_to_dataset = {}
            for mid, mname in (model.names or {}).items():
                key = str(mname).lower()
                if key in name_to_id:
                    model_to_dataset[int(mid)] = name_to_id[key]

            stream = model.predict(
                source=paths_str,
                save=False,
                conf=0.25,
                verbose=False,
                retina_masks=True,
                device=device,
                half=use_cuda,
                stream=True,
            )
            for idx, res in enumerate(stream):
                pred_overall = np.zeros((H, W), dtype=np.uint8)
                pred_critical = {}
                pred_class_set = set()
                if res.masks is not None and res.boxes is not None:
                    cls_ids = res.boxes.cls.tolist()
                    for mask_xyn, mcls_f in zip(res.masks.xyn, cls_ids):
                        mcls = int(mcls_f)
                        pts = np.array(
                            [[int(x * W), int(y * H)] for x, y in mask_xyn],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(pred_overall, [pts], 1)
                        if mcls in model_to_dataset:
                            dcls = model_to_dataset[mcls]
                            pred_class_set.add(dcls)
                            if dcls in critical_id_set:
                                if dcls not in pred_critical:
                                    pred_critical[dcls] = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(pred_critical[dcls], [pts], 1)

                gt_o = gt_overall[idx]
                inter = int(np.logical_and(gt_o, pred_overall).sum())
                union = int(np.logical_or(gt_o, pred_overall).sum())
                mask_iou = 1.0 if union == 0 else inter / union
                mask_ious.append(float(mask_iou))

                gt_cls = gt_class_sets[idx]
                if gt_cls:
                    matched = gt_cls & pred_class_set
                    class_recalls.append(len(matched) / len(gt_cls))
                # When GT has no classes, recall is undefined — skip rather than
                # rewarding empty predictions with a free 1.0.

                per_img_critical = []
                for cid in critical_ids:
                    gt_m = gt_critical_masks[idx].get(cid)
                    pr_m = pred_critical.get(cid)
                    if gt_m is None and pr_m is None:
                        continue
                    if gt_m is None or pr_m is None:
                        per_img_critical.append(0.0)
                        continue
                    ci = int(np.logical_and(gt_m, pr_m).sum())
                    cu = int(np.logical_or(gt_m, pr_m).sum())
                    per_img_critical.append(1.0 if cu == 0 else ci / cu)
                if per_img_critical:
                    critical_iou_values.append(float(np.mean(per_img_critical)))
        except Exception as e:
            mean_mask = float(np.mean(mask_ious)) if mask_ious else None
            mean_class = float(np.mean(class_recalls)) if class_recalls else None
            mean_crit = float(np.mean(critical_iou_values)) if critical_iou_values else None
            results.append({
                "model": model_name,
                "error": str(e),
                "score": round(composite_from_means(mean_mask, mean_class, mean_crit), 2),
                "mean_iou": round(mean_mask, 4) if mean_mask is not None else 0.0,
                "median_iou": round(float(np.median(mask_ious)), 4) if mask_ious else 0.0,
                "class_recall": round(mean_class, 4) if mean_class is not None else None,
                "critical_iou": round(mean_crit, 4) if mean_crit is not None else None,
                "image_count": len(mask_ious),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            continue
        finally:
            del model
            if use_cuda:
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        mean_mask = float(np.mean(mask_ious)) if mask_ious else None
        mean_class = float(np.mean(class_recalls)) if class_recalls else None
        mean_crit = float(np.mean(critical_iou_values)) if critical_iou_values else None
        results.append({
            "model": model_name,
            "score": round(composite_from_means(mean_mask, mean_class, mean_crit), 2),
            "mean_iou": round(mean_mask, 4) if mean_mask is not None else 0.0,
            "median_iou": round(float(np.median(mask_ious)), 4) if mask_ious else 0.0,
            "class_recall": round(mean_class, 4) if mean_class is not None else None,
            "critical_iou": round(mean_crit, 4) if mean_crit is not None else None,
            "image_count": len(mask_ious),
            "elapsed_sec": round(elapsed, 2),
        })

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    payload = {
        "status": "ok",
        "split": req.split,
        "image_count": len(image_paths),
        "device": "cuda" if use_cuda else "cpu",
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": {"mask": w_mask, "class": w_class, "critical": w_critical},
        "critical_classes_used": critical_names_used,
        "critical_classes_requested": list(req.critical_classes),
        "results": results,
    }

    bench_file = DATASETS_DIR / dataset_name / "benchmark.json"
    try:
        with open(bench_file, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

    return payload

@app.get("/api/dataset/{dataset_name}/benchmark_result")
async def api_benchmark_result(dataset_name: str):
    bench_file = DATASETS_DIR / dataset_name / "benchmark.json"
    if not bench_file.exists():
        return {"status": "none"}
    try:
        with open(bench_file, "r") as f:
            return json.load(f)
    except Exception:
        return {"status": "none"}

class AutoSplitRequest(BaseModel):
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1

@app.post("/api/dataset/{dataset_name}/auto_split")
def api_auto_split(dataset_name: str, req: AutoSplitRequest):
    # validate ratios
    total = req.train_ratio + req.valid_ratio + req.test_ratio
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")
        
    all_images = []
    # Collect all existing images and labels. Pending images are intentionally
    # excluded — they're unlabeled and the user is holding them out of training.
    for split in TRAINING_SPLITS:
        split_dir = DATASETS_DIR / dataset_name / split / "images"
        if split_dir.exists():
            for img_file in split_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                     labels_dir = DATASETS_DIR / dataset_name / split / "labels"
                     label_file = labels_dir / (img_file.stem + ".txt")
                     all_images.append({
                         "img_path": img_file,
                         "label_path": label_file if label_file.exists() else None
                     })

    # Shuffle
    random.shuffle(all_images)
    
    total_imgs = len(all_images)
    train_idx = int(total_imgs * req.train_ratio)
    valid_idx = train_idx + int(total_imgs * req.valid_ratio)
    
    train_imgs = all_images[:train_idx]
    valid_imgs = all_images[train_idx:valid_idx]
    test_imgs = all_images[valid_idx:]
    
    def move_files(imgs, target_split):
        img_dest_dir = DATASETS_DIR / dataset_name / target_split / "images"
        lbl_dest_dir = DATASETS_DIR / dataset_name / target_split / "labels"
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)
        
        for item in imgs:
            shutil.move(str(item["img_path"]), str(img_dest_dir / item["img_path"].name))
            if item["label_path"]:
                shutil.move(str(item["label_path"]), str(lbl_dest_dir / item["label_path"].name))

    move_files(train_imgs, "train")
    if valid_imgs: move_files(valid_imgs, "valid")
    if test_imgs: move_files(test_imgs, "test")
    
    return {"status": "ok", "moved": total_imgs}

class MoveImageRequest(BaseModel):
    image_path: str
    target_split: str # train, valid, test

@app.post("/api/dataset/{dataset_name}/move_image")
async def api_move_image(dataset_name: str, req: MoveImageRequest):
    if req.target_split not in ALL_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid target split")
        
    decoded_img_path = urllib.parse.unquote(req.image_path)
    physical_img_path = BASE_DIR / decoded_img_path.lstrip("/")
    
    if not physical_img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Find active split from the path
    parts = list(physical_img_path.parts)
    # Ex: f:\ai\train_dataset\datasets\dogcat\train\images\000.jpg
    # The split dir is parts[-3]
    try:
        active_split = parts[-3]
    except IndexError:
        raise HTTPException(status_code=400, detail="Malformed image path")
        
    if active_split == req.target_split:
        return {"status": "ok", "message": "Already in target split"}
        
    label_path = physical_img_path.parent.parent / "labels" / (physical_img_path.stem + ".txt")
    
    img_dest_dir = DATASETS_DIR / dataset_name / req.target_split / "images"
    lbl_dest_dir = DATASETS_DIR / dataset_name / req.target_split / "labels"
    
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)
    
    new_img_path = img_dest_dir / physical_img_path.name
    shutil.move(str(physical_img_path), str(new_img_path))
    
    if label_path.exists():
        new_lbl_path = lbl_dest_dir / label_path.name
        shutil.move(str(label_path), str(new_lbl_path))
        
    return {
        "status": "ok",
        "new_image_path": f"/datasets/{dataset_name}/{req.target_split}/images/{physical_img_path.name}"
    }

@app.post("/api/dataset/{dataset_name}/move_unlabeled_to_pending")
async def api_move_unlabeled_to_pending(dataset_name: str):
    """Bulk-move every unlabeled image in train/valid/test/val into pending.

    Treats both "no label file" and "empty label file" as unlabeled. The empty
    label is removed so it doesn't get re-read as a valid (but bogus) entry
    after the move.
    """
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    negatives = load_negative_samples(dataset_name)
    dest_img_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "images"
    dest_img_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for split in TRAINING_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue
        for img_file in list(images_dir.glob("*.*")):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_file.name in negatives:
                continue
            lbl_file = labels_dir / (img_file.stem + ".txt")
            if lbl_file.exists() and lbl_file.stat().st_size > 0:
                continue
            dest_path = dest_img_dir / img_file.name
            if dest_path.exists():
                # Same hash already in pending — drop the duplicate source.
                img_file.unlink()
            else:
                shutil.move(str(img_file), str(dest_path))
            if lbl_file.exists():
                lbl_file.unlink()
            moved += 1

    return {"status": "ok", "moved": moved}

class CreateDatasetRequest(BaseModel):
    dataset_name: str

@app.post("/api/create_dataset")
async def api_create_dataset(req: CreateDatasetRequest):
    if not req.dataset_name or not req.dataset_name.strip():
        raise HTTPException(status_code=400, detail="Invalid dataset name")
        
    ds_dir = DATASETS_DIR / req.dataset_name.strip()
    if ds_dir.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
        
    # Scaffold directories
    for split in ["train", "valid", "test", PENDING_SPLIT]:
        (ds_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (ds_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    yaml_path = ds_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": [], "nc": 0}, f, sort_keys=False)
        
    return {"status": "ok", "dataset_name": req.dataset_name.strip()}

from fastapi import UploadFile, File

@app.post("/api/dataset/{dataset_name}/upload_images")
async def api_upload_images(dataset_name: str, files: List[UploadFile] = File(...)):
    # Track existing hash-stems across all splits (assumes filenames are already
    # normalized to hash form; if not, normalize_filenames will dedupe later).
    existing_stems = set()
    for split in ALL_SPLITS:
        split_dir = DATASETS_DIR / dataset_name / split / "images"
        if split_dir.exists():
            for img_file in split_dir.glob("*.*"):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    existing_stems.add(img_file.stem)

    # Newly uploaded images land in "pending" so they don't pollute training
    # until the user has labeled them and moved them into train/valid/test.
    target_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    skipped = 0
    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in IMAGE_EXTS:
            continue

        content = await file.read()
        hash_stem = hashlib.sha1(content).hexdigest()[:16]

        if hash_stem in existing_stems:
            skipped += 1
            continue

        new_filename = f"{hash_stem}{suffix}"
        file_path = target_dir / new_filename
        with open(file_path, "wb") as f:
            f.write(content)
        existing_stems.add(hash_stem)
        saved_files.append(new_filename)

    return {"status": "ok", "uploaded": len(saved_files), "skipped": skipped}

def _clip_polygon_to_unit_square(points):
    """Sutherland-Hodgman polygon clip against [0,1] x [0,1].

    Each iteration clips against one half-plane (one image edge), inserting
    new vertices at the intersections between in-segment and out-segment so
    the polygon outline is preserved instead of being collapsed to corners.
    Returns the clipped polygon (possibly empty if fully outside).
    """
    # (axis, boundary, inside_sign) — inside_sign=+1 means coord >= boundary
    edges = [
        ("x", 0.0, +1),  # left
        ("x", 1.0, -1),  # right
        ("y", 0.0, +1),  # top
        ("y", 1.0, -1),  # bottom
    ]

    def is_inside(pt, axis, boundary, sign):
        v = pt[0] if axis == "x" else pt[1]
        return (v >= boundary) if sign > 0 else (v <= boundary)

    def intersect(p1, p2, axis, boundary):
        x1, y1 = p1
        x2, y2 = p2
        if axis == "x":
            dx = x2 - x1
            t = 0.0 if dx == 0 else (boundary - x1) / dx
            return (boundary, y1 + t * (y2 - y1))
        dy = y2 - y1
        t = 0.0 if dy == 0 else (boundary - y1) / dy
        return (x1 + t * (x2 - x1), boundary)

    output = list(points)
    for axis, boundary, sign in edges:
        if not output:
            return output
        inp = output
        output = []
        s = inp[-1]
        s_in = is_inside(s, axis, boundary, sign)
        for e in inp:
            e_in = is_inside(e, axis, boundary, sign)
            if e_in:
                if not s_in:
                    output.append(intersect(s, e, axis, boundary))
                output.append(e)
            elif s_in:
                output.append(intersect(s, e, axis, boundary))
            s, s_in = e, e_in
    return output


def _clip_labels_in_dataset(dataset_name: str) -> dict:
    files_modified = 0
    polygons_clipped = 0
    polygons_dropped = 0

    for split in ALL_SPLITS:
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not labels_dir.exists():
            continue
        for lbl_file in labels_dir.glob("*.txt"):
            file_changed = False
            new_lines = []
            try:
                with open(lbl_file, "r", encoding="utf-8") as fp:
                    raw_lines = fp.readlines()
            except Exception:
                continue

            for line in raw_lines:
                parts = line.strip().split()
                if len(parts) < 7 or len(parts) % 2 != 1:
                    new_lines.append(line)
                    continue
                class_id = parts[0]
                try:
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    new_lines.append(line)
                    continue

                pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                has_oob = any(x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0 for x, y in pts)
                if not has_oob:
                    new_lines.append(line)
                    continue

                clipped = _clip_polygon_to_unit_square(pts)
                file_changed = True
                if len(clipped) < 3:
                    polygons_dropped += 1
                    continue
                polygons_clipped += 1
                clipped_strs = " ".join(
                    f"{max(0.0, min(1.0, c)):.6f}" for pt in clipped for c in pt
                )
                new_lines.append(f"{class_id} {clipped_strs}\n")

            if file_changed:
                with open(lbl_file, "w", encoding="utf-8") as fp:
                    fp.writelines(new_lines)
                files_modified += 1

    return {
        "files_modified": files_modified,
        "polygons_clipped": polygons_clipped,
        "polygons_dropped": polygons_dropped,
    }


@app.post("/api/dataset/{dataset_name}/clip_labels")
async def api_clip_labels(dataset_name: str):
    """Clip polygons against the image frame using Sutherland-Hodgman.

    Out-of-frame vertices are not naively clamped (which would collapse the
    polygon to image corners). Instead, intersections between an inside
    vertex and an outside vertex are inserted at the boundary, and the
    outside vertex is dropped. Polygons fully outside the frame are removed.
    """
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "ok", **_clip_labels_in_dataset(dataset_name)}


@app.post("/api/clip_all_labels")
async def api_clip_all_labels():
    """Run polygon clipping across every dataset under DATASETS_DIR."""
    per_dataset = []
    totals = {"files_modified": 0, "polygons_clipped": 0, "polygons_dropped": 0}
    if DATASETS_DIR.exists():
        for item in sorted(DATASETS_DIR.iterdir()):
            if not item.is_dir() or not (item / "data.yaml").exists():
                continue
            stats = _clip_labels_in_dataset(item.name)
            per_dataset.append({"dataset": item.name, **stats})
            for k in totals:
                totals[k] += stats[k]
    return {"status": "ok", "datasets": per_dataset, "totals": totals}

@app.post("/api/dataset/{dataset_name}/normalize_filenames")
async def api_normalize_filenames(dataset_name: str):
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    seen_hashes = {}     # hash_stem -> final Path of the kept image
    rename_map = {}      # old image filename -> new image filename
    renamed = 0
    deleted_dup = 0
    deleted_orphan = 0

    # Phase 1: rename images (and labels) to {hash}.{ext}; drop content duplicates.
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue

        for img_file in list(images_dir.glob("*.*")):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue

            old_name = img_file.name

            # Trust hash-pattern filenames to avoid re-hashing on every reload.
            if HASH_STEM_PATTERN.match(img_file.stem):
                hash_stem = img_file.stem
            else:
                hash_stem = compute_image_hash(img_file)

            new_suffix = img_file.suffix.lower()
            old_label = labels_dir / f"{img_file.stem}.txt"

            if hash_stem in seen_hashes:
                img_file.unlink()
                if old_label.exists():
                    old_label.unlink()
                deleted_dup += 1
                continue

            new_img_path = images_dir / f"{hash_stem}{new_suffix}"
            new_label = labels_dir / f"{hash_stem}.txt"

            if new_img_path != img_file:
                if new_img_path.exists():
                    img_file.unlink()
                    if old_label.exists():
                        old_label.unlink()
                    deleted_dup += 1
                    continue
                img_file.rename(new_img_path)
                if old_label.exists():
                    if new_label.exists():
                        old_label.unlink()
                    else:
                        old_label.rename(new_label)
                rename_map[old_name] = new_img_path.name
                renamed += 1

            seen_hashes[hash_stem] = new_img_path

    # Phase 2: drop label files whose image no longer exists.
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not labels_dir.exists():
            continue

        image_stems = set()
        if images_dir.exists():
            for img_file in images_dir.glob("*.*"):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    image_stems.add(img_file.stem)

        for lbl_file in list(labels_dir.glob("*.txt")):
            if lbl_file.stem not in image_stems:
                lbl_file.unlink()
                deleted_orphan += 1

    # Update auto_check.json keys to follow renames; drop entries for removed images.
    scores_file = DATASETS_DIR / dataset_name / "auto_check.json"
    current_image_names = {p.name for p in seen_hashes.values()}
    if scores_file.exists():
        try:
            with open(scores_file, "r") as f:
                scores = json.load(f)
            new_scores = {}
            for k, v in scores.items():
                new_k = rename_map.get(k, k)
                if new_k in current_image_names:
                    new_scores[new_k] = v
            with open(scores_file, "w") as f:
                json.dump(new_scores, f)
        except Exception:
            pass

    # Mirror renames into negative_samples.json; drop entries for removed images.
    negatives = load_negative_samples(dataset_name)
    if negatives:
        new_negatives = set()
        for name in negatives:
            mapped = rename_map.get(name, name)
            if mapped in current_image_names:
                new_negatives.add(mapped)
        if new_negatives != negatives:
            save_negative_samples(dataset_name, new_negatives)

    return {
        "status": "ok",
        "renamed": renamed,
        "deleted_duplicates": deleted_dup,
        "deleted_orphans": deleted_orphan,
    }

class ToggleNegativeRequest(BaseModel):
    image_filename: str
    value: bool


@app.get("/api/dataset/{dataset_name}/negative_samples")
async def api_get_negative_samples(dataset_name: str):
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"items": sorted(load_negative_samples(dataset_name))}


@app.post("/api/dataset/{dataset_name}/toggle_negative")
async def api_toggle_negative(dataset_name: str, req: ToggleNegativeRequest):
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not req.image_filename or "/" in req.image_filename or "\\" in req.image_filename:
        raise HTTPException(status_code=400, detail="image_filename must be a bare filename")
    negatives = load_negative_samples(dataset_name)
    if req.value:
        negatives.add(req.image_filename)
    else:
        negatives.discard(req.image_filename)
    save_negative_samples(dataset_name, negatives)
    return {"status": "ok", "image_filename": req.image_filename, "is_negative": req.value}


class DeleteImageRequest(BaseModel):
    image_path: str

@app.post("/api/dataset/{dataset_name}/delete_image")
async def api_delete_image(dataset_name: str, req: DeleteImageRequest):
    # Security: Ensure path is within the dataset
    try:
        # Expected format: /datasets/dataset_name/split/images/filename.jpg
        # unquote and get path in case it's a full URL
        image_url = urllib.parse.unquote(req.image_path)
        if image_url.startswith("http"):
            image_url = urllib.parse.urlparse(image_url).path
            
        parts = image_url.strip("/").split("/")
        if len(parts) != 5 or parts[0] != "datasets" or parts[1] != dataset_name or parts[3] != "images":
             raise HTTPException(status_code=400, detail="Invalid image path format")
             
        split = parts[2]
        filename = parts[4]

        if split not in ALL_SPLITS:
            raise HTTPException(status_code=400, detail="Invalid split")
            
        img_file = DATASETS_DIR / dataset_name / split / "images" / filename
        
        # Determine the label file
        label_filename = os.path.splitext(filename)[0] + ".txt"
        lbl_file = DATASETS_DIR / dataset_name / split / "labels" / label_filename
        
        if img_file.exists():
            img_file.unlink()

        if lbl_file.exists():
            lbl_file.unlink()

        negatives = load_negative_samples(dataset_name)
        if filename in negatives:
            negatives.discard(filename)
            save_negative_samples(dataset_name, negatives)

        return {"status": "ok"}
    except Exception as e:
        print(f"Delete image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/{dataset_name}/next_unlabeled")
async def api_next_unlabeled(dataset_name: str):
    negatives = load_negative_samples(dataset_name)
    # Scan splits in order for an image without a label file
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue

        for img_file in sorted(images_dir.glob("*.*")):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                if img_file.name in negatives:
                    continue
                label_file = labels_dir / (img_file.stem + ".txt")
                if not label_file.exists() or label_file.stat().st_size == 0:
                    return {
                        "status": "ok",
                        "next_image": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
                        "next_label": f"/datasets/{dataset_name}/{split}/labels/{img_file.stem}.txt"
                    }
                    
    return {"status": "none"}

def get_all_images(dataset_name: str):
    images = []
    # Using the same order as in gallery (read_dataset)
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.*")):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                images.append({
                    "image_url": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
                    "label_url": f"/datasets/{dataset_name}/{split}/labels/{img_file.stem}.txt"
                })
    return images

import urllib.parse

@app.get("/api/dataset/{dataset_name}/next_image")
async def api_next_image(dataset_name: str, current_image: str):
    images = get_all_images(dataset_name)
    current_image = urllib.parse.unquote(current_image)
    if current_image.startswith("http"):
        current_image = urllib.parse.urlparse(current_image).path
        
    for i, img in enumerate(images):
        if img["image_url"] == current_image:
            if i + 1 < len(images):
                return {"status": "ok", "next": images[i + 1]}
            else:
                # Wrap around to the first image
                if len(images) > 0:
                    return {"status": "ok", "next": images[0]}
                break
    return {"status": "none"}

@app.get("/api/dataset/{dataset_name}/prev_image")
async def api_prev_image(dataset_name: str, current_image: str):
    images = get_all_images(dataset_name)
    current_image = urllib.parse.unquote(current_image)
    if current_image.startswith("http"):
        current_image = urllib.parse.urlparse(current_image).path

    for i, img in enumerate(images):
        if img["image_url"] == current_image:
            if i - 1 >= 0:
                return {"status": "ok", "prev": images[i - 1]}
            else:
                # Wrap around to the last image
                if len(images) > 0:
                     return {"status": "ok", "prev": images[-1]}
                break
    return {"status": "none"}


def _list_datasets() -> List[str]:
    out = []
    if DATASETS_DIR.exists():
        for item in sorted(DATASETS_DIR.iterdir()):
            if item.is_dir() and (item / "data.yaml").exists():
                out.append(item.name)
    return out


def _normalized_label_signature(label_path: Path) -> Optional[str]:
    """Order-independent fingerprint of a YOLO label file.

    Polygons that were re-ordered (e.g. by an editor that re-sorts on save)
    but otherwise identical should compare equal. Returns None if the file
    doesn't exist; empty string for an empty file.
    """
    if not label_path.exists():
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [" ".join(line.strip().split()) for line in f if line.strip()]
    except Exception:
        return None
    lines.sort()
    return "\n".join(lines)


def _index_dataset_images(dataset_name: str) -> dict:
    """Map image stem -> {split, image_filename, label_signature}.

    With hash-based filenames, identical image content shares the same stem
    across datasets, so stem is the right join key for "is this the same
    image?" comparisons.
    """
    by_stem = {}
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue
        for img_file in images_dir.glob("*.*"):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_file = labels_dir / (img_file.stem + ".txt")
            by_stem[img_file.stem] = {
                "split": split,
                "image_filename": img_file.name,
                "label_filename": lbl_file.name,
                "has_label": lbl_file.exists(),
                "label_signature": _normalized_label_signature(lbl_file),
            }
    return by_stem


@app.get("/compare", response_class=HTMLResponse)
async def read_compare(request: Request, a: Optional[str] = None, b: Optional[str] = None):
    return templates.TemplateResponse(
        request=request, name="compare.html", context={
            "datasets": _list_datasets(),
            "preselect_a": a or "",
            "preselect_b": b or "",
        }
    )


@app.get("/api/compare")
async def api_compare(dataset_a: str, dataset_b: str):
    if dataset_a == dataset_b:
        raise HTTPException(status_code=400, detail="Pick two different datasets")

    for name in (dataset_a, dataset_b):
        if not get_yaml_path(name).exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {name}")

    a_map = _index_dataset_images(dataset_a)
    b_map = _index_dataset_images(dataset_b)

    a_stems = set(a_map.keys())
    b_stems = set(b_map.keys())

    only_a_stems = sorted(a_stems - b_stems)
    only_b_stems = sorted(b_stems - a_stems)
    common_stems = a_stems & b_stems

    def entry(dataset_name: str, stem: str, info: dict) -> dict:
        return {
            "stem": stem,
            "dataset": dataset_name,
            "split": info["split"],
            "image_url": f"/datasets/{dataset_name}/{info['split']}/images/{info['image_filename']}",
            "label_url": f"/datasets/{dataset_name}/{info['split']}/labels/{info['label_filename']}",
            "has_label": info["has_label"],
        }

    label_diff = []
    for stem in sorted(common_stems):
        ia = a_map[stem]
        ib = b_map[stem]
        if ia["label_signature"] == ib["label_signature"]:
            continue
        label_diff.append({
            "stem": stem,
            "a": entry(dataset_a, stem, ia),
            "b": entry(dataset_b, stem, ib),
            "a_has_label": ia["has_label"],
            "b_has_label": ib["has_label"],
        })

    only_a = [entry(dataset_a, s, a_map[s]) for s in only_a_stems]
    only_b = [entry(dataset_b, s, b_map[s]) for s in only_b_stems]

    # Class definition diff from data.yaml
    def load_classes(name: str) -> List[str]:
        try:
            with open(get_yaml_path(name), "r", encoding="utf-8") as f:
                return (yaml.safe_load(f) or {}).get("names", []) or []
        except Exception:
            return []
    classes_a = load_classes(dataset_a)
    classes_b = load_classes(dataset_b)

    return {
        "status": "ok",
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "classes_a": classes_a,
        "classes_b": classes_b,
        "counts": {
            "a_total": len(a_map),
            "b_total": len(b_map),
            "common": len(common_stems),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "label_diff": len(label_diff),
        },
        "only_a": only_a,
        "only_b": only_b,
        "label_diff": label_diff,
    }


# Optional, gitignored local customizations. The contract is that the
# module defines `register(app, datasets_dir)` and adds whatever routes
# the user needs. Public repo stays clean of personal endpoints/secrets.
HOOKS_LOCAL_ENABLED = False
try:
    import hooks_local  # type: ignore
    if hasattr(hooks_local, "register"):
        hooks_local.register(app, datasets_dir=DATASETS_DIR)
        HOOKS_LOCAL_ENABLED = True
except ImportError:
    pass
except Exception as e:
    print(f"hooks_local failed to initialize: {e}")


