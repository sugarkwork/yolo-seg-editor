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
import time
import gc
import threading
import urllib.parse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

HASH_STEM_PATTERN = re.compile(r"^[0-9a-f]{16}$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Splits that participate in training. "pending" is a holding area for
# unlabeled images and is intentionally excluded from auto-split, benchmark,
# and the gallery's "All Splits" view.
TRAINING_SPLITS = ["train", "valid", "test", "val"]
PENDING_SPLIT = "pending"
ALL_SPLITS = TRAINING_SPLITS + [PENDING_SPLIT]
MULTIPOLYGON_LABEL_DIR = "cocolabels"

def compute_image_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
from config import DATASETS_DIR, BASE_DIR, MODELS_DIR
from tag_search import TagSearchDatabase, split_tags, tag_missing_images

app = FastAPI(title="YOLO Segmentation Dataset Editor")

# Setup templates and static files
# We will create static and templates directories later
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount an endpoint to serve raw datasets images
app.mount("/datasets", StaticFiles(directory=str(DATASETS_DIR)), name="datasets")

templates = Jinja2Templates(directory="templates")

def get_yaml_path(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name / "data.yaml"

def current_dataset_image_items(dataset_name: str):
    items = []
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.*")):
            if img_file.suffix.lower() in IMAGE_EXTS:
                items.append((split, img_file))
    return items

def resolve_yolo_device(torch_module):
    configured = (
        os.environ.get("AUTO_SEGMENT_DEVICE", "").strip()
        or os.environ.get("YOLO_DEVICE", "").strip()
    )
    if configured:
        return int(configured) if configured.isdigit() else configured
    return 0 if torch_module.cuda.is_available() else "cpu"

def yolo_uses_cuda(torch_module, device) -> bool:
    if not torch_module.cuda.is_available():
        return False
    if isinstance(device, int):
        return True
    return str(device).lower() not in {"cpu", "mps"}

def log_yolo_device(torch_module, device, context: str) -> None:
    if yolo_uses_cuda(torch_module, device):
        index = device if isinstance(device, int) else 0
        if isinstance(device, str) and ":" in device:
            try:
                index = int(device.rsplit(":", 1)[1])
            except ValueError:
                index = 0
        try:
            name = torch_module.cuda.get_device_name(index)
        except Exception:
            name = "CUDA"
        print(f"[inference] {context} using GPU device={device} ({name})", flush=True)
    else:
        print(f"[inference] {context} using CPU device={device}", flush=True)

def _coerce_yolo_imgsz(value, default: int = 640):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        size = int(value)
        return size if size >= 32 else None
    if isinstance(value, str):
        nums = [int(x) for x in re.findall(r"\d+", value)]
        if not nums:
            return None
        return nums[0] if len(nums) == 1 or nums[0] == nums[1] else nums[:2]
    if isinstance(value, (list, tuple)):
        sizes = []
        for item in value:
            coerced = _coerce_yolo_imgsz(item, default)
            if isinstance(coerced, int):
                sizes.append(coerced)
            elif isinstance(coerced, list):
                sizes.extend(coerced)
        sizes = [s for s in sizes if s >= 32]
        if not sizes:
            return None
        return sizes[0] if len(sizes) == 1 or sizes[0] == sizes[1] else sizes[:2]
    return None

def _get_arg_value(args, key: str):
    if isinstance(args, dict):
        return args.get(key)
    return getattr(args, key, None)

def resolve_yolo_model_imgsz(model, default: int = 640):
    """Read the training image size stored in an Ultralytics model checkpoint."""
    sources = [
        getattr(getattr(model, "model", None), "args", None),
        getattr(model, "overrides", None),
        getattr(model, "args", None),
    ]
    ckpt = getattr(model, "ckpt", None)
    if isinstance(ckpt, dict):
        sources.append(ckpt.get("train_args"))

    for args in sources:
        imgsz = _coerce_yolo_imgsz(_get_arg_value(args, "imgsz"), default)
        if imgsz is not None:
            return imgsz
    return default

def log_yolo_imgsz(model, context: str, default: int = 640):
    imgsz = resolve_yolo_model_imgsz(model, default)
    print(f"[inference] {context} using imgsz={imgsz}", flush=True)
    return imgsz


def _resolve_benchmark_batch_size(requested=None) -> int:
    raw = requested if requested not in (None, "") else os.environ.get("YOLO_BENCHMARK_BATCH", "8")
    try:
        return max(1, min(64, int(raw)))
    except (TypeError, ValueError):
        return 8


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


YOLO_MODEL_CACHE = OrderedDict()
YOLO_MODEL_CACHE_LOCK = threading.Lock()


def _yolo_model_cache_limit() -> int:
    try:
        return max(1, int(os.environ.get("YOLO_MODEL_CACHE_MAX", "2")))
    except ValueError:
        return 2


def _yolo_model_cache_key(model_path: Path, device) -> tuple:
    stat = model_path.stat()
    return (
        str(model_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        str(device),
    )


def _get_cached_yolo_model(model_path: Path, device, context: str, yolo_cls):
    key = _yolo_model_cache_key(model_path, device)
    model_label = model_path.name
    with YOLO_MODEL_CACHE_LOCK:
        entry = YOLO_MODEL_CACHE.get(key)
        if entry is not None:
            YOLO_MODEL_CACHE.move_to_end(key)
            print(
                f"[inference] {context} using cached model={model_label} imgsz={entry['imgsz']}",
                flush=True,
            )
            return entry["model"], entry["imgsz"], entry["predict_lock"]

        started = time.time()
        model = yolo_cls(str(model_path))
        imgsz = resolve_yolo_model_imgsz(model)
        entry = {
            "model": model,
            "imgsz": imgsz,
            "predict_lock": threading.Lock(),
            "model_name": model_label,
        }
        YOLO_MODEL_CACHE[key] = entry
        print(
            f"[inference] {context} loaded model={model_label} imgsz={imgsz} "
            f"load_sec={time.time() - started:.2f}",
            flush=True,
        )

        while len(YOLO_MODEL_CACHE) > _yolo_model_cache_limit():
            _, evicted = YOLO_MODEL_CACHE.popitem(last=False)
            print(f"[inference] evicted cached model={evicted.get('model_name')}", flush=True)
            del evicted
            gc.collect()

        return entry["model"], entry["imgsz"], entry["predict_lock"]


def _clamp01(value) -> float:
    return max(0.0, min(1.0, float(value)))


def _prediction_extent_ratio(prediction: dict) -> float:
    points = prediction.get("points") or []
    if not points:
        return 0.0
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _polygon_iou_ratio(poly_a: dict, poly_b: dict, cv2_module, np_module, size: int = 256) -> float:
    points_a = poly_a.get("points") or []
    points_b = poly_b.get("points") or []
    if len(points_a) < 3 or len(points_b) < 3:
        return 0.0

    def to_array(points):
        return np_module.array(
            [[int(_clamp01(pt["x"]) * (size - 1)), int(_clamp01(pt["y"]) * (size - 1))] for pt in points],
            dtype=np_module.int32,
        )

    mask_a = np_module.zeros((size, size), dtype=np_module.uint8)
    mask_b = np_module.zeros((size, size), dtype=np_module.uint8)
    cv2_module.fillPoly(mask_a, [to_array(points_a)], 1)
    cv2_module.fillPoly(mask_b, [to_array(points_b)], 1)
    union = np_module.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np_module.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def _merge_padding_predictions(
    normal_predictions: list,
    padded_predictions: list,
    merge_mode: str,
    large_threshold_ratio: float,
    match_iou: float,
    cv2_module,
    np_module,
) -> list:
    mode = merge_mode if merge_mode in {"all", "large_only"} else "large_only"
    merged = [dict(pred) for pred in normal_predictions]
    large_normal_indices = {
        idx for idx, pred in enumerate(normal_predictions)
        if _prediction_extent_ratio(pred) >= large_threshold_ratio
    }

    for padded_pred in padded_predictions:
        is_large_padded = _prediction_extent_ratio(padded_pred) >= large_threshold_ratio
        best_idx = None
        best_iou = 0.0
        for idx, normal_pred in enumerate(merged):
            if normal_pred.get("classId") != padded_pred.get("classId"):
                continue
            iou = _polygon_iou_ratio(normal_pred, padded_pred, cv2_module, np_module)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        best_is_large_normal = best_idx in large_normal_indices if best_idx is not None else False
        should_merge = mode == "all" or is_large_padded or best_is_large_normal
        if not should_merge:
            continue

        if best_idx is not None and best_iou >= match_iou:
            merged[best_idx] = dict(padded_pred)
        else:
            merged.append(dict(padded_pred))

    return merged

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


def _multipolygon_path_for_label(label_path: Path) -> Path:
    return label_path.parent.parent / MULTIPOLYGON_LABEL_DIR / f"{label_path.stem}.json"


def _multipolygon_path_for_image(img_path: Path) -> Path:
    return img_path.parent.parent / MULTIPOLYGON_LABEL_DIR / f"{img_path.stem}.json"


def _point_xy(point):
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point.x), float(point.y)


def _polygon_value(poly, key, default=None):
    if isinstance(poly, dict):
        return poly.get(key, default)
    return getattr(poly, key, default)


def _normalize_polygon_dicts(polygons) -> list:
    normalized = []
    for idx, poly in enumerate(polygons):
        class_id = _polygon_value(poly, "classId")
        label_id = _polygon_value(poly, "labelId") or _polygon_value(poly, "label_id") or f"label-{idx + 1}"
        points = _polygon_value(poly, "points", [])
        pts = [{"x": _point_xy(pt)[0], "y": _point_xy(pt)[1]} for pt in points]
        if len(pts) >= 3:
            normalized.append({"labelId": str(label_id), "classId": int(class_id), "points": pts})
    return normalized


def _read_yolo_label(label_path: Path) -> list:
    polygons = []
    if not label_path.exists():
        return polygons
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) >= 7 and len(parts) % 2 == 1:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                points = [{"x": coords[i], "y": coords[i + 1]} for i in range(0, len(coords), 2)]
                label_id = f"label-{line_no}"
                for piece in _split_zero_width_bridged_points(points):
                    polygons.append({"labelId": label_id, "classId": class_id, "points": piece})
    return polygons


def _point_key(pt, precision: int = 6) -> tuple:
    return (round(float(pt["x"]), precision), round(float(pt["y"]), precision))


def _polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += float(p["x"]) * float(q["y"]) - float(q["x"]) * float(p["y"])
    return abs(area) * 0.5


def _valid_polygon_piece(points) -> bool:
    return len({_point_key(pt) for pt in points}) >= 3 and _polygon_area(points) > 1e-12


def _split_zero_width_bridged_points(points: list) -> list:
    """Split a self-touching YOLO ring back into multiple polygon islands."""
    cleaned = []
    last_key = None
    for pt in points:
        key = _point_key(pt)
        if key == last_key:
            continue
        cleaned.append({"x": float(pt["x"]), "y": float(pt["y"])})
        last_key = key
    if len(cleaned) >= 2 and _point_key(cleaned[0]) == _point_key(cleaned[-1]):
        cleaned = cleaned[:-1]

    stack = []
    index_by_key = {}
    pieces = []

    for pt in cleaned:
        key = _point_key(pt)
        if key in index_by_key:
            start = index_by_key[key]
            cycle = stack[start:]
            if _valid_polygon_piece(cycle):
                pieces.append(cycle)
            stack = stack[:start + 1]
            index_by_key = {_point_key(p): i for i, p in enumerate(stack)}
            continue
        index_by_key[key] = len(stack)
        stack.append(pt)

    if pieces:
        return pieces
    return [cleaned] if _valid_polygon_piece(cleaned) else []


def _read_multipolygon_label(coco_path: Path) -> list:
    """Read the app's per-image COCO-style multipolygon JSON."""
    if not coco_path.exists():
        return []
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polygons = []
    for ann_idx, ann in enumerate(data.get("annotations", []), start=1):
        label_id = str(ann.get("labelId") or ann.get("id") or f"label-{ann_idx}")
        class_id = int(ann.get("classId", ann.get("category_id", 0)))
        for seg in ann.get("segmentation", []):
            if len(seg) < 6 or len(seg) % 2 != 0:
                continue
            points = [{"x": float(seg[i]), "y": float(seg[i + 1])} for i in range(0, len(seg), 2)]
            polygons.append({"labelId": label_id, "classId": class_id, "points": points})
    return polygons


def _write_multipolygon_label(coco_path: Path, polygons) -> None:
    grouped = {}
    for poly in _normalize_polygon_dicts(polygons):
        label = grouped.setdefault(poly["labelId"], {"labelId": poly["labelId"], "classId": poly["classId"], "segmentation": []})
        label["classId"] = poly["classId"]
        label["segmentation"].append([
            coord
            for pt in poly["points"]
            for coord in (round(float(pt["x"]), 6), round(float(pt["y"]), 6))
        ])

    payload = {
        "version": 1,
        "type": "image_segmentation_multipolygon",
        "annotations": [
            {"id": label_id, "labelId": label_id, "classId": item["classId"], "segmentation": item["segmentation"]}
            for label_id, item in sorted(grouped.items())
        ],
    }
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def _rotate_points(points: list, index: int) -> list:
    return points[index:] + points[:index]


def _connect_polygons_with_zero_width_bridges(polygons: list) -> list:
    """Flatten one label's polygons into one YOLO-compatible ring.

    YOLO segmentation has one contour per label line. For multiple islands of
    the same label, this creates self-touching zero-width bridges so the mask
    rasterizes like separate polygons in most YOLO tooling.
    """
    rings = [[{"x": float(pt["x"]), "y": float(pt["y"])} for pt in poly["points"]] for poly in polygons]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return []

    connected = list(rings[0])
    anchor = connected[0]
    remaining = rings[1:]
    while remaining:
        best = None
        for ring_idx, ring in enumerate(remaining):
            for pt_idx, pt in enumerate(ring):
                dist = (pt["x"] - anchor["x"]) ** 2 + (pt["y"] - anchor["y"]) ** 2
                if best is None or dist < best[0]:
                    best = (dist, ring_idx, pt_idx)
        _, ring_idx, pt_idx = best
        ring = remaining.pop(ring_idx)
        bridge = ring[pt_idx]
        connected.extend([anchor, bridge])
        connected.extend(_rotate_points(ring, pt_idx))
        connected.extend([bridge, anchor])
        anchor = bridge
    return connected


def _write_yolo_label_from_polygons(label_path: Path, polygons) -> None:
    grouped = {}
    for poly in _normalize_polygon_dicts(polygons):
        label = grouped.setdefault(poly["labelId"], {"classId": poly["classId"], "polygons": []})
        label["classId"] = poly["classId"]
        label["polygons"].append(poly)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for label_id, label in sorted(grouped.items()):
            label_polygons = label["polygons"]
            points = (
                label_polygons[0]["points"]
                if len(label_polygons) == 1
                else _connect_polygons_with_zero_width_bridges(label_polygons)
            )
            if len(points) < 3:
                continue
            coords_str = " ".join(f"{pt['x']:.6f} {pt['y']:.6f}" for pt in points)
            f.write(f"{label['classId']} {coords_str}\n")


def _compute_same_class_mask_overlap_score(polygons, mask_size: int = 256) -> float:
    """Return max same-class overlap between different labels in one image.

    The score is intersection / smaller-mask-area, so a near-duplicate label
    contained in another label scores close to 1.0 even when IoU is lower.
    """
    labels = {}
    for poly in _normalize_polygon_dicts(polygons):
        item = labels.setdefault(poly["labelId"], {"classId": poly["classId"], "polygons": []})
        item["classId"] = poly["classId"]
        item["polygons"].append(poly)

    by_class = {}
    for label_id, item in labels.items():
        by_class.setdefault(item["classId"], []).append((label_id, item["polygons"]))

    if not any(len(items) >= 2 for items in by_class.values()):
        return 0.0

    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    def rasterize(label_polygons):
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        for poly in label_polygons:
            pts = []
            for pt in poly["points"]:
                x = int(round(max(0.0, min(1.0, float(pt["x"]))) * (mask_size - 1)))
                y = int(round(max(0.0, min(1.0, float(pt["y"]))) * (mask_size - 1)))
                pts.append([x, y])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
        return mask

    best = 0.0
    for items in by_class.values():
        if len(items) < 2:
            continue
        masks = []
        for label_id, label_polygons in items:
            mask = rasterize(label_polygons)
            area = int(mask.sum())
            if area > 0:
                masks.append((label_id, mask, area))
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                _, ma, area_a = masks[i]
                _, mb, area_b = masks[j]
                inter = int((ma & mb).sum())
                if inter <= 0:
                    continue
                score = inter / max(1, min(area_a, area_b))
                if score > best:
                    best = score
    return round(float(best), 4)


def _compute_label_polygon_stats(label_path: Path):
    total = 0
    point_counts = []
    edge_scores = []
    polygons_for_overlap = []
    coco_path = _multipolygon_path_for_label(label_path)
    if coco_path.exists():
        try:
            polygons_for_overlap = _read_multipolygon_label(coco_path)
            for poly in polygons_for_overlap:
                coords = [coord for pt in poly["points"] for coord in (pt["x"], pt["y"])]
                total += 1
                point_counts.append(len(poly["points"]))
                edge_scores.append(_polygon_edge_score(coords))
        except Exception:
            pass
    elif label_path.exists():
        try:
            polygons_for_overlap = _read_yolo_label(label_path)
            for poly in polygons_for_overlap:
                coords = [coord for pt in poly["points"] for coord in (pt["x"], pt["y"])]
                total += 1
                point_counts.append(len(poly["points"]))
                edge_scores.append(_polygon_edge_score(coords))
        except Exception:
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
        "mask_overlap_score": _compute_same_class_mask_overlap_score(polygons_for_overlap),
    }

def _label_has_segmentation_annotations(label_path: Path) -> bool:
    coco_path = _multipolygon_path_for_label(label_path)
    if coco_path.exists():
        try:
            return any(len(poly.get("points", [])) >= 3 for poly in _read_multipolygon_label(coco_path))
        except Exception:
            pass
    if not label_path.exists():
        return False
    try:
        with open(label_path, "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 7 and len(parts) % 2 == 1:
                    return True
    except Exception:
        return False
    return False

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


def _auto_unreviewed_file(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name / "auto_labeled_unreviewed.json"


def load_auto_labeled_unreviewed(dataset_name: str) -> set:
    p = _auto_unreviewed_file(dataset_name)
    if not p.exists():
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_auto_labeled_unreviewed(dataset_name: str, names) -> None:
    p = _auto_unreviewed_file(dataset_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sorted(names), f)


def clear_auto_labeled_unreviewed(dataset_name: str, image_filename: str) -> bool:
    names = load_auto_labeled_unreviewed(dataset_name)
    if image_filename not in names:
        return False
    names.discard(image_filename)
    save_auto_labeled_unreviewed(dataset_name, names)
    return True


def _image_filename_for_label_path(label_path: Path) -> Optional[str]:
    images_dir = label_path.parent.parent / "images"
    if not images_dir.exists():
        return None
    for img_file in images_dir.glob(f"{label_path.stem}.*"):
        if img_file.suffix.lower() in IMAGE_EXTS:
            return img_file.name
    return None


META_SCHEMA_VERSION = 3

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
            coco_file = _multipolygon_path_for_label(lbl_file)
            coco_mtime = coco_file.stat().st_mtime if coco_file.exists() else 0.0

            existing = meta.get(img_file.name)
            if (not force_rebuild
                    and existing
                    and existing.get("v") == META_SCHEMA_VERSION
                    and existing.get("img_mtime") == img_mtime
                    and existing.get("lbl_mtime") == lbl_mtime
                    and existing.get("coco_mtime") == coco_mtime):
                continue

            created = getattr(img_stat, "st_birthtime", img_stat.st_ctime)
            poly_stats = _compute_label_polygon_stats(lbl_file)

            meta[img_file.name] = {
                "v": META_SCHEMA_VERSION,
                "created": created,
                "modified": img_mtime,
                "img_mtime": img_mtime,
                "lbl_mtime": lbl_mtime,
                "coco_mtime": coco_mtime,
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
    auto_unreviewed = load_auto_labeled_unreviewed(dataset_name)
    image_tags_by_name = {}
    try:
        tag_db = TagSearchDatabase(DATASETS_DIR / dataset_name)
        # Filled after image discovery below.
    except Exception:
        tag_db = None

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

    if tag_db is not None:
        try:
            image_tags_by_name = tag_db.tags_for_filenames([p.name for _, p in all_img_files], limit_per_image=24)
        except Exception:
            image_tags_by_name = {}

    for split, img_file in all_img_files:
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        label_file = labels_dir / (img_file.stem + ".txt")

        classes_present = set()
        coco_file = _multipolygon_path_for_label(label_file)
        if coco_file.exists():
            try:
                for poly in _read_multipolygon_label(coco_file):
                    classes_present.add(int(poly["classId"]))
            except Exception:
                pass
        elif label_file.exists():
            with open(label_file, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes_present.add(int(parts[0]))
                        except ValueError:
                            pass

        m = image_meta.get(img_file.name, {})
        tag_items = image_tags_by_name.get(img_file.name, [])
        images.append({
            "name": img_file.name,
            "path": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
            "label_path": f"/datasets/{dataset_name}/{split}/labels/{label_file.name}",
            "has_label": _label_has_segmentation_annotations(label_file),
            "is_negative": img_file.name in negative_samples,
            "is_auto_unreviewed": img_file.name in auto_unreviewed,
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
            "mask_overlap_score": m.get("mask_overlap_score", 0.0),
            "tags": tag_items,
            "tag_names": [t.get("tag", "") for t in tag_items],
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
    image_filename = Path(urllib.parse.unquote(img).split("?")[0]).name
    is_auto_unreviewed = image_filename in load_auto_labeled_unreviewed(dataset_name)
    
    return templates.TemplateResponse(
        request=request, name="editor.html", context={
            "dataset_name": dataset_name,
            "image_url": img,
            "label_url": lbl,
            "classes": classes,
            "is_auto_unreviewed": is_auto_unreviewed,
        }
    )

from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float

class Polygon(BaseModel):
    labelId: Optional[str] = None
    classId: int
    points: List[Point]

class SaveLabelsRequest(BaseModel):
    dataset_name: str
    label_path: str
    polygons: List[Polygon]

import urllib.parse


def _polygon_response(poly: Polygon, points=None) -> dict:
    return {
        "labelId": poly.labelId,
        "classId": poly.classId,
        "points": points if points is not None else [{"x": pt.x, "y": pt.y} for pt in poly.points],
    }

@app.get("/api/labels")
async def api_get_labels(dataset: str, label_path: str):
    # Construct full physical path from request path
    # label_path is something like /datasets/dogcat/train/labels/000.txt
    decoded_path = urllib.parse.unquote(label_path)
    # the frontend requests it as /datasets/... so lstrip the first slash to append to BASE_DIR correctly
    # actually, since BASE_DIR is f:/ai/train_dataset, and the path is /datasets/dogcat...
    # stripping the absolute leading slash makes it a relative path to concatenate.
    physical_path = BASE_DIR / decoded_path.lstrip("/")
    
    coco_path = _multipolygon_path_for_label(physical_path)
    polygons = _read_multipolygon_label(coco_path) if coco_path.exists() else _read_yolo_label(physical_path)
    return {"polygons": polygons}

@app.post("/api/save_labels")
async def api_save_labels(req: SaveLabelsRequest):
    physical_path = BASE_DIR / req.label_path.lstrip("/")
    polygons = _normalize_polygon_dicts(req.polygons)

    _write_multipolygon_label(_multipolygon_path_for_label(physical_path), polygons)
    _write_yolo_label_from_polygons(physical_path, polygons)
    image_filename = _image_filename_for_label_path(physical_path)
    if image_filename:
        clear_auto_labeled_unreviewed(req.dataset_name, image_filename)

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
        if labels_dir.exists():
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
                    _write_yolo_label_from_polygons(txt_file, _read_yolo_label(txt_file))

        coco_dir = DATASETS_DIR / dataset_name / split / MULTIPOLYGON_LABEL_DIR
        if not coco_dir.exists():
            continue
        for json_file in coco_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            annotations = []
            modified = False
            for ann in data.get("annotations", []):
                try:
                    c_id = int(ann.get("classId", ann.get("category_id", -1)))
                except (TypeError, ValueError):
                    annotations.append(ann)
                    continue
                if c_id == deleted_id:
                    if merge_target_id == -1:
                        modified = True
                        continue
                    new_id = merge_target_id - 1 if merge_target_id > deleted_id else merge_target_id
                    modified = True
                else:
                    new_id = c_id - 1 if c_id > deleted_id else c_id
                    modified = modified or (new_id != c_id)
                ann["classId"] = new_id
                ann.pop("category_id", None)
                annotations.append(ann)

            if modified:
                data["annotations"] = annotations
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                _write_yolo_label_from_polygons(labels_dir / f"{json_file.stem}.txt", _read_multipolygon_label(json_file))

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
    use_padding_inference: bool = False
    pad_y_percent: float = 30.0
    pad_x_percent: float = 0.0
    large_threshold: float = 60.0
    padding_merge_mode: str = "large_only"
    padding_match_iou: float = 0.2


def _collect_yolo_result_polygons(result, dataset_classes: list) -> list:
    predictions = []
    if result is None or result.masks is None or result.boxes is None:
        return predictions

    name_to_id = {str(name).lower(): idx for idx, name in enumerate(dataset_classes)}
    model_classes = result.names or {}
    masks_xyn = result.masks.xyn
    class_ids = result.boxes.cls.tolist()

    for mask_xyn, cls_id_float in zip(masks_xyn, class_ids):
        model_cls_id = int(cls_id_float)
        model_cls_name = str(model_classes.get(model_cls_id, f"class_{model_cls_id}"))
        dataset_cls_id = name_to_id.get(model_cls_name.lower())
        if dataset_cls_id is None:
            print(f"[auto_label] skipping unmapped model class: {model_cls_name}", flush=True)
            continue

        points = [{"x": _clamp01(float(x)), "y": _clamp01(float(y))} for x, y in mask_xyn]
        if len({(round(pt["x"], 6), round(pt["y"], 6)) for pt in points}) >= 3:
            predictions.append({
                "classId": dataset_cls_id,
                "points": points,
            })
    return predictions


def _group_auto_label_predictions(predictions: list, merge_mask_size: int = 1024) -> list:
    by_class = {}
    for pred in predictions:
        class_id = int(pred.get("classId", 0))
        points = pred.get("points") or []
        if len(points) < 3:
            continue
        by_class.setdefault(class_id, []).append(points)

    try:
        import cv2
        import numpy as np
    except Exception:
        grouped = []
        for class_id, polygons in by_class.items():
            for points in polygons:
                grouped.append({
                    "labelId": f"auto-class-{class_id}",
                    "classId": class_id,
                    "points": points,
                })
        return grouped

    size = max(128, min(4096, int(merge_mask_size)))
    merged = []
    for class_id, polygons in sorted(by_class.items()):
        mask = np.zeros((size, size), dtype=np.uint8)
        for points in polygons:
            pts = np.array(
                [
                    [
                        int(round(_clamp01(pt["x"]) * (size - 1))),
                        int(round(_clamp01(pt["y"]) * (size - 1))),
                    ]
                    for pt in points
                ],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) <= 1:
                continue
            epsilon = max(0.5, size * 0.0008)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                approx = contour
            out_points = [
                {
                    "x": _clamp01(float(pt[0][0]) / (size - 1)),
                    "y": _clamp01(float(pt[0][1]) / (size - 1)),
                }
                for pt in approx
            ]
            if len({(round(pt["x"], 6), round(pt["y"], 6)) for pt in out_points}) >= 3:
                merged.append({
                    "labelId": f"auto-class-{class_id}",
                    "classId": class_id,
                    "points": out_points,
                })
    return merged


def _resolve_auto_label_devices(torch_module):
    configured = (
        os.environ.get("AUTO_SEGMENT_DEVICE", "").strip()
        or os.environ.get("YOLO_DEVICE", "").strip()
    )
    if configured:
        device = int(configured) if configured.isdigit() else configured
        return [device, device]
    if torch_module.cuda.is_available() and torch_module.cuda.device_count() >= 2:
        return [0, 1]
    device = resolve_yolo_device(torch_module)
    return [device, device]


@app.post("/api/auto_segment")
def api_auto_segment(req: AutoSegmentRequest):
    request_started = time.time()
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
        import torch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load inference libraries: {e}")

    device = resolve_yolo_device(torch)
    use_cuda = yolo_uses_cuda(torch, device)
    log_yolo_device(torch, device, "auto_segment")
    model, model_imgsz, model_predict_lock = _get_cached_yolo_model(
        model_path, device, "auto_segment", YOLO
    )

    base_img_bgr = None
    use_padding_inference = bool(getattr(req, "use_padding_inference", False))
    if getattr(req, "use_denoise", False) or use_padding_inference:
        img_bgr = cv2.imread(str(physical_img_path))
        if img_bgr is None:
            raise HTTPException(status_code=500, detail="Failed to read image for inference.")
        if getattr(req, "use_denoise", False):
            base_img_bgr = cv2.fastNlMeansDenoisingColored(
                img_bgr, None, float(req.h_lum), float(req.h_col), int(req.tw), int(req.sw)
            )
        else:
            base_img_bgr = img_bgr

    def dataset_class_id(original_cls_id: int, model_classes: dict) -> int:
        original_cls_name = model_classes.get(original_cls_id, f"class_{original_cls_id}")
        if original_cls_name in dataset_classes:
            return dataset_classes.index(original_cls_name)
        print(f"Dataset '{req.dataset_name}' missing model class '{original_cls_name}'. Assuming 0.")
        return 0

    def collect_predictions(result, padded_shape=None, original_shape=None, pad_x_px=0, pad_y_px=0):
        predictions = []
        if result.masks is None:
            return predictions

        model_classes = result.names
        masks_xyn = result.masks.xyn
        class_ids = result.boxes.cls.tolist()

        padded_w = padded_h = orig_w = orig_h = None
        if padded_shape and original_shape:
            padded_h, padded_w = padded_shape[:2]
            orig_h, orig_w = original_shape[:2]

        for mask_xyn, cls_id_float in zip(masks_xyn, class_ids):
            mapped_dataset_id = dataset_class_id(int(cls_id_float), model_classes)
            points = []
            for x, y in mask_xyn:
                if padded_w and padded_h and orig_w and orig_h:
                    px = (float(x) * padded_w - pad_x_px) / orig_w
                    py = (float(y) * padded_h - pad_y_px) / orig_h
                    points.append({"x": _clamp01(px), "y": _clamp01(py)})
                else:
                    points.append({"x": _clamp01(x), "y": _clamp01(y)})
            if len({(round(pt["x"], 6), round(pt["y"], 6)) for pt in points}) >= 3:
                predictions.append({
                    "classId": mapped_dataset_id,
                    "points": points,
                })
        return predictions

    predict_kwargs = {
        "save": False,
        "conf": 0.25,
        "retina_masks": True,
        "device": device,
        "half": use_cuda,
        "imgsz": model_imgsz,
    }

    normal_source = base_img_bgr if base_img_bgr is not None else str(physical_img_path)
    with model_predict_lock:
        results = model.predict(source=normal_source, **predict_kwargs)
    predicted_polygons = collect_predictions(results[0])

    padding_info = {
        "enabled": False,
        "normal_count": len(predicted_polygons),
        "padded_count": 0,
        "merged_count": len(predicted_polygons),
    }

    if use_padding_inference:
        if base_img_bgr is None:
            base_img_bgr = cv2.imread(str(physical_img_path))
        if base_img_bgr is None:
            raise HTTPException(status_code=500, detail="Failed to read image for padding inference.")

        orig_h, orig_w = base_img_bgr.shape[:2]
        pad_y_ratio = max(0.0, float(getattr(req, "pad_y_percent", 30.0))) / 100.0
        pad_x_ratio = max(0.0, float(getattr(req, "pad_x_percent", 0.0))) / 100.0
        pad_y_px = int(round(orig_h * pad_y_ratio))
        pad_x_px = int(round(orig_w * pad_x_ratio))

        if pad_y_px > 0 or pad_x_px > 0:
            padded_img = cv2.copyMakeBorder(
                base_img_bgr,
                pad_y_px,
                pad_y_px,
                pad_x_px,
                pad_x_px,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        else:
            padded_img = base_img_bgr

        with model_predict_lock:
            padded_results = model.predict(source=padded_img, **predict_kwargs)
        padded_polygons = collect_predictions(
            padded_results[0],
            padded_shape=padded_img.shape,
            original_shape=base_img_bgr.shape,
            pad_x_px=pad_x_px,
            pad_y_px=pad_y_px,
        )
        large_threshold_ratio = max(0.0, min(1.0, float(getattr(req, "large_threshold", 60.0)) / 100.0))
        match_iou = max(0.0, min(1.0, float(getattr(req, "padding_match_iou", 0.2))))
        predicted_polygons = _merge_padding_predictions(
            predicted_polygons,
            padded_polygons,
            getattr(req, "padding_merge_mode", "large_only"),
            large_threshold_ratio,
            match_iou,
            cv2,
            np,
        )
        padding_info = {
            "enabled": True,
            "pad_y_percent": pad_y_ratio * 100.0,
            "pad_x_percent": pad_x_ratio * 100.0,
            "pad_y_px": pad_y_px,
            "pad_x_px": pad_x_px,
            "large_threshold": large_threshold_ratio * 100.0,
            "merge_mode": getattr(req, "padding_merge_mode", "large_only"),
            "match_iou": match_iou,
            "normal_count": padding_info["normal_count"],
            "padded_count": len(padded_polygons),
            "merged_count": len(predicted_polygons),
        }

    print(
        f"[inference] auto_segment completed model={req.model_name} "
        f"polygons={len(predicted_polygons)} padding={padding_info['enabled']} "
        f"elapsed_sec={time.time() - request_started:.2f}",
        flush=True,
    )
    return {"polygons": predicted_polygons, "imgsz": model_imgsz, "padding_inference": padding_info}


class AutoLabelPendingRequest(BaseModel):
    model_a: str
    model_b: str
    conf: float = 0.25


@app.post("/api/dataset/{dataset_name}/auto_label_pending")
def api_auto_label_pending(dataset_name: str, req: AutoLabelPendingRequest):
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not req.model_a or not req.model_b:
        raise HTTPException(status_code=400, detail="Select two models")
    if req.model_a == req.model_b:
        raise HTTPException(status_code=400, detail="Select two different models")

    progress_key = f"auto_label_{dataset_name}"
    current_progress = PROGRESS_CACHE.get(progress_key) or {}
    if current_progress.get("status") == "running":
        raise HTTPException(status_code=409, detail="Auto labeling is already running")

    model_path_a = MODELS_DIR / req.model_a
    model_path_b = MODELS_DIR / req.model_b
    if not model_path_a.exists() or not model_path_b.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    yaml_path = get_yaml_path(dataset_name)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="data.yaml not found")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    dataset_classes = data_yaml.get("names", []) or []

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load inference libraries: {e}")

    pending_img_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "images"
    pending_lbl_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "labels"
    pending_coco_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / MULTIPOLYGON_LABEL_DIR
    pending_lbl_dir.mkdir(parents=True, exist_ok=True)
    pending_coco_dir.mkdir(parents=True, exist_ok=True)

    negatives = load_negative_samples(dataset_name)
    auto_unreviewed = load_auto_labeled_unreviewed(dataset_name)

    target_images = []
    skipped_negative = 0
    skipped_reviewed = 0
    if pending_img_dir.exists():
        for img_file in sorted(pending_img_dir.glob("*.*")):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_file.name in negatives:
                skipped_negative += 1
                continue
            lbl_file = pending_lbl_dir / f"{img_file.stem}.txt"
            if img_file.name not in auto_unreviewed and _label_has_segmentation_annotations(lbl_file):
                skipped_reviewed += 1
                continue
            target_images.append(img_file)

    started_at = time.time()

    def set_auto_label_progress(current, total, last_duration=0.0, message="", status="running"):
        PROGRESS_CACHE[progress_key] = {
            "status": status,
            "current": current,
            "total": total,
            "last_duration": last_duration,
            "phase": "auto_label" if status == "running" else status,
            "message": message,
            "phase_started_at": started_at,
            "started_at": started_at,
            "updated_at": time.time(),
        }

    total = len(target_images)
    set_auto_label_progress(0, total, message=f"Auto Label 0/{total}")
    if total == 0:
        set_auto_label_progress(
            0,
            0,
            message=f"No pending images to auto label. Skipped reviewed {skipped_reviewed}, negative {skipped_negative}.",
            status="done",
        )
        return {
            "status": "ok",
            "processed": 0,
            "skipped_reviewed": skipped_reviewed,
            "skipped_negative": skipped_negative,
        }

    devices = _resolve_auto_label_devices(torch)
    use_cuda_a = yolo_uses_cuda(torch, devices[0])
    use_cuda_b = yolo_uses_cuda(torch, devices[1])
    log_yolo_device(torch, devices[0], f"auto_label:{req.model_a}")
    log_yolo_device(torch, devices[1], f"auto_label:{req.model_b}")

    model_a, imgsz_a, lock_a = _get_cached_yolo_model(model_path_a, devices[0], f"auto_label:{req.model_a}", YOLO)
    model_b, imgsz_b, lock_b = _get_cached_yolo_model(model_path_b, devices[1], f"auto_label:{req.model_b}", YOLO)

    conf = max(0.0, min(1.0, float(req.conf)))

    def run_model(model, lock, img_path, device, use_cuda, imgsz):
        with lock:
            results = model.predict(
                source=str(img_path),
                save=False,
                conf=conf,
                verbose=False,
                retina_masks=True,
                device=device,
                half=use_cuda,
                imgsz=imgsz,
            )
        return _collect_yolo_result_polygons(results[0] if results else None, dataset_classes)

    processed = 0
    total_polygons = 0
    errors = []
    for idx, img_file in enumerate(target_images, start=1):
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(run_model, model_a, lock_a, img_file, devices[0], use_cuda_a, imgsz_a)
                future_b = executor.submit(run_model, model_b, lock_b, img_file, devices[1], use_cuda_b, imgsz_b)
                predictions = future_a.result() + future_b.result()

            polygons = _group_auto_label_predictions(predictions)
            lbl_file = pending_lbl_dir / f"{img_file.stem}.txt"
            coco_file = pending_coco_dir / f"{img_file.stem}.json"
            _write_multipolygon_label(coco_file, polygons)
            _write_yolo_label_from_polygons(lbl_file, polygons)

            auto_unreviewed.add(img_file.name)
            save_auto_labeled_unreviewed(dataset_name, auto_unreviewed)
            processed += 1
            total_polygons += len(polygons)
            set_auto_label_progress(
                idx,
                total,
                last_duration=time.time() - t0,
                message=f"Auto Label {idx}/{total} · {img_file.name} · polygons {len(polygons)}",
            )
        except Exception as e:
            errors.append({"image": img_file.name, "error": str(e)})
            set_auto_label_progress(
                idx,
                total,
                last_duration=time.time() - t0,
                message=f"Auto Label {idx}/{total} · error: {img_file.name}",
            )

    status = "error" if errors else "done"
    set_auto_label_progress(
        total,
        total,
        message=f"Auto labeling complete: {processed}/{total}, polygons {total_polygons}, errors {len(errors)}",
        status=status,
    )
    return {
        "status": "ok" if not errors else "partial_error",
        "processed": processed,
        "target_count": total,
        "total_polygons": total_polygons,
        "skipped_reviewed": skipped_reviewed,
        "skipped_negative": skipped_negative,
        "errors": errors[:20],
        "error_count": len(errors),
        "devices": [str(devices[0]), str(devices[1])],
        "imgsz": [imgsz_a, imgsz_b],
    }


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

    if req.shrink_percent == 0:
        return {"polygons": [_polygon_response(p) for p in req.polygons]}

    target_set = set(req.indices) if req.indices is not None else None
    out = []
    for i, poly in enumerate(req.polygons):
        passthrough = (target_set is not None and i not in target_set) or len(poly.points) < 3
        if passthrough:
            out.append(_polygon_response(poly))
            continue

        pts_arr = np.array([(pt.x, pt.y) for pt in poly.points], dtype=np.float64)
        min_xy = pts_arr.min(axis=0)
        max_xy = pts_arr.max(axis=0)
        bbox_w = float(max_xy[0] - min_xy[0])
        bbox_h = float(max_xy[1] - min_xy[1])
        min_side = min(bbox_w, bbox_h)
        if min_side <= 0:
            out.append(_polygon_response(poly, []))
            continue

        is_expand = req.shrink_percent < 0
        abs_percent = abs(req.shrink_percent)

        shrink_px = max(1, int(round(min_side * abs_percent / 100.0)))
        pad = shrink_px + 2
        mask_w = int(np.ceil(bbox_w)) + 2 * pad
        mask_h = int(np.ceil(bbox_h)) + 2 * pad
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        local_pts = np.round(pts_arr - min_xy + pad).astype(np.int32)
        cv2.fillPoly(mask, [local_pts], 1)

        k = 2 * shrink_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if is_expand:
            eroded = cv2.dilate(mask, kernel)
        else:
            eroded = cv2.erode(mask, kernel)

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out.append(_polygon_response(poly, []))
            continue

        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.0, True)
        if len(contour) < 3:
            out.append(_polygon_response(poly, []))
            continue

        new_pts = [
            {"x": float(c[0][0] - pad + min_xy[0]), "y": float(c[0][1] - pad + min_xy[1])}
            for c in contour
        ]
        out.append(_polygon_response(poly, new_pts))

    return {"polygons": out}

PROGRESS_CACHE = {}

@app.get("/api/dataset/{dataset_name}/progress/{task_type}")
def api_get_progress(dataset_name: str, task_type: str):
    key = f"{task_type}_{dataset_name}"
    return PROGRESS_CACHE.get(key, {
        "status": "idle",
        "current": 0,
        "total": 0,
        "last_duration": 0.0,
        "phase": "",
        "message": "",
        "started_at": None,
        "updated_at": None,
    })

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
            out.append(_polygon_response(poly))
            continue

        pts = np.array([(pt.x, pt.y) for pt in poly.points], dtype=np.float64)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)

        x0 = max(0, int(np.floor(min_xy[0])) - margin)
        y0 = max(0, int(np.floor(min_xy[1])) - margin)
        x1 = min(W_img, int(np.ceil(max_xy[0])) + margin)
        y1 = min(H_img, int(np.ceil(max_xy[1])) + margin)

        if x1 - x0 < 4 or y1 - y0 < 4:
            out.append(_polygon_response(poly))
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
            out.append(_polygon_response(poly))
            continue

        fg_mask = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out.append(_polygon_response(poly))
            continue

        contour = max(contours, key=cv2.contourArea)
        if smooth_eps > 0.0:
            contour = cv2.approxPolyDP(contour, smooth_eps, True)
        if len(contour) < 3:
            out.append(_polygon_response(poly))
            continue

        new_pts = [
            {"x": float(c[0][0] + x0), "y": float(c[0][1] + y0)}
            for c in contour
        ]
        out.append(_polygon_response(poly, new_pts))

    return {"polygons": out}


class SplitPolygonRequest(BaseModel):
    polygon: Polygon
    stroke: List[Point]
    brush_radius: float = 8.0
    image_width: int
    image_height: int


@app.post("/api/split_polygon_by_stroke")
async def api_split_polygon_by_stroke(req: SplitPolygonRequest):
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="cv2/numpy not installed")

    width = max(1, min(int(req.image_width), 10000))
    height = max(1, min(int(req.image_height), 10000))
    if len(req.polygon.points) < 3:
        return {"polygons": []}

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(
        [
            [
                int(round(max(0, min(width - 1, pt.x)))),
                int(round(max(0, min(height - 1, pt.y)))),
            ]
            for pt in req.polygon.points
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)

    radius = max(1.0, min(float(req.brush_radius), 200.0))
    thickness = max(1, int(round(radius * 2)))
    stroke_pts = [
        (
            int(round(max(0, min(width - 1, pt.x)))),
            int(round(max(0, min(height - 1, pt.y)))),
        )
        for pt in req.stroke
    ]
    if len(stroke_pts) == 1:
        cv2.circle(mask, stroke_pts[0], int(round(radius)), 0, -1, lineType=cv2.LINE_AA)
    else:
        for p1, p2 in zip(stroke_pts, stroke_pts[1:]):
            cv2.line(mask, p1, p2, 0, thickness=thickness, lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    min_area = max(4.0, radius * radius * 0.25)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(0.75, min(3.0, radius * 0.08))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = [{"x": float(c[0][0]), "y": float(c[0][1])} for c in approx]
        if not _valid_polygon_piece(points):
            continue
        out.append({
            "labelId": req.polygon.labelId,
            "classId": req.polygon.classId,
            "points": points,
        })

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
        import time
        import torch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Auto Check libraries: {e}")

    device = resolve_yolo_device(torch)
    use_cuda = yolo_uses_cuda(torch, device)
    log_yolo_device(torch, device, "auto_check")
    model = YOLO(str(model_path))
    model_imgsz = log_yolo_imgsz(model, "auto_check")
    scores = {}

    # List up target images first to get the total count for ETA estimation
    target_images = []
    for split in TRAINING_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists(): continue
        for img_file in images_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]: continue
            target_images.append((img_file, labels_dir / (img_file.stem + ".txt")))

    total_images = len(target_images)
    PROGRESS_CACHE[f"auto_check_{dataset_name}"] = {
        "status": "running",
        "current": 0,
        "total": total_images,
        "last_duration": 0.0
    }

    try:
        for idx, (img_file, lbl_file) in enumerate(target_images):
            t0 = time.time()
            
            # Predict
            results = model.predict(source=str(img_file), save=False, conf=0.25, verbose=False, retina_masks=True, device=device, half=use_cuda, imgsz=model_imgsz)
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

            # Clean up VRAM immediately to avoid memory accumulation
            del res
            if idx % 20 == 0:
                if use_cuda:
                    torch.cuda.empty_cache()

            elapsed = time.time() - t0
            PROGRESS_CACHE[f"auto_check_{dataset_name}"] = {
                "status": "running",
                "current": idx + 1,
                "total": total_images,
                "last_duration": elapsed
            }
        
        PROGRESS_CACHE[f"auto_check_{dataset_name}"] = {
            "status": "done",
            "current": total_images,
            "total": total_images,
            "last_duration": 0.0
        }
    except Exception as e:
        PROGRESS_CACHE[f"auto_check_{dataset_name}"] = {
            "status": "error",
            "current": len(scores),
            "total": total_images,
            "last_duration": 0.0
        }
        raise e

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
        import torch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Auto Check libraries: {e}")
        
    decoded_img_path = urllib.parse.unquote(req.image_path)
    physical_img_path = BASE_DIR / decoded_img_path.lstrip("/")
    
    if not physical_img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    device = resolve_yolo_device(torch)
    use_cuda = yolo_uses_cuda(torch, device)
    log_yolo_device(torch, device, "auto_check_single")
    model, model_imgsz, model_predict_lock = _get_cached_yolo_model(
        model_path, device, "auto_check_single", YOLO
    )
    
    lbl_file = physical_img_path.parent.parent / "labels" / (physical_img_path.stem + ".txt")
    
    if getattr(req, 'use_denoise', False):
        img_bgr = cv2.imread(str(physical_img_path))
        if img_bgr is not None:
            denoised_img = cv2.fastNlMeansDenoisingColored(
                img_bgr, None, float(req.h_lum), float(req.h_col), int(req.tw), int(req.sw)
            )
            with model_predict_lock:
                results = model.predict(source=denoised_img, save=False, conf=0.25, verbose=False, retina_masks=True, device=device, half=use_cuda, imgsz=model_imgsz)
        else:
            with model_predict_lock:
                results = model.predict(source=str(physical_img_path), save=False, conf=0.25, verbose=False, retina_masks=True, device=device, half=use_cuda, imgsz=model_imgsz)
    else:
        with model_predict_lock:
            results = model.predict(source=str(physical_img_path), save=False, conf=0.25, verbose=False, retina_masks=True, device=device, half=use_cuda, imgsz=model_imgsz)
        
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
    batch_size: Optional[int] = None
    # Kept for request compatibility with older clients. Current scoring uses
    # fixed point penalties instead of weighted IoU.
    weight_mask: float = 0.65
    weight_class: float = 0.15
    weight_critical: float = 0.20
    # Class names that should be detected with high spatial accuracy (case-insensitive).
    # Names absent from the dataset's data.yaml are silently dropped.
    critical_classes: List[str] = ["penis", "pussy", "anus"]
    # Classes to ignore completely during benchmark scoring. Ground-truth masks,
    # predicted masks, class recall, false positives, and critical metrics for
    # these classes are excluded.
    ignored_classes: List[str] = ["nipple"]

@app.post("/api/dataset/{dataset_name}/benchmark")
def api_benchmark(dataset_name: str, req: BenchmarkRequest):
    if not req.model_names:
        raise HTTPException(status_code=400, detail="No models specified")

    valid_splits = {"train", "valid", "test", "val", "all"}
    if req.split not in valid_splits:
        raise HTTPException(status_code=400, detail=f"Invalid split: {req.split}")

    benchmark_key = f"benchmark_{dataset_name}"
    current_progress = PROGRESS_CACHE.get(benchmark_key) or {}
    if current_progress.get("status") == "running":
        raise HTTPException(status_code=409, detail="Benchmark is already running")

    splits = list(TRAINING_SPLITS) if req.split == "all" else [req.split]

    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch
        import time
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark libraries: {e}")

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
    ignored_ids = []
    ignored_names_used = []
    for n in req.ignored_classes:
        key = n.lower()
        if key in name_to_id:
            ignored_ids.append(name_to_id[key])
            ignored_names_used.append(dataset_classes[name_to_id[key]])
    ignored_id_set = set(ignored_ids)

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

    # Pre-parse ground truth polygons instead of rasterizing all 640x640 masks into memory.
    # This prevents massive RAM consumption (OOM) on large datasets.
    gt_polygons = []  # list of list of dict: [{"cls_id": int, "pts": np.ndarray}]
    gt_class_sets = []
    for lbl_file in label_paths:
        polys = []
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
                        if cls_id in ignored_id_set:
                            continue
                        coords = [float(x) for x in parts[1:]]
                        pts = np.array(
                            [[int(coords[i] * W), int(coords[i + 1] * H)] for i in range(0, len(coords), 2)],
                            dtype=np.int32,
                        )
                        polys.append({"cls_id": cls_id, "pts": pts})
                        cls_set.add(cls_id)
        gt_polygons.append(polys)
        gt_class_sets.append(cls_set)

    def rasterize_gt(polys, critical_ids_set):
        overall = np.zeros((H, W), dtype=np.uint8)
        crit = {}
        for poly in polys:
            cls_id = poly["cls_id"]
            pts = poly["pts"]
            cv2.fillPoly(overall, [pts], 1)
            if cls_id in critical_ids_set:
                if cls_id not in crit:
                    crit[cls_id] = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(crit[cls_id], [pts], 1)
        return overall, crit

    popcount_table = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

    def packed_popcount(mask):
        if mask is None:
            return 0
        return int(popcount_table[mask].sum())

    def packed_coverage_score(gt_packed, gt_area, pred_packed):
        """Coverage metrics using bit-packed masks without unpacking to 640x640."""
        pred_area = packed_popcount(pred_packed)
        inter = packed_popcount(np.bitwise_and(gt_packed, pred_packed))
        union = gt_area + pred_area - inter
        iou = 1.0 if union == 0 else inter / union
        coverage = 1.0 if gt_area == 0 and pred_area == 0 else (inter / gt_area if gt_area > 0 else 0.0)
        outside = max(pred_area - inter, 0)
        false_positive_ratio = outside / pred_area if pred_area > 0 else 0.0
        return {
            "score": coverage,
            "coverage": coverage,
            "iou": iou,
            "under_penalty": 1.0 - coverage if gt_area > 0 else 0.0,
            "false_positive_ratio": false_positive_ratio,
        }

    def packed_critical_coverage(gt_info, pred_packed):
        gt_area = gt_info["area"]
        if gt_area == 0:
            return None
        if pred_packed is None:
            return 0.0
        inter = packed_popcount(np.bitwise_and(gt_info["packed"], pred_packed))
        return inter / gt_area

    gt_mask_cache = []
    for polys in gt_polygons:
        gt_o, gt_crit_masks = rasterize_gt(polys, critical_id_set)
        gt_o_bool = gt_o > 0
        crit_cache = {}
        for cid, crit_mask in gt_crit_masks.items():
            crit_bool = crit_mask > 0
            crit_cache[cid] = {
                "packed": np.packbits(crit_bool),
                "area": int(crit_bool.sum()),
            }
        gt_mask_cache.append({
            "overall": np.packbits(gt_o_bool),
            "overall_area": int(gt_o_bool.sum()),
            "critical": crit_cache,
        })


    device = resolve_yolo_device(torch)
    use_cuda = yolo_uses_cuda(torch, device)
    benchmark_batch_size = _resolve_benchmark_batch_size(req.batch_size)
    log_yolo_device(torch, device, "benchmark")
    print(f"[inference] benchmark using batch={benchmark_batch_size}", flush=True)
    paths_str = [str(p) for p in image_paths]

    CRITICAL_MISS_MAX_POINTS = 50.0
    FALSE_POSITIVE_MAX_POINTS = 20.0

    def benchmark_penalty(mean_critical_coverage, mean_false_positive_ratio):
        critical_penalty = 0.0
        if mean_critical_coverage is not None:
            critical_penalty = (1.0 - mean_critical_coverage) * CRITICAL_MISS_MAX_POINTS
        false_positive_penalty = mean_false_positive_ratio * FALSE_POSITIVE_MAX_POINTS
        penalty_points = min(100.0, critical_penalty + false_positive_penalty)
        return {
            "score": max(0.0, 100.0 - penalty_points),
            "penalty": penalty_points / 100.0,
            "penalty_points": penalty_points,
            "critical_miss_penalty": critical_penalty,
            "false_positive_penalty": false_positive_penalty,
        }

    benchmark_started_at = time.time()
    model_phase_started_at = time.time()

    def set_benchmark_progress(
        current,
        total,
        last_duration=0.0,
        phase="models",
        message="",
        phase_started_at=None,
        status="running",
    ):
        now = time.time()
        PROGRESS_CACHE[benchmark_key] = {
            "status": status,
            "current": current,
            "total": total,
            "last_duration": last_duration,
            "phase": phase,
            "message": message,
            "phase_started_at": phase_started_at if phase_started_at is not None else time.time(),
            "started_at": benchmark_started_at,
            "updated_at": now,
        }

    total_models = len(req.model_names)
    set_benchmark_progress(
        0,
        total_models,
        phase="models",
        message=f"Models 0/{total_models} · batch {benchmark_batch_size}",
        phase_started_at=model_phase_started_at,
    )

    model_caches = {}
    results = []
    adaptive_batch_size_by_imgsz = {}
    for m_idx, model_name in enumerate(req.model_names):
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
                "batch_size": benchmark_batch_size,
            })
            set_benchmark_progress(
                m_idx + 1,
                total_models,
                0.0,
                phase="models",
                message=f"Models {m_idx + 1}/{total_models} · batch {benchmark_batch_size}",
                phase_started_at=model_phase_started_at,
            )
            continue

        mask_scores = []
        raw_ious = []
        mask_coverages = []
        undercoverage_penalties = []
        false_positive_ratios = []
        class_recalls = []
        critical_coverage_values = []
        model_predictions = []
        t0 = time.time()
        model = None
        current_batch_size = benchmark_batch_size
        imgsz_key = "default"
        try:
            model = YOLO(str(model_path))
            model_imgsz = log_yolo_imgsz(model, f"benchmark:{model_name}")
            imgsz_key = json.dumps(model_imgsz, sort_keys=True)
            current_batch_size = adaptive_batch_size_by_imgsz.get(imgsz_key, benchmark_batch_size)
            # Build a translation table from model class IDs to dataset class IDs
            # via name match (case-insensitive). Predictions whose model class has
            # no corresponding dataset class are ignored for class-aware metrics.
            model_to_dataset = {}
            for mid, mname in (model.names or {}).items():
                key = str(mname).lower()
                if key in name_to_id:
                    model_to_dataset[int(mid)] = name_to_id[key]

            while True:
                mask_scores = []
                raw_ious = []
                mask_coverages = []
                undercoverage_penalties = []
                false_positive_ratios = []
                class_recalls = []
                critical_coverage_values = []
                model_predictions = []
                try:
                    predict_kwargs = {
                        "save": False,
                        "conf": 0.25,
                        "verbose": False,
                        "retina_masks": True,
                        "device": device,
                        "half": use_cuda,
                        "imgsz": model_imgsz,
                    }

                    def iter_prediction_results():
                        if current_batch_size <= 1:
                            for path_str in paths_str:
                                single_results = model.predict(
                                    source=path_str,
                                    stream=False,
                                    **predict_kwargs,
                                )
                                yield single_results[0] if single_results else None
                        else:
                            yield from model.predict(
                                source=paths_str,
                                stream=True,
                                batch=current_batch_size,
                                **predict_kwargs,
                            )

                    result_stream = iter_prediction_results()
                    for idx, res in enumerate(result_stream):
                        pred_overall = np.zeros((H, W), dtype=np.uint8)
                        pred_critical = {}
                        pred_class_set = set()
                        if res is not None and res.masks is not None and res.boxes is not None:
                            cls_ids = res.boxes.cls.tolist()
                            for mask_xyn, mcls_f in zip(res.masks.xyn, cls_ids):
                                mcls = int(mcls_f)
                                pts = np.array(
                                    [[int(x * W), int(y * H)] for x, y in mask_xyn],
                                    dtype=np.int32,
                                )
                                dcls = model_to_dataset.get(mcls)
                                if dcls in ignored_id_set:
                                    continue
                                cv2.fillPoly(pred_overall, [pts], 1)
                                if dcls is not None:
                                    pred_class_set.add(dcls)
                                    if dcls in critical_id_set:
                                        if dcls not in pred_critical:
                                            pred_critical[dcls] = np.zeros((H, W), dtype=np.uint8)
                                        cv2.fillPoly(pred_critical[dcls], [pts], 1)

                        pred_overall_packed = np.packbits(pred_overall > 0)
                        pred_critical_packed = {
                            cid: np.packbits(mask > 0)
                            for cid, mask in pred_critical.items()
                        }

                        # Only cache predictions if combination evaluation is possible (>= 2 models).
                        # To save RAM, masks stay bit-packed and are scored with popcount.
                        if len(req.model_names) >= 2:
                            model_predictions.append({
                                "overall": pred_overall_packed,
                                "class_set": pred_class_set,
                                "critical": pred_critical_packed,
                            })

                        gt_info = gt_mask_cache[idx]

                        mask_metric = packed_coverage_score(
                            gt_info["overall"],
                            gt_info["overall_area"],
                            pred_overall_packed,
                        )
                        mask_scores.append(float(mask_metric["score"]))
                        raw_ious.append(float(mask_metric["iou"]))
                        mask_coverages.append(float(mask_metric["coverage"]))
                        undercoverage_penalties.append(float(mask_metric["under_penalty"]))
                        false_positive_ratios.append(float(mask_metric["false_positive_ratio"]))

                        gt_cls = gt_class_sets[idx]
                        if gt_cls:
                            matched = gt_cls & pred_class_set
                            class_recalls.append(len(matched) / len(gt_cls))
                        # When GT has no classes, recall is undefined — skip rather than
                        # rewarding empty predictions with a free 1.0.

                        per_img_critical = []
                        for cid in critical_ids:
                            gt_crit_info = gt_info["critical"].get(cid)
                            if gt_crit_info is None:
                                continue
                            crit_score = packed_critical_coverage(
                                gt_crit_info,
                                pred_critical_packed.get(cid),
                            )
                            if crit_score is not None:
                                per_img_critical.append(float(crit_score))
                        if per_img_critical:
                            critical_coverage_values.append(float(np.mean(per_img_critical)))

                        # Explicitly delete the prediction results object to free VRAM immediately
                        del res
                    break
                except RuntimeError as e:
                    if use_cuda and current_batch_size > 1 and _is_cuda_oom_error(e):
                        next_batch_size = max(1, current_batch_size // 2)
                        print(
                            f"[inference] benchmark:{model_name} CUDA OOM at batch={current_batch_size}; "
                            f"retrying batch={next_batch_size}",
                            flush=True,
                        )
                        current_batch_size = next_batch_size
                        adaptive_batch_size_by_imgsz[imgsz_key] = current_batch_size
                        torch.cuda.empty_cache()
                        continue
                    raise
        except Exception as e:
            mean_mask = float(np.mean(mask_scores)) if mask_scores else None
            mean_class = float(np.mean(class_recalls)) if class_recalls else None
            mean_crit = float(np.mean(critical_coverage_values)) if critical_coverage_values else None
            mean_false_positive = float(np.mean(false_positive_ratios)) if false_positive_ratios else 0.0
            penalty_info = benchmark_penalty(mean_crit, mean_false_positive)
            if not mask_scores:
                penalty_info = {
                    "score": 0.0,
                    "penalty": 1.0,
                    "penalty_points": 100.0,
                    "critical_miss_penalty": 0.0,
                    "false_positive_penalty": 0.0,
                }
            results.append({
                "model": model_name,
                "error": str(e),
                "score": round(penalty_info["score"], 2),
                "penalty": round(penalty_info["penalty"], 4),
                "penalty_points": round(penalty_info["penalty_points"], 2),
                "critical_miss_penalty": round(penalty_info["critical_miss_penalty"], 2),
                "false_positive_penalty": round(penalty_info["false_positive_penalty"], 2),
                "mean_iou": round(mean_mask, 4) if mean_mask is not None else 0.0,
                "raw_iou": round(float(np.mean(raw_ious)), 4) if raw_ious else 0.0,
                "mask_coverage": round(float(np.mean(mask_coverages)), 4) if mask_coverages else 0.0,
                "overreach": round(mean_false_positive, 4),
                "undercoverage": round(float(np.mean(undercoverage_penalties)), 4) if undercoverage_penalties else 0.0,
                "median_iou": round(float(np.median(mask_scores)), 4) if mask_scores else 0.0,
                "class_recall": round(mean_class, 4) if mean_class is not None else None,
                "critical_iou": round(mean_crit, 4) if mean_crit is not None else None,
                "image_count": len(mask_scores),
                "elapsed_sec": round(time.time() - t0, 2),
                "batch_size": current_batch_size,
            })
            set_benchmark_progress(
                m_idx + 1,
                total_models,
                time.time() - t0,
                phase="models",
                message=f"Models {m_idx + 1}/{total_models} · batch {current_batch_size}",
                phase_started_at=model_phase_started_at,
            )
            continue
        finally:
            del model
            if use_cuda:
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        mean_mask = float(np.mean(mask_scores)) if mask_scores else None
        mean_class = float(np.mean(class_recalls)) if class_recalls else None
        mean_crit = float(np.mean(critical_coverage_values)) if critical_coverage_values else None
        mean_false_positive = float(np.mean(false_positive_ratios)) if false_positive_ratios else 0.0
        penalty_info = benchmark_penalty(mean_crit, mean_false_positive)

        results.append({
            "model": model_name,
            "score": round(penalty_info["score"], 2),
            "penalty": round(penalty_info["penalty"], 4),
            "penalty_points": round(penalty_info["penalty_points"], 2),
            "critical_miss_penalty": round(penalty_info["critical_miss_penalty"], 2),
            "false_positive_penalty": round(penalty_info["false_positive_penalty"], 2),
            "mean_iou": round(mean_mask, 4) if mean_mask is not None else 0.0,
            "raw_iou": round(float(np.mean(raw_ious)), 4) if raw_ious else 0.0,
            "mask_coverage": round(float(np.mean(mask_coverages)), 4) if mask_coverages else 0.0,
            "overreach": round(mean_false_positive, 4),
            "undercoverage": round(float(np.mean(undercoverage_penalties)), 4) if undercoverage_penalties else 0.0,
            "median_iou": round(float(np.median(mask_scores)), 4) if mask_scores else 0.0,
            "class_recall": round(mean_class, 4) if mean_class is not None else None,
            "critical_iou": round(mean_crit, 4) if mean_crit is not None else None,
            "image_count": len(mask_scores),
            "elapsed_sec": round(elapsed, 2),
            "batch_size": current_batch_size,
        })
        model_caches[model_name] = model_predictions

        set_benchmark_progress(
            m_idx + 1,
            total_models,
            elapsed,
            phase="models",
            message=f"Models {m_idx + 1}/{total_models} · batch {current_batch_size}",
            phase_started_at=model_phase_started_at,
        )

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    # Evaluate 2-model combinations
    import itertools
    combo_results = []
    combo_min_score = 80.0
    combo_max_penalty = 0.05
    combo_rank_limit = 80
    result_by_model = {r.get("model"): r for r in results}
    valid_models = []
    for ranked_result in results[:combo_rank_limit]:
        model_name = ranked_result.get("model")
        result = result_by_model.get(model_name) or {}
        if (
            model_name in model_caches
            and len(model_caches[model_name]) == len(image_paths)
            and float(result.get("score") or 0.0) >= combo_min_score
            and float(result.get("penalty") if result.get("penalty") is not None else 1.0) <= combo_max_penalty
        ):
            valid_models.append(model_name)

    combo_pairs = list(itertools.combinations(valid_models, 2))
    combo_phase_started_at = time.time()
    if combo_pairs:
        print(
            f"[benchmark] combinations using packed popcount: models={len(valid_models)} pairs={len(combo_pairs)}",
            flush=True,
        )
        set_benchmark_progress(
            0,
            len(combo_pairs),
            phase="combinations",
            message=f"Combinations 0/{len(combo_pairs)}",
            phase_started_at=combo_phase_started_at,
        )
    if len(combo_pairs) > 0:
        for combo_idx, (modelA_name, modelB_name) in enumerate(combo_pairs):
            combo_t0 = time.time()
            cache_A = model_caches[modelA_name]
            cache_B = model_caches[modelB_name]

            combo_mask_scores = []
            combo_raw_ious = []
            combo_coverages = []
            combo_undercoverage_penalties = []
            combo_false_positive_ratios = []
            combo_class_recalls = []
            combo_critical_coverages = []

            for idx in range(len(image_paths)):
                pred_A = cache_A[idx]
                pred_B = cache_B[idx]

                # 1. Combined overall coverage-first mask score.
                # Keep masks bit-packed: OR/AND + byte popcount is much faster
                # than unpacking every pair to a 640x640 bool array.
                gt_info = gt_mask_cache[idx]
                comb_overall = np.bitwise_or(pred_A["overall"], pred_B["overall"])
                combo_metric = packed_coverage_score(
                    gt_info["overall"],
                    gt_info["overall_area"],
                    comb_overall,
                )
                combo_mask_scores.append(float(combo_metric["score"]))
                combo_raw_ious.append(float(combo_metric["iou"]))
                combo_coverages.append(float(combo_metric["coverage"]))
                combo_undercoverage_penalties.append(float(combo_metric["under_penalty"]))
                combo_false_positive_ratios.append(float(combo_metric["false_positive_ratio"]))

                # 2. Combined class recall
                comb_cls_set = pred_A["class_set"] | pred_B["class_set"]
                gt_cls = gt_class_sets[idx]
                if gt_cls:
                    matched = gt_cls & comb_cls_set
                    combo_class_recalls.append(len(matched) / len(gt_cls))

                # 3. Combined critical coverage
                per_img_critical = []
                for cid in critical_ids:
                    gt_crit_info = gt_info["critical"].get(cid)
                    if gt_crit_info is None:
                        continue

                    packed_mask_A = pred_A["critical"].get(cid)
                    packed_mask_B = pred_B["critical"].get(cid)

                    if packed_mask_A is None and packed_mask_B is None:
                        pr_m = None
                    elif packed_mask_A is not None and packed_mask_B is None:
                        pr_m = packed_mask_A
                    elif packed_mask_A is None and packed_mask_B is not None:
                        pr_m = packed_mask_B
                    else:
                        pr_m = np.bitwise_or(packed_mask_A, packed_mask_B)

                    crit_score = packed_critical_coverage(gt_crit_info, pr_m)
                    if crit_score is not None:
                        per_img_critical.append(float(crit_score))

                if per_img_critical:
                    combo_critical_coverages.append(float(np.mean(per_img_critical)))

            mean_mask = float(np.mean(combo_mask_scores)) if combo_mask_scores else None
            mean_class = float(np.mean(combo_class_recalls)) if combo_class_recalls else None
            mean_crit = float(np.mean(combo_critical_coverages)) if combo_critical_coverages else None
            mean_false_positive = float(np.mean(combo_false_positive_ratios)) if combo_false_positive_ratios else 0.0
            penalty_info = benchmark_penalty(mean_crit, mean_false_positive)

            combo_results.append({
                "model_a": modelA_name,
                "model_b": modelB_name,
                "score": round(penalty_info["score"], 2),
                "penalty": round(penalty_info["penalty"], 4),
                "penalty_points": round(penalty_info["penalty_points"], 2),
                "critical_miss_penalty": round(penalty_info["critical_miss_penalty"], 2),
                "false_positive_penalty": round(penalty_info["false_positive_penalty"], 2),
                "mean_iou": round(mean_mask, 4) if mean_mask is not None else 0.0,
                "raw_iou": round(float(np.mean(combo_raw_ious)), 4) if combo_raw_ious else 0.0,
                "mask_coverage": round(float(np.mean(combo_coverages)), 4) if combo_coverages else 0.0,
                "overreach": round(mean_false_positive, 4),
                "undercoverage": round(float(np.mean(combo_undercoverage_penalties)), 4) if combo_undercoverage_penalties else 0.0,
                "class_recall": round(mean_class, 4) if mean_class is not None else None,
                "critical_iou": round(mean_crit, 4) if mean_crit is not None else None,
            })
            set_benchmark_progress(
                combo_idx + 1,
                len(combo_pairs),
                time.time() - combo_t0,
                phase="combinations",
                message=f"Combinations {combo_idx + 1}/{len(combo_pairs)}",
                phase_started_at=combo_phase_started_at,
            )

        # Sort combinations by composite score descending
        combo_results.sort(key=lambda x: x["score"], reverse=True)

    payload = {
        "status": "ok",
        "split": req.split,
        "image_count": len(image_paths),
        "device": "cuda" if use_cuda else "cpu",
        "batch_size": benchmark_batch_size,
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scoring": {
            "mode": "penalty_points",
            "base_score": 100.0,
            "critical_miss_max_points": CRITICAL_MISS_MAX_POINTS,
            "false_positive_max_points": FALSE_POSITIVE_MAX_POINTS,
        },
        "critical_classes_used": critical_names_used,
        "critical_classes_requested": list(req.critical_classes),
        "ignored_classes_used": ignored_names_used,
        "ignored_classes_requested": list(req.ignored_classes),
        "combination_filter": {
            "min_score": combo_min_score,
            "max_penalty": combo_max_penalty,
            "rank_limit": combo_rank_limit,
            "backend": "packed_popcount",
            "eligible_models": valid_models,
            "pair_count": len(combo_pairs),
        },
        "results": results,
        "combinations": combo_results,
    }

    bench_file = DATASETS_DIR / dataset_name / "benchmark.json"
    try:
        with open(bench_file, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

    final_total = len(combo_pairs) if combo_pairs else total_models
    final_current = final_total
    set_benchmark_progress(
        final_current,
        final_total,
        phase="done",
        message="Benchmark complete",
        phase_started_at=combo_phase_started_at if combo_pairs else model_phase_started_at,
        status="done",
    )

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
                     coco_file = _multipolygon_path_for_image(img_file)
                     all_images.append({
                         "img_path": img_file,
                         "label_path": label_file if label_file.exists() else None,
                         "coco_label_path": coco_file if coco_file.exists() else None,
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
        coco_dest_dir = DATASETS_DIR / dataset_name / target_split / MULTIPOLYGON_LABEL_DIR
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)
        coco_dest_dir.mkdir(parents=True, exist_ok=True)
        
        for item in imgs:
            shutil.move(str(item["img_path"]), str(img_dest_dir / item["img_path"].name))
            if item["label_path"]:
                shutil.move(str(item["label_path"]), str(lbl_dest_dir / item["label_path"].name))
            if item["coco_label_path"]:
                shutil.move(str(item["coco_label_path"]), str(coco_dest_dir / item["coco_label_path"].name))

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
    coco_label_path = _multipolygon_path_for_image(physical_img_path)
    
    img_dest_dir = DATASETS_DIR / dataset_name / req.target_split / "images"
    lbl_dest_dir = DATASETS_DIR / dataset_name / req.target_split / "labels"
    coco_dest_dir = DATASETS_DIR / dataset_name / req.target_split / MULTIPOLYGON_LABEL_DIR
    
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)
    coco_dest_dir.mkdir(parents=True, exist_ok=True)
    
    new_img_path = img_dest_dir / physical_img_path.name
    shutil.move(str(physical_img_path), str(new_img_path))
    
    if label_path.exists():
        new_lbl_path = lbl_dest_dir / label_path.name
        shutil.move(str(label_path), str(new_lbl_path))
    if coco_label_path.exists():
        new_coco_path = coco_dest_dir / coco_label_path.name
        shutil.move(str(coco_label_path), str(new_coco_path))
        
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
            coco_file = _multipolygon_path_for_image(img_file)
            if _label_has_segmentation_annotations(lbl_file):
                continue
            dest_path = dest_img_dir / img_file.name
            if dest_path.exists():
                # Same hash already in pending — drop the duplicate source.
                img_file.unlink()
            else:
                shutil.move(str(img_file), str(dest_path))
            if lbl_file.exists():
                lbl_file.unlink()
            if coco_file.exists():
                coco_file.unlink()
            moved += 1

    return {"status": "ok", "moved": moved}

@app.post("/api/dataset/{dataset_name}/move_pending_labeled_to_test")
async def api_move_pending_labeled_to_test(dataset_name: str):
    """Bulk-move completed pending images into the test split.

    Images are considered complete when they either have at least one valid
    YOLO segmentation annotation line, or are explicitly marked as negative
    samples. Empty unfinished labels remain pending.
    """
    if not (DATASETS_DIR / dataset_name).exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    negatives = load_negative_samples(dataset_name)
    auto_unreviewed = load_auto_labeled_unreviewed(dataset_name)
    src_img_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "images"
    src_lbl_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / "labels"
    src_coco_dir = DATASETS_DIR / dataset_name / PENDING_SPLIT / MULTIPOLYGON_LABEL_DIR
    dest_img_dir = DATASETS_DIR / dataset_name / "test" / "images"
    dest_lbl_dir = DATASETS_DIR / dataset_name / "test" / "labels"
    dest_coco_dir = DATASETS_DIR / dataset_name / "test" / MULTIPOLYGON_LABEL_DIR
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    dest_coco_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    duplicates = 0
    skipped_unreviewed = 0
    if not src_img_dir.exists():
        return {"status": "ok", "moved": moved, "duplicates": duplicates, "skipped_unreviewed": skipped_unreviewed}

    for img_file in list(src_img_dir.glob("*.*")):
        if img_file.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_file.name in auto_unreviewed:
            skipped_unreviewed += 1
            continue
        lbl_file = src_lbl_dir / (img_file.stem + ".txt")
        coco_file = src_coco_dir / (img_file.stem + ".json")
        is_negative = img_file.name in negatives
        has_annotations = _label_has_segmentation_annotations(lbl_file)
        if not is_negative and not has_annotations:
            continue

        dest_img_path = dest_img_dir / img_file.name
        dest_lbl_path = dest_lbl_dir / lbl_file.name
        dest_coco_path = dest_coco_dir / coco_file.name
        has_label_file = lbl_file.exists()
        has_coco_file = coco_file.exists()
        if dest_img_path.exists() and (not has_label_file or dest_lbl_path.exists()) and (not has_coco_file or dest_coco_path.exists()):
            # Avoid overwriting an existing test sample. With normalized hashed
            # filenames this is normally the same image, so remove pending copy.
            img_file.unlink()
            if has_label_file:
                lbl_file.unlink()
            if has_coco_file:
                coco_file.unlink()
            duplicates += 1
            continue

        if dest_img_path.exists():
            img_file.unlink()
        else:
            shutil.move(str(img_file), str(dest_img_path))

        if has_label_file:
            if dest_lbl_path.exists():
                lbl_file.unlink()
            else:
                shutil.move(str(lbl_file), str(dest_lbl_path))
        if has_coco_file:
            if dest_coco_path.exists():
                coco_file.unlink()
            else:
                shutil.move(str(coco_file), str(dest_coco_path))
        moved += 1

    return {"status": "ok", "moved": moved, "duplicates": duplicates, "skipped_unreviewed": skipped_unreviewed}

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
        (ds_dir / split / MULTIPOLYGON_LABEL_DIR).mkdir(parents=True, exist_ok=True)
        
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
        coco_dir = DATASETS_DIR / dataset_name / split / MULTIPOLYGON_LABEL_DIR

        if coco_dir.exists():
            for coco_file in coco_dir.glob("*.json"):
                label_file = labels_dir / f"{coco_file.stem}.txt"
                try:
                    polygons = _read_multipolygon_label(coco_file)
                except Exception:
                    continue

                file_changed = False
                new_polygons = []
                for poly in polygons:
                    pts = [(pt["x"], pt["y"]) for pt in poly["points"]]
                    has_oob = any(x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0 for x, y in pts)
                    if not has_oob:
                        new_polygons.append(poly)
                        continue

                    clipped = _clip_polygon_to_unit_square(pts)
                    file_changed = True
                    if len(clipped) < 3:
                        polygons_dropped += 1
                        continue
                    polygons_clipped += 1
                    new_polygons.append({
                        "labelId": poly["labelId"],
                        "classId": poly["classId"],
                        "points": [
                            {"x": max(0.0, min(1.0, x)), "y": max(0.0, min(1.0, y))}
                            for x, y in clipped
                        ],
                    })

                if file_changed:
                    _write_multipolygon_label(coco_file, new_polygons)
                    _write_yolo_label_from_polygons(label_file, new_polygons)
                    files_modified += 1

        if not labels_dir.exists():
            continue
        for lbl_file in labels_dir.glob("*.txt"):
            if _multipolygon_path_for_label(lbl_file).exists():
                continue
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

    progress_key = f"reload_{dataset_name}"
    t0 = time.time()
    PROGRESS_CACHE[progress_key] = {
        "status": "running",
        "current": 0,
        "total": 0,
        "last_duration": 0.0,
        "phase": "normalize",
        "message": "Normalizing filenames...",
    }

    seen_hashes = {}     # hash_stem -> final Path of the kept image
    rename_map = {}      # old image filename -> new image filename
    renamed = 0
    deleted_dup = 0
    deleted_orphan = 0

    # Phase 1: rename images (and labels) to {hash}.{ext}; drop content duplicates.
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        coco_dir = DATASETS_DIR / dataset_name / split / MULTIPOLYGON_LABEL_DIR
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
            old_coco = coco_dir / f"{img_file.stem}.json"

            if hash_stem in seen_hashes:
                img_file.unlink()
                if old_label.exists():
                    old_label.unlink()
                if old_coco.exists():
                    old_coco.unlink()
                deleted_dup += 1
                continue

            new_img_path = images_dir / f"{hash_stem}{new_suffix}"
            new_label = labels_dir / f"{hash_stem}.txt"
            new_coco = coco_dir / f"{hash_stem}.json"

            if new_img_path != img_file:
                if new_img_path.exists():
                    img_file.unlink()
                    if old_label.exists():
                        old_label.unlink()
                    if old_coco.exists():
                        old_coco.unlink()
                    deleted_dup += 1
                    continue
                img_file.rename(new_img_path)
                if old_label.exists():
                    if new_label.exists():
                        old_label.unlink()
                    else:
                        old_label.rename(new_label)
                if old_coco.exists():
                    coco_dir.mkdir(parents=True, exist_ok=True)
                    if new_coco.exists():
                        old_coco.unlink()
                    else:
                        old_coco.rename(new_coco)
                rename_map[old_name] = new_img_path.name
                renamed += 1

            seen_hashes[hash_stem] = new_img_path

    # Phase 2: drop label files whose image no longer exists.
    for split in ALL_SPLITS:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        coco_dir = DATASETS_DIR / dataset_name / split / MULTIPOLYGON_LABEL_DIR
        if not labels_dir.exists() and not coco_dir.exists():
            continue

        image_stems = set()
        if images_dir.exists():
            for img_file in images_dir.glob("*.*"):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    image_stems.add(img_file.stem)

        if labels_dir.exists():
            for lbl_file in list(labels_dir.glob("*.txt")):
                if lbl_file.stem not in image_stems:
                    lbl_file.unlink()
                    deleted_orphan += 1
        if coco_dir.exists():
            for coco_file in list(coco_dir.glob("*.json")):
                if coco_file.stem not in image_stems:
                    coco_file.unlink()
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

    auto_unreviewed = load_auto_labeled_unreviewed(dataset_name)
    if auto_unreviewed:
        new_auto_unreviewed = set()
        for name in auto_unreviewed:
            mapped = rename_map.get(name, name)
            if mapped in current_image_names:
                new_auto_unreviewed.add(mapped)
        if new_auto_unreviewed != auto_unreviewed:
            save_auto_labeled_unreviewed(dataset_name, new_auto_unreviewed)

    tag_result = {"tagged": 0, "missing_before": 0, "errors": []}
    try:
        current_image_items = []
        for img_path in seen_hashes.values():
            try:
                split = img_path.parent.parent.name
            except Exception:
                split = ""
            if split in ALL_SPLITS and img_path.exists():
                current_image_items.append((split, img_path))

        def update_reload_tag_progress(current, total, message):
            PROGRESS_CACHE[progress_key] = {
                "status": "running",
                "current": int(current),
                "total": int(total),
                "last_duration": time.time() - t0,
                "phase": "tagging",
                "message": message,
            }

        tag_result = tag_missing_images(
            DATASETS_DIR / dataset_name,
            current_image_items,
            BASE_DIR,
            progress_callback=update_reload_tag_progress,
        )
    except Exception as e:
        tag_result = {"tagged": 0, "missing_before": 0, "errors": [{"error": str(e)}]}
        PROGRESS_CACHE[progress_key] = {
            "status": "error",
            "current": 0,
            "total": 0,
            "last_duration": time.time() - t0,
            "phase": "tagging",
            "message": str(e),
        }
    else:
        PROGRESS_CACHE[progress_key] = {
            "status": "done",
            "current": int(tag_result.get("missing_before", 0)),
            "total": int(tag_result.get("missing_before", 0)),
            "last_duration": time.time() - t0,
            "phase": "done",
            "message": f"Reload complete. Tagged {int(tag_result.get('tagged', 0))}/{int(tag_result.get('missing_before', 0))} image(s).",
        }

    return {
        "status": "ok",
        "renamed": renamed,
        "deleted_duplicates": deleted_dup,
        "deleted_orphans": deleted_orphan,
        "tagging": tag_result,
    }


class TagSearchRequest(BaseModel):
    positive_tags: List[str] = []
    negative_tags: List[str] = []
    split: str = "all"
    limit: int = 500


@app.post("/api/dataset/{dataset_name}/tag_search")
async def api_tag_search(dataset_name: str, req: TagSearchRequest):
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if req.split not in set(ALL_SPLITS + ["all"]):
        raise HTTPException(status_code=400, detail="Invalid split")

    positive = []
    for tag in req.positive_tags:
        positive.extend(split_tags(tag))
    negative = []
    for tag in req.negative_tags:
        negative.extend(split_tags(tag))

    db = TagSearchDatabase(dataset_dir)
    db.sync_current_images(current_dataset_image_items(dataset_name))
    rows = db.search(positive, negative, split=req.split, limit=max(1, min(int(req.limit), 2000)))
    filenames = [row["filename"] for row in rows]
    tags_by_name = db.tags_for_filenames(filenames, limit_per_image=24)

    results = []
    for row in rows:
        path = Path(row["filepath"])
        try:
            split = row["split"] or path.parent.parent.name
        except Exception:
            split = row["split"]
        results.append({
            "id": row["id"],
            "filename": row["filename"],
            "split": split,
            "image_path": f"/datasets/{dataset_name}/{split}/images/{row['filename']}",
            "match_count": row["match_count"],
            "avg_confidence": row["avg_confidence"],
            "tags": tags_by_name.get(row["filename"], []),
        })

    return {"status": "ok", "results": results, "total_count": len(results)}


@app.get("/api/dataset/{dataset_name}/tag_suggestions")
async def api_tag_suggestions(dataset_name: str, q: str = "", limit: int = 30):
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    db = TagSearchDatabase(dataset_dir)
    return {"items": db.suggestions(q, limit=max(1, min(int(limit), 100)))}


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
        clear_auto_labeled_unreviewed(dataset_name, req.image_filename)
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
        coco_file = DATASETS_DIR / dataset_name / split / MULTIPOLYGON_LABEL_DIR / (os.path.splitext(filename)[0] + ".json")
        
        if img_file.exists():
            img_file.unlink()

        if lbl_file.exists():
            lbl_file.unlink()
        if coco_file.exists():
            coco_file.unlink()

        negatives = load_negative_samples(dataset_name)
        if filename in negatives:
            negatives.discard(filename)
            save_negative_samples(dataset_name, negatives)
        clear_auto_labeled_unreviewed(dataset_name, filename)

        return {"status": "ok"}
    except Exception as e:
        print(f"Delete image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/{dataset_name}/next_unlabeled")
async def api_next_unlabeled(dataset_name: str):
    negatives = load_negative_samples(dataset_name)
    auto_unreviewed = load_auto_labeled_unreviewed(dataset_name)
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
                if img_file.name in auto_unreviewed or not label_file.exists() or label_file.stat().st_size == 0:
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
