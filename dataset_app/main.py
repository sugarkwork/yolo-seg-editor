from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import yaml
import shutil
import random
from pathlib import Path
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
    
    # Let's find images (looking into train/images for now, or val/images, test/images)
    images = []
    for split in ["train", "valid", "test", "val"]:
        split_dir = DATASETS_DIR / dataset_name / split / "images"
        if split_dir.exists():
            for img_file in split_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
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
                                         
                     images.append({
                         "name": img_file.name,
                         "path": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
                         "label_path": f"/datasets/{dataset_name}/{split}/labels/{label_file.name}",
                         "has_label": label_file.exists(),
                         "split": split,
                         "classes_present": list(classes_present)
                     })
    
    return templates.TemplateResponse(
        request=request, name="dataset.html", context={
            "dataset_name": dataset_name,
            "classes": classes,
            "images": images
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
from typing import List

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
    for split in ["train", "valid", "test", "val"]:
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

@app.post("/api/auto_segment")
async def api_auto_segment(req: AutoSegmentRequest):
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
    except ImportError:
        raise HTTPException(status_code=500, detail="ultralytics library is not installed.")

    model = YOLO(str(model_path))
    
    # Run prediction
    results = model.predict(source=str(physical_img_path), save=False, conf=0.25)
    
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

class AutoSplitRequest(BaseModel):
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1

@app.post("/api/dataset/{dataset_name}/auto_split")
async def api_auto_split(dataset_name: str, req: AutoSplitRequest):
    # validate ratios
    total = req.train_ratio + req.valid_ratio + req.test_ratio
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")
        
    all_images = []
    # Collect all existing images and labels
    for split in ["train", "valid", "test", "val"]:
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
    if req.target_split not in ["train", "valid", "test", "val"]:
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
    for split in ["train", "valid", "test"]:
        (ds_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (ds_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    yaml_path = ds_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": [], "nc": 0}, f, sort_keys=False)
        
    return {"status": "ok", "dataset_name": req.dataset_name.strip()}

from fastapi import UploadFile, File

@app.post("/api/dataset/{dataset_name}/upload_images")
async def api_upload_images(dataset_name: str, files: List[UploadFile] = File(...)):
    target_dir = DATASETS_DIR / dataset_name / "train" / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename:
            file_path = target_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file.filename)
            
    return {"status": "ok", "uploaded": len(saved_files)}

@app.get("/api/dataset/{dataset_name}/next_unlabeled")
async def api_next_unlabeled(dataset_name: str):
    # Scan splits in order for an image without a label file
    for split in ["train", "valid", "test", "val"]:
        images_dir = DATASETS_DIR / dataset_name / split / "images"
        labels_dir = DATASETS_DIR / dataset_name / split / "labels"
        if not images_dir.exists():
            continue
            
        for img_file in images_dir.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                label_file = labels_dir / (img_file.stem + ".txt")
                if not label_file.exists() or label_file.stat().st_size == 0:
                    return {
                        "status": "ok",
                        "next_image": f"/datasets/{dataset_name}/{split}/images/{img_file.name}",
                        "next_label": f"/datasets/{dataset_name}/{split}/labels/{img_file.stem}.txt"
                    }
                    
    return {"status": "none"}

