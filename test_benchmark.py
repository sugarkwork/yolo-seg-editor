"""Standalone benchmark test that mirrors `/api/dataset/{name}/benchmark`.

Replicates the same composite scoring used by the FastAPI endpoint:

    Score = w_mask  * overall_mask_iou
          + w_class * class_detection_recall
          + w_crit  * critical_class_iou       (renormalized when N/A)

For the censoring use case the default weights heavily favor "did we even
detect the right class" over "is the mask shape perfect":

    mask = 0.30   class = 0.40   critical = 0.30

Example:
    python test_benchmark.py --dataset sensitive6_illust \\
        --model v17.pt --split test --device cuda

    # Compare CPU vs CUDA back-to-back on the same images:
    python test_benchmark.py --dataset sensitive6_illust \\
        --model v17.pt --split test --device both
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TRAINING_SPLITS = ["train", "valid", "test", "val"]
H, W = 640, 640

DEFAULT_CRITICAL = ["penis", "pussy"]
DEFAULT_WEIGHTS = {"mask": 0.30, "class": 0.40, "critical": 0.30}


def collect_images(dataset_name: str, split: str):
    splits = TRAINING_SPLITS if split == "all" else [split]
    image_paths, label_paths = [], []
    for s in splits:
        images_dir = DATASETS_DIR / dataset_name / s / "images"
        labels_dir = DATASETS_DIR / dataset_name / s / "labels"
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.*")):
            if img_file.suffix.lower() in IMAGE_EXTS:
                image_paths.append(img_file)
                label_paths.append(labels_dir / f"{img_file.stem}.txt")
    return image_paths, label_paths


def load_dataset_classes(dataset_name: str):
    yaml_path = DATASETS_DIR / dataset_name / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", []) or []


def parse_ground_truth(label_paths, critical_id_set):
    """Return (gt_overall, gt_critical_per_image, gt_class_sets)."""
    gt_overall, gt_critical, gt_class_sets = [], [], []
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
                            [[int(coords[i] * W), int(coords[i + 1] * H)]
                             for i in range(0, len(coords), 2)],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(overall, [pts], 1)
                        cls_set.add(cls_id)
                        if cls_id in critical_id_set:
                            if cls_id not in crit:
                                crit[cls_id] = np.zeros((H, W), dtype=np.uint8)
                            cv2.fillPoly(crit[cls_id], [pts], 1)
        gt_overall.append(overall)
        gt_critical.append(crit)
        gt_class_sets.append(cls_set)
    return gt_overall, gt_critical, gt_class_sets


def composite(mean_mask, mean_class, mean_crit, weights):
    total, used = 0.0, 0.0
    if mean_mask is not None:
        total += weights["mask"] * mean_mask
        used += weights["mask"]
    if mean_class is not None:
        total += weights["class"] * mean_class
        used += weights["class"]
    if mean_crit is not None:
        total += weights["critical"] * mean_crit
        used += weights["critical"]
    return (total / used * 100.0) if used > 0 else 0.0


def run_one(model_path: Path, image_paths, gt_overall, gt_critical, gt_class_sets,
            name_to_id, critical_ids, device, half: bool):
    paths_str = [str(p) for p in image_paths]
    model = YOLO(str(model_path))

    # Map model class IDs to dataset class IDs by case-insensitive name match.
    model_to_dataset = {}
    for mid, mname in (model.names or {}).items():
        key = str(mname).lower()
        if key in name_to_id:
            model_to_dataset[int(mid)] = name_to_id[key]

    critical_id_set = set(critical_ids)
    mask_ious, class_recalls, crit_ious = [], [], []

    t0 = time.time()
    stream = model.predict(
        source=paths_str,
        save=False,
        conf=0.25,
        verbose=False,
        retina_masks=True,
        device=device,
        half=half,
        stream=True,
    )
    for idx, res in enumerate(stream):
        pred_overall = np.zeros((H, W), dtype=np.uint8)
        pred_critical = {}
        pred_cls_set = set()
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
                    pred_cls_set.add(dcls)
                    if dcls in critical_id_set:
                        if dcls not in pred_critical:
                            pred_critical[dcls] = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(pred_critical[dcls], [pts], 1)

        gt_o = gt_overall[idx]
        inter = int(np.logical_and(gt_o, pred_overall).sum())
        union = int(np.logical_or(gt_o, pred_overall).sum())
        mask_ious.append(1.0 if union == 0 else inter / union)

        gt_cls = gt_class_sets[idx]
        if gt_cls:
            matched = gt_cls & pred_cls_set
            class_recalls.append(len(matched) / len(gt_cls))

        per_img_crit = []
        for cid in critical_ids:
            gt_m = gt_critical[idx].get(cid)
            pr_m = pred_critical.get(cid)
            if gt_m is None and pr_m is None:
                continue
            if gt_m is None or pr_m is None:
                per_img_crit.append(0.0)
                continue
            ci = int(np.logical_and(gt_m, pr_m).sum())
            cu = int(np.logical_or(gt_m, pr_m).sum())
            per_img_crit.append(1.0 if cu == 0 else ci / cu)
        if per_img_crit:
            crit_ious.append(float(np.mean(per_img_crit)))

    elapsed = time.time() - t0
    del model
    if isinstance(device, int):
        torch.cuda.empty_cache()
    return mask_ious, class_recalls, crit_ious, elapsed


def summarize(label, mask_ious, class_recalls, crit_ious, elapsed, n, weights):
    if not mask_ious:
        return f"  [{label}] no results"
    mean_mask = float(np.mean(mask_ious))
    mean_class = float(np.mean(class_recalls)) if class_recalls else None
    mean_crit = float(np.mean(crit_ious)) if crit_ious else None
    score = composite(mean_mask, mean_class, mean_crit, weights)
    fps = n / elapsed if elapsed > 0 else float("inf")
    crit_str = f"{mean_crit * 100:5.1f}%" if mean_crit is not None else "  N/A"
    cls_str = f"{mean_class * 100:5.1f}%" if mean_class is not None else "  N/A"
    return (
        f"  [{label}] score={score:5.1f}  "
        f"mask={mean_mask * 100:5.1f}%  class={cls_str}  crit={crit_str}  "
        f"elapsed={elapsed:.2f}s ({fps:.2f} img/s)"
    )


def resolve_devices(choice: str):
    if choice == "cpu":
        return [("cpu", "cpu", False)]
    if choice == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA requested but torch.cuda.is_available() is False",
                  file=sys.stderr)
            sys.exit(2)
        return [("cuda", 0, True)]
    if choice == "both":
        runs = [("cpu", "cpu", False)]
        if torch.cuda.is_available():
            runs.append(("cuda", 0, True))
        else:
            print("WARN: CUDA unavailable, running CPU only", file=sys.stderr)
        return runs
    raise ValueError(f"unknown device choice: {choice}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset name under datasets/")
    parser.add_argument("--model", action="append", required=True,
                        help="model file under models/ (repeat for multiple)")
    parser.add_argument("--split", default="test",
                        choices=["train", "valid", "test", "val", "all"])
    parser.add_argument("--device", default="cuda",
                        choices=["cpu", "cuda", "both"])
    parser.add_argument("--limit", type=int, default=0,
                        help="cap image count for a quick smoke test (0 = no cap)")
    parser.add_argument("--critical", default=",".join(DEFAULT_CRITICAL),
                        help="comma-separated critical class names (default: penis,pussy)")
    parser.add_argument("--w-mask", type=float, default=DEFAULT_WEIGHTS["mask"])
    parser.add_argument("--w-class", type=float, default=DEFAULT_WEIGHTS["class"])
    parser.add_argument("--w-critical", type=float, default=DEFAULT_WEIGHTS["critical"])
    args = parser.parse_args()

    weights = {"mask": args.w_mask, "class": args.w_class, "critical": args.w_critical}

    print(f"PyTorch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}"
          f"  built with CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)}  "
                  f"free={free / 1024 ** 3:.2f}GB  total={total / 1024 ** 3:.2f}GB")
    print()

    dataset_classes = load_dataset_classes(args.dataset)
    name_to_id = {n.lower(): i for i, n in enumerate(dataset_classes)}
    requested_critical = [c.strip() for c in args.critical.split(",") if c.strip()]
    critical_ids = [name_to_id[c.lower()] for c in requested_critical if c.lower() in name_to_id]
    critical_names_used = [dataset_classes[i] for i in critical_ids]
    missing = [c for c in requested_critical if c.lower() not in name_to_id]
    print(f"Dataset classes ({len(dataset_classes)}): {dataset_classes}")
    print(f"Critical classes used: {critical_names_used}"
          + (f"  (skipped: {missing})" if missing else ""))
    print(f"Weights: mask={weights['mask']}  class={weights['class']}  critical={weights['critical']}")

    image_paths, label_paths = collect_images(args.dataset, args.split)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        label_paths = label_paths[:args.limit]
    if not image_paths:
        print(f"ERROR: no images for dataset={args.dataset} split={args.split}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Dataset: {args.dataset}  split: {args.split}  images: {len(image_paths)}\n")

    print("Parsing ground truth...")
    gt_overall, gt_critical, gt_class_sets = parse_ground_truth(label_paths, set(critical_ids))

    runs = resolve_devices(args.device)
    summary_rows = []
    for model_name in args.model:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"  SKIP {model_name}: not found in {MODELS_DIR}", file=sys.stderr)
            continue
        print(f"\nModel: {model_name}")
        per_device = {}
        for label, device, half in runs:
            print(f"  [{label}] running...", flush=True)
            mask_ious, class_recalls, crit_ious, elapsed = run_one(
                model_path, image_paths,
                gt_overall, gt_critical, gt_class_sets,
                name_to_id, critical_ids, device, half,
            )
            print(summarize(label, mask_ious, class_recalls, crit_ious,
                            elapsed, len(image_paths), weights))
            per_device[label] = (elapsed, mask_ious, class_recalls, crit_ious)
        summary_rows.append((model_name, per_device))

    if any("cpu" in r[1] and "cuda" in r[1] for r in summary_rows):
        print("\nSpeedup (CUDA vs CPU):")
        for model_name, per in summary_rows:
            if "cpu" in per and "cuda" in per:
                ratio = per["cpu"][0] / per["cuda"][0] if per["cuda"][0] > 0 else float("inf")
                print(f"  {model_name}: {ratio:.2f}x  "
                      f"(cpu {per['cpu'][0]:.2f}s -> cuda {per['cuda'][0]:.2f}s)")


if __name__ == "__main__":
    main()
