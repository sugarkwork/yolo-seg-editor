"""Standalone benchmark test that mirrors `/api/dataset/{name}/benchmark`.

Replicates the same penalty-point scoring used by the FastAPI endpoint:

    Score = 100
          - (1 - critical_gt_coverage) * 50
          - gt_outside_detection_ratio * 20

Example:
    python test_benchmark.py --dataset sensitive6_illust \\
        --model v17.pt --split test --device cuda

    # Compare CPU vs CUDA back-to-back on the same images:
    python test_benchmark.py --dataset sensitive6_illust \\
        --model v17.pt --split test --device both
"""

import argparse
import os
import re
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

DEFAULT_CRITICAL = ["penis", "pussy", "anus"]
DEFAULT_IGNORED = ["nipple"]
DEFAULT_WEIGHTS = {"mask": 0.30, "class": 0.40, "critical": 0.30}
CRITICAL_MISS_MAX_POINTS = 50.0
FALSE_POSITIVE_MAX_POINTS = 20.0
DEFAULT_BATCH_SIZE = 8


def resolve_batch_size(requested=None) -> int:
    raw = requested if requested not in (None, "") else os.environ.get("YOLO_BENCHMARK_BATCH", DEFAULT_BATCH_SIZE)
    try:
        return max(1, min(64, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_BATCH_SIZE


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def coerce_yolo_imgsz(value, default: int = 640):
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
            coerced = coerce_yolo_imgsz(item, default)
            if isinstance(coerced, int):
                sizes.append(coerced)
            elif isinstance(coerced, list):
                sizes.extend(coerced)
        sizes = [s for s in sizes if s >= 32]
        if not sizes:
            return None
        return sizes[0] if len(sizes) == 1 or sizes[0] == sizes[1] else sizes[:2]
    return None


def get_arg_value(args, key: str):
    if isinstance(args, dict):
        return args.get(key)
    return getattr(args, key, None)


def resolve_yolo_model_imgsz(model, default: int = 640):
    sources = [
        getattr(getattr(model, "model", None), "args", None),
        getattr(model, "overrides", None),
        getattr(model, "args", None),
    ]
    ckpt = getattr(model, "ckpt", None)
    if isinstance(ckpt, dict):
        sources.append(ckpt.get("train_args"))

    for args in sources:
        imgsz = coerce_yolo_imgsz(get_arg_value(args, "imgsz"), default)
        if imgsz is not None:
            return imgsz
    return default


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


def parse_ground_truth(label_paths, critical_id_set, ignored_id_set=None):
    """Return (gt_polygons, gt_class_sets) instead of huge rasterized masks to save memory."""
    ignored_id_set = ignored_id_set or set()
    gt_polygons, gt_class_sets = [], []
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
                            [[int(coords[i] * W), int(coords[i + 1] * H)]
                             for i in range(0, len(coords), 2)],
                            dtype=np.int32,
                        )
                        polys.append({"cls_id": cls_id, "pts": pts})
                        cls_set.add(cls_id)
        gt_polygons.append(polys)
        gt_class_sets.append(cls_set)
    return gt_polygons, gt_class_sets


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


def coverage_metrics(gt_mask, pred_mask):
    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0
    gt_area = int(gt_bool.sum())
    pred_area = int(pred_bool.sum())
    inter = int(np.logical_and(gt_bool, pred_bool).sum())
    union = int(np.logical_or(gt_bool, pred_bool).sum())
    coverage = 1.0 if gt_area == 0 and pred_area == 0 else (inter / gt_area if gt_area > 0 else 0.0)
    outside = max(pred_area - inter, 0)
    fp_ratio = outside / pred_area if pred_area > 0 else 0.0
    return {
        "coverage": coverage,
        "raw_iou": 1.0 if union == 0 else inter / union,
        "false_positive_ratio": fp_ratio,
    }


def critical_coverage(gt_mask, pred_mask):
    gt_bool = gt_mask > 0
    gt_area = int(gt_bool.sum())
    if gt_area == 0:
        return None
    if pred_mask is None:
        return 0.0
    pred_bool = pred_mask > 0
    inter = int(np.logical_and(gt_bool, pred_bool).sum())
    return inter / gt_area


def penalty_score(mean_crit_coverage, mean_fp_ratio):
    crit_penalty = 0.0
    if mean_crit_coverage is not None:
        crit_penalty = (1.0 - mean_crit_coverage) * CRITICAL_MISS_MAX_POINTS
    fp_penalty = mean_fp_ratio * FALSE_POSITIVE_MAX_POINTS
    points = min(100.0, crit_penalty + fp_penalty)
    return max(0.0, 100.0 - points), points / 100.0


def run_one(model_path: Path, image_paths, gt_polygons, gt_class_sets,
            name_to_id, critical_ids, ignored_ids, device, half: bool,
            cache_predictions: bool = False, batch_size: int = DEFAULT_BATCH_SIZE):
    paths_str = [str(p) for p in image_paths]
    model = YOLO(str(model_path))
    model_imgsz = resolve_yolo_model_imgsz(model)

    # Map model class IDs to dataset class IDs by case-insensitive name match.
    model_to_dataset = {}
    for mid, mname in (model.names or {}).items():
        key = str(mname).lower()
        if key in name_to_id:
            model_to_dataset[int(mid)] = name_to_id[key]

    critical_id_set = set(critical_ids)
    ignored_id_set = set(ignored_ids)
    mask_coverages, class_recalls, crit_coverages = [], [], []
    predictions = []
    false_positive_ratios = []

    t0 = time.time()
    current_batch_size = resolve_batch_size(batch_size)
    while True:
        mask_coverages, class_recalls, crit_coverages = [], [], []
        predictions = []
        false_positive_ratios = []
        try:
            result_stream = model.predict(
                source=paths_str,
                save=False,
                conf=0.25,
                verbose=False,
                retina_masks=True,
                device=device,
                half=half,
                imgsz=model_imgsz,
                stream=True,
                batch=current_batch_size,
            )
            for idx, res in enumerate(result_stream):
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

                if cache_predictions:
                    predictions.append({
                        "overall": np.packbits(pred_overall > 0),
                        "class_set": pred_class_set,
                        "critical": {cid: np.packbits(mask > 0) for cid, mask in pred_critical.items()}
                    })

                # On-demand Ground Truth rasterization
                gt_o, gt_crit_masks = rasterize_gt(gt_polygons[idx], critical_id_set)

                mask_metric = coverage_metrics(gt_o, pred_overall)
                mask_coverages.append(float(mask_metric["coverage"]))
                false_positive_ratios.append(float(mask_metric["false_positive_ratio"]))

                gt_cls = gt_class_sets[idx]
                if gt_cls:
                    matched = gt_cls & pred_class_set
                    class_recalls.append(len(matched) / len(gt_cls))

                per_img_crit = []
                for cid in critical_ids:
                    gt_m = gt_crit_masks.get(cid)
                    if gt_m is None:
                        continue
                    crit_score = critical_coverage(gt_m, pred_critical.get(cid))
                    if crit_score is not None:
                        per_img_crit.append(float(crit_score))
                if per_img_crit:
                    crit_coverages.append(float(np.mean(per_img_crit)))

                # Explicitly delete the prediction results object to free VRAM immediately
                del res
            break
        except RuntimeError as e:
            if half and current_batch_size > 1 and is_cuda_oom_error(e):
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"  CUDA OOM at batch={current_batch_size}; retrying batch={next_batch_size}",
                    flush=True,
                )
                current_batch_size = next_batch_size
                torch.cuda.empty_cache()
                continue
            raise

    elapsed = time.time() - t0
    del model
    if isinstance(device, int):
        torch.cuda.empty_cache()
    
    mean_crit = float(np.mean(crit_coverages)) if crit_coverages else None
    mean_fp = float(np.mean(false_positive_ratios)) if false_positive_ratios else 0.0
    _, penalty = penalty_score(mean_crit, mean_fp)
    
    return mask_coverages, class_recalls, crit_coverages, elapsed, predictions, penalty, current_batch_size


def summarize(label, mask_ious, class_recalls, crit_ious, elapsed, n, weights, penalty=0.0):
    if not mask_ious:
        return f"  [{label}] no results"
    mean_mask = float(np.mean(mask_ious))
    mean_class = float(np.mean(class_recalls)) if class_recalls else None
    mean_crit = float(np.mean(crit_ious)) if crit_ious else None
    score = max(0.0, 100.0 - (penalty * 100.0))
    fps = n / elapsed if elapsed > 0 else float("inf")
    crit_str = f"{mean_crit * 100:5.1f}%" if mean_crit is not None else "  N/A"
    cls_str = f"{mean_class * 100:5.1f}%" if mean_class is not None else "  N/A"
    pen_str = f" (-{penalty * 100:.0f}pt)" if penalty > 0 else ""
    return (
        f"  [{label}] score={score:5.1f}{pen_str}  "
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
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"YOLO predict batch size (default: {DEFAULT_BATCH_SIZE}, or YOLO_BENCHMARK_BATCH)")
    parser.add_argument("--critical", default=",".join(DEFAULT_CRITICAL),
                        help=f"comma-separated critical class names (default: {','.join(DEFAULT_CRITICAL)})")
    parser.add_argument("--ignore-class", "--ignore-classes", default=",".join(DEFAULT_IGNORED),
                        help="comma-separated class names ignored in benchmark scoring (default: nipple)")
    parser.add_argument("--w-mask", type=float, default=DEFAULT_WEIGHTS["mask"])
    parser.add_argument("--w-class", type=float, default=DEFAULT_WEIGHTS["class"])
    parser.add_argument("--w-critical", type=float, default=DEFAULT_WEIGHTS["critical"])
    args = parser.parse_args()

    weights = {"mask": args.w_mask, "class": args.w_class, "critical": args.w_critical}
    batch_size = resolve_batch_size(args.batch_size)

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
    requested_ignored = [c.strip() for c in args.ignore_class.split(",") if c.strip()]
    ignored_ids = {name_to_id[c.lower()] for c in requested_ignored if c.lower() in name_to_id}
    ignored_names_used = [dataset_classes[i] for i in sorted(ignored_ids)]
    missing_ignored = [c for c in requested_ignored if c.lower() not in name_to_id]
    print(f"Dataset classes ({len(dataset_classes)}): {dataset_classes}")
    print(f"Critical classes used: {critical_names_used}"
          + (f"  (skipped: {missing})" if missing else ""))
    print(f"Ignored classes used: {ignored_names_used}"
          + (f"  (skipped: {missing_ignored})" if missing_ignored else ""))
    print(
        f"Scoring: 100 - critical miss up to {CRITICAL_MISS_MAX_POINTS:.0f}pt"
        f" - GT-outside detection up to {FALSE_POSITIVE_MAX_POINTS:.0f}pt"
    )

    image_paths, label_paths = collect_images(args.dataset, args.split)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        label_paths = label_paths[:args.limit]
    if not image_paths:
        print(f"ERROR: no images for dataset={args.dataset} split={args.split}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Dataset: {args.dataset}  split: {args.split}  images: {len(image_paths)}  batch: {batch_size}\n")

    print("Parsing ground truth...")
    gt_polygons, gt_class_sets = parse_ground_truth(label_paths, set(critical_ids), ignored_ids)

    runs = resolve_devices(args.device)
    summary_rows = []
    model_caches = {}
    for model_name in args.model:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"  SKIP {model_name}: not found in {MODELS_DIR}", file=sys.stderr)
            continue
        print(f"\nModel: {model_name}")
        per_device = {}
        for label, device, half in runs:
            print(f"  [{label}] running...", flush=True)
            cache_preds = len(args.model) >= 2
            mask_ious, class_recalls, crit_ious, elapsed, predictions, penalty, used_batch_size = run_one(
                model_path, image_paths,
                gt_polygons, gt_class_sets,
                name_to_id, critical_ids, ignored_ids, device, half,
                cache_predictions=cache_preds,
                batch_size=batch_size,
            )
            print(summarize(label, mask_ious, class_recalls, crit_ious,
                            elapsed, len(image_paths), weights, penalty)
                  + f"  batch={used_batch_size}")
            per_device[label] = (elapsed, mask_ious, class_recalls, crit_ious, penalty)
            if label not in model_caches:
                model_caches[label] = {}
            model_caches[label][model_name] = predictions
        summary_rows.append((model_name, per_device))

    if any("cpu" in r[1] and "cuda" in r[1] for r in summary_rows):
        print("\nSpeedup (CUDA vs CPU):")
        for model_name, per in summary_rows:
            if "cpu" in per and "cuda" in per:
                ratio = per["cpu"][0] / per["cuda"][0] if per["cuda"][0] > 0 else float("inf")
                print(f"  {model_name}: {ratio:.2f}x  "
                      f"(cpu {per['cpu'][0]:.2f}s -> cuda {per['cuda'][0]:.2f}s)")

    # Evaluate 2-model combinations
    import itertools
    for label, device, half in runs:
        caches = model_caches.get(label, {})
        valid_models = [m for m in args.model if m in caches and len(caches[m]) == len(image_paths)]
        if len(valid_models) >= 2:
            print(f"\nBest 2-Model Combinations ({label}):")
            combo_results = []
            for modelA_name, modelB_name in itertools.combinations(valid_models, 2):
                cache_A = caches[modelA_name]
                cache_B = caches[modelB_name]

                combo_mask_coverages = []
                combo_class_recalls = []
                combo_critical_coverages = []
                combo_false_positive_ratios = []

                for idx in range(len(image_paths)):
                    pred_A = cache_A[idx]
                    pred_B = cache_B[idx]

                    # Unpack masks
                    mask_A_overall = np.unpackbits(pred_A["overall"])[:H*W].reshape((H, W)) > 0
                    mask_B_overall = np.unpackbits(pred_B["overall"])[:H*W].reshape((H, W)) > 0

                    # 1. Combined overall coverage and GT-outside detection ratio
                    comb_overall = np.logical_or(mask_A_overall, mask_B_overall)
                    
                    # On-demand GT rasterization
                    gt_o, gt_crit_masks = rasterize_gt(gt_polygons[idx], set(critical_ids))
                    mask_metric = coverage_metrics(gt_o, comb_overall.astype(np.uint8))
                    combo_mask_coverages.append(float(mask_metric["coverage"]))
                    combo_false_positive_ratios.append(float(mask_metric["false_positive_ratio"]))

                    # 2. Combined class recall
                    comb_cls_set = pred_A["class_set"] | pred_B["class_set"]
                    gt_cls = gt_class_sets[idx]
                    if gt_cls:
                        matched = gt_cls & comb_cls_set
                        combo_class_recalls.append(len(matched) / len(gt_cls))

                    # 3. Combined critical coverage
                    per_img_critical = []
                    for cid in critical_ids:
                        gt_m = gt_crit_masks.get(cid)
                        if gt_m is None:
                            continue

                        packed_mask_A = pred_A["critical"].get(cid)
                        packed_mask_B = pred_B["critical"].get(cid)

                        mask_A = np.unpackbits(packed_mask_A)[:H*W].reshape((H, W)) > 0 if packed_mask_A is not None else None
                        mask_B = np.unpackbits(packed_mask_B)[:H*W].reshape((H, W)) > 0 if packed_mask_B is not None else None

                        if mask_A is None and mask_B is None:
                            pr_m = None
                        elif mask_A is not None and mask_B is None:
                            pr_m = mask_A
                        elif mask_A is None and mask_B is not None:
                            pr_m = mask_B
                        else:
                            pr_m = np.logical_or(mask_A, mask_B)

                        crit_score = critical_coverage(gt_m, pr_m)
                        if crit_score is not None:
                            per_img_critical.append(float(crit_score))

                    if per_img_critical:
                        combo_critical_coverages.append(float(np.mean(per_img_critical)))

                mean_mask = float(np.mean(combo_mask_coverages)) if combo_mask_coverages else None
                mean_class = float(np.mean(combo_class_recalls)) if combo_class_recalls else None
                mean_crit = float(np.mean(combo_critical_coverages)) if combo_critical_coverages else None
                mean_fp = float(np.mean(combo_false_positive_ratios)) if combo_false_positive_ratios else 0.0
                score, penalty = penalty_score(mean_crit, mean_fp)

                combo_results.append((modelA_name, modelB_name, score, mean_mask, mean_class, mean_crit, penalty))

            # Sort by score descending and display top combinations
            combo_results.sort(key=lambda x: x[2], reverse=True)
            for rank, (mA, mB, score, m_mask, m_cls, m_crit, pen) in enumerate(combo_results[:5], 1):
                crit_str = f"{m_crit * 100:5.1f}%" if m_crit is not None else "  N/A"
                cls_str = f"{m_cls * 100:5.1f}%" if m_cls is not None else "  N/A"
                pen_str = f" (-{pen * 100:.0f}pt)" if pen > 0 else ""
                print(f"  #{rank} {mA} + {mB} : score={score:5.1f}{pen_str}  mask={m_mask * 100:5.1f}%  class={cls_str}  crit={crit_str}")


if __name__ == "__main__":
    main()
