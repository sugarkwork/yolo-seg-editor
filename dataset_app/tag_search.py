from __future__ import annotations

import csv
import os
import sqlite3
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_TAGGER_MODEL_NAME = "wd-eva02-large-tagger-v3-fp16.onnx"
IMAGE_TAGGER_CSV_NAME = "wd-eva02-large-tagger-v3.csv"
HF_BASE = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main"
HF_CSV_URL = f"{HF_BASE}/selected_tags.csv?download=true"

DEFAULT_MODEL_CANDIDATES = [
    Path(r"F:\ai\ComfyUI\custom_nodes\comfyui-onnxtagger\models") / IMAGE_TAGGER_MODEL_NAME,
    Path(r"F:\ai\ComfyUI\models\wd14_tagger") / IMAGE_TAGGER_MODEL_NAME,
]
DEFAULT_CSV_CANDIDATES = [
    Path(r"F:\ai\ComfyUI\models\wd14_tagger") / IMAGE_TAGGER_CSV_NAME,
    Path(r"F:\ai\ComfyUI\custom_nodes\ComfyUI-WD14-Tagger\models") / IMAGE_TAGGER_CSV_NAME,
    Path(r"F:\ai\ComfyUI\custom_nodes\comfyui-onnxtagger\models") / IMAGE_TAGGER_CSV_NAME,
]

_DLL_DIRECTORY_HANDLES = []
_TORCH_CUDA_DLLS_LOADED = False


def _ensure_torch_cuda_dlls_loaded() -> None:
    """Let PyTorch load its bundled CUDA DLLs before ONNX Runtime adds search paths."""
    global _TORCH_CUDA_DLLS_LOADED
    if _TORCH_CUDA_DLLS_LOADED:
        return
    _TORCH_CUDA_DLLS_LOADED = True
    if os.environ.get("TAGGER_SKIP_TORCH_PRELOAD", "").strip() == "1":
        return
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[tagging] torch CUDA DLL preload skipped: {e}", flush=True)


def _preload_cuda_libs() -> None:
    """Load CUDA/cuDNN DLLs from pip-installed nvidia wheels when available."""
    _ensure_torch_cuda_dlls_loaded()
    try:
        import site
        bases = []
        for base in [*site.getsitepackages(), site.getusersitepackages()]:
            if base and base not in bases:
                bases.append(base)
    except Exception:
        return

    if sys.platform == "win32":
        dll_dirs = []
        for base in bases:
            for sub in (
                "nvidia/cuda_runtime/bin",
                "nvidia/cuda_nvrtc/bin",
                "nvidia/cublas/bin",
                "nvidia/cufft/bin",
                "nvidia/nvjitlink/bin",
                "nvidia/cudnn/bin",
            ):
                path = os.path.join(base, sub)
                if not os.path.isdir(path):
                    continue
                dll_dirs.append(path)
                try:
                    handle = os.add_dll_directory(path)
                    _DLL_DIRECTORY_HANDLES.append(handle)
                except (OSError, AttributeError):
                    pass
        if dll_dirs:
            current_path = os.environ.get("PATH", "")
            existing = {p.lower() for p in current_path.split(os.pathsep) if p}
            prepend = [p for p in dll_dirs if p.lower() not in existing]
            if prepend:
                os.environ["PATH"] = os.pathsep.join(prepend + [current_path])
        try:
            import ctypes
            for dll_name in (
                "cudart64_12.dll",
                "nvrtc64_120_0.dll",
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "nvJitLink_120_0.dll",
                "cufft64_11.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
            ):
                for directory in dll_dirs:
                    dll_path = os.path.join(directory, dll_name)
                    if os.path.exists(dll_path):
                        try:
                            ctypes.WinDLL(dll_path)
                        except OSError:
                            pass
                        break
        except Exception:
            pass
        return

    if sys.platform.startswith("linux"):
        try:
            import ctypes
            search_dirs = []
            for base in bases:
                for sub in ("nvidia/cuda_runtime/lib", "nvidia/cublas/lib", "nvidia/cudnn/lib"):
                    path = os.path.join(base, sub)
                    if os.path.isdir(path):
                        search_dirs.append(path)
            priority = ["libcudart", "libnvrtc", "libcublasLt", "libcublas", "libcudnn"]
            loaded = set()
            for prefix in priority:
                for directory in search_dirs:
                    for filename in sorted(os.listdir(directory)):
                        if filename in loaded or not filename.startswith(prefix) or ".so" not in filename:
                            continue
                        try:
                            ctypes.CDLL(os.path.join(directory, filename), mode=ctypes.RTLD_GLOBAL)
                            loaded.add(filename)
                        except OSError:
                            pass
        except Exception:
            pass

GROUPS = {
    "girls": {
        "1girl": (1, 1),
        "2girls": (2, 2),
        "3girls": (3, 3),
        "4girls": (4, 4),
        "5girls": (5, 5),
        "6girls": (6, 6),
        "multiple_girls": (2, inf),
    },
    "boys": {
        "1boy": (1, 1),
        "2boys": (2, 2),
        "3boys": (3, 3),
        "4boys": (4, 4),
        "5boys": (5, 5),
        "6boys": (6, 6),
        "multiple_boys": (2, inf),
    },
    "solo": {
        "solo": (1, 1),
    },
}


def _overlap(a, b) -> bool:
    a1, a2 = a
    b1, b2 = b
    return not (a2 < b1 or b2 < a1)


def build_tag_query(positive_tags: Iterable[str], negative_tags: Iterable[str] = ()):
    pos = {normalize_tag(t) for t in positive_tags if normalize_tag(t)}
    neg = {normalize_tag(t) for t in negative_tags if normalize_tag(t)}

    for tag_map in GROUPS.values():
        chosen = {t for t in tag_map if t in pos}
        if not chosen:
            continue
        allowed = [tag_map[t] for t in chosen]
        for tag, rng in tag_map.items():
            if tag not in pos and not any(_overlap(rng, ar) for ar in allowed):
                neg.add(tag)

    return sorted(pos), sorted(t for t in neg if t not in pos)


def normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def split_tags(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = value
    return [normalize_tag(t) for t in raw if normalize_tag(t)]


def resolve_tagger_paths(base_dir: Path) -> tuple[Path, Path]:
    configured_model = os.environ.get("TAGGER_ONNX_PATH", "").strip()
    configured_csv = os.environ.get("TAGGER_CSV_PATH", "").strip()

    model_candidates = []
    csv_candidates = []
    if configured_model:
        model_candidates.append(Path(configured_model))
    if configured_csv:
        csv_candidates.append(Path(configured_csv))

    local_dir = base_dir / "tagger_models"
    model_candidates.extend([
        local_dir / IMAGE_TAGGER_MODEL_NAME,
        *DEFAULT_MODEL_CANDIDATES,
    ])
    csv_candidates.extend([
        local_dir / IMAGE_TAGGER_CSV_NAME,
        *DEFAULT_CSV_CANDIDATES,
    ])

    model_path = next((p for p in model_candidates if p.exists()), None)
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is None:
        local_dir.mkdir(parents=True, exist_ok=True)
        csv_path = local_dir / IMAGE_TAGGER_CSV_NAME
        _download(HF_CSV_URL, csv_path)

    if model_path is None:
        raise FileNotFoundError(
            "ONNX tagger model not found. Set TAGGER_ONNX_PATH or place "
            f"{IMAGE_TAGGER_MODEL_NAME} under tagger_models/."
        )

    return model_path, csv_path


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-app-tagger/1.0"})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)


@dataclass
class TaggerOutput:
    tags: list[tuple[str, float]]
    inference_ms: float


class OnnxImageTagger:
    def __init__(self, model_path: Path, csv_path: Path, providers: Iterable[str] | None = None):
        _preload_cuda_libs()
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for image tagging") from e

        available = set(ort.get_available_providers())
        if providers is None:
            cuda_device_id = int(os.environ.get("TAGGER_CUDA_DEVICE_ID", "0"))
            provider_candidates = [
                ("CUDAExecutionProvider", {"device_id": cuda_device_id}),
                "CPUExecutionProvider",
            ]
        else:
            provider_candidates = list(providers)

        filtered_providers = []
        for provider in provider_candidates:
            name = provider[0] if isinstance(provider, tuple) else provider
            if name in available:
                filtered_providers.append(provider)
        filtered_providers = filtered_providers or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=filtered_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = int(self.session.get_inputs()[0].shape[1])
        self.tags, self.general_index, self.character_index = self._load_tags(csv_path)
        print(f"[tagging] ONNX Runtime providers: {self.session.get_providers()}", flush=True)

    @staticmethod
    def _load_tags(csv_path: Path) -> tuple[list[str], int, int]:
        tags = []
        general_index = None
        character_index = None
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])
        return tags, general_index or 0, character_index or len(tags)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = self.input_size
        ratio = float(size) / max(image.size)
        new_size = tuple(max(1, int(x * ratio)) for x in image.size)
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (size, size), (255, 255, 255))
        square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
        arr = np.array(square).astype(np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, 0)

    def tag_image(
        self,
        image_path: Path,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
    ) -> TaggerOutput:
        with Image.open(image_path) as image:
            x = self._preprocess(image)
        t0 = time.perf_counter()
        probs = self.session.run([self.output_name], {self.input_name: x})[0][0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        general = [
            (self.tags[i], float(probs[i]))
            for i in range(self.general_index, self.character_index)
            if probs[i] > threshold
        ]
        character = [
            (self.tags[i], float(probs[i]))
            for i in range(self.character_index, len(self.tags))
            if probs[i] > character_threshold
        ]
        return TaggerOutput(tags=character + general, inference_ms=inference_ms)


_GLOBAL_TAGGER = None
_GLOBAL_TAGGER_KEY = None


def get_tagger(base_dir: Path) -> OnnxImageTagger:
    global _GLOBAL_TAGGER, _GLOBAL_TAGGER_KEY
    model_path, csv_path = resolve_tagger_paths(base_dir)
    key = (str(model_path), str(csv_path))
    if _GLOBAL_TAGGER is None or _GLOBAL_TAGGER_KEY != key:
        _GLOBAL_TAGGER = OnnxImageTagger(model_path, csv_path)
        _GLOBAL_TAGGER_KEY = key
    return _GLOBAL_TAGGER


class TagSearchDatabase:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.db_path = dataset_dir / "tag_search.db"
        self.init_database()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    split TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    size INTEGER NOT NULL DEFAULT 0,
                    mtime REAL NOT NULL DEFAULT 0,
                    tagged_at REAL,
                    inference_ms REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT NOT NULL UNIQUE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (image_id, tag_id),
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_name ON tags(tag_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_tag_image ON image_tags(tag_id, image_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id)")

    def sync_current_images(self, image_items: list[tuple[str, Path]]) -> None:
        current = {path.name for split, path in image_items}
        with self.connect() as conn:
            for split, path in image_items:
                stat = path.stat()
                conn.execute(
                    """
                    INSERT INTO images(filename, split, filepath, size, mtime)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(filename) DO UPDATE SET
                        split=excluded.split,
                        filepath=excluded.filepath,
                        size=excluded.size,
                        mtime=excluded.mtime,
                        tagged_at=CASE
                            WHEN images.size != excluded.size OR ABS(images.mtime - excluded.mtime) > 0.001
                            THEN NULL
                            ELSE images.tagged_at
                        END
                    """,
                    (path.name, split, str(path), int(stat.st_size), float(stat.st_mtime)),
                )
            if current:
                placeholders = ",".join("?" for _ in current)
                stale_ids = [
                    row["id"]
                    for row in conn.execute(
                        f"SELECT id FROM images WHERE filename NOT IN ({placeholders})",
                        tuple(current),
                    )
                ]
            else:
                stale_ids = [row["id"] for row in conn.execute("SELECT id FROM images")]
            for image_id in stale_ids:
                conn.execute("DELETE FROM image_tags WHERE image_id = ?", (image_id,))
                conn.execute("DELETE FROM images WHERE id = ?", (image_id,))

    def missing_tag_images(self, image_items: list[tuple[str, Path]], limit: int | None = None) -> list[tuple[str, Path]]:
        with self.connect() as conn:
            missing = []
            for split, path in image_items:
                row = conn.execute(
                    "SELECT id, size, mtime, tagged_at FROM images WHERE filename = ?",
                    (path.name,),
                ).fetchone()
                if row is None or row["tagged_at"] is None:
                    missing.append((split, path))
                    continue
                stat = path.stat()
                if int(row["size"]) != int(stat.st_size) or abs(float(row["mtime"]) - float(stat.st_mtime)) > 0.001:
                    missing.append((split, path))
            return missing[:limit] if limit else missing

    def upsert_tags(self, split: str, image_path: Path, tags: list[tuple[str, float]], inference_ms: float) -> None:
        stat = image_path.stat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO images(filename, split, filepath, size, mtime, tagged_at, inference_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    split=excluded.split,
                    filepath=excluded.filepath,
                    size=excluded.size,
                    mtime=excluded.mtime,
                    tagged_at=excluded.tagged_at,
                    inference_ms=excluded.inference_ms
                """,
                (image_path.name, split, str(image_path), int(stat.st_size), float(stat.st_mtime), time.time(), inference_ms),
            )
            image_id = conn.execute("SELECT id FROM images WHERE filename = ?", (image_path.name,)).fetchone()["id"]
            conn.execute("DELETE FROM image_tags WHERE image_id = ?", (image_id,))
            for tag, confidence in tags:
                tag_name = normalize_tag(tag)
                if not tag_name:
                    continue
                conn.execute("INSERT OR IGNORE INTO tags(tag_name) VALUES (?)", (tag_name,))
                tag_id = conn.execute("SELECT id FROM tags WHERE tag_name = ?", (tag_name,)).fetchone()["id"]
                conn.execute(
                    "INSERT OR REPLACE INTO image_tags(image_id, tag_id, confidence) VALUES (?, ?, ?)",
                    (image_id, tag_id, float(confidence)),
                )

    def tags_for_filenames(self, filenames: Iterable[str], limit_per_image: int = 24) -> dict[str, list[dict]]:
        names = list(filenames)
        if not names:
            return {}
        placeholders = ",".join("?" for _ in names)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT i.filename, t.tag_name, it.confidence
                FROM images i
                JOIN image_tags it ON it.image_id = i.id
                JOIN tags t ON t.id = it.tag_id
                WHERE i.filename IN ({placeholders})
                ORDER BY i.filename, it.confidence DESC
                """,
                tuple(names),
            ).fetchall()
        out: dict[str, list[dict]] = {name: [] for name in names}
        for row in rows:
            bucket = out.setdefault(row["filename"], [])
            if len(bucket) < limit_per_image:
                bucket.append({"tag": row["tag_name"], "confidence": float(row["confidence"])})
        return out

    def search(self, positive_tags: list[str], negative_tags: list[str], split: str = "all", limit: int = 500) -> list[dict]:
        positive_tags, negative_tags = build_tag_query(positive_tags, negative_tags)
        if not positive_tags:
            return []

        with self.connect() as conn:
            params: list = []
            pos_placeholders = ",".join("?" for _ in positive_tags)
            params.extend(positive_tags)
            split_clause = ""
            if split != "all":
                split_clause = "AND i.split = ?"
                params.append(split)

            neg_clause = ""
            if negative_tags:
                neg_placeholders = ",".join("?" for _ in negative_tags)
                neg_clause = f"""
                    AND NOT EXISTS (
                        SELECT 1
                        FROM image_tags it2
                        JOIN tags t2 ON t2.id = it2.tag_id
                        WHERE it2.image_id = i.id
                        AND t2.tag_name IN ({neg_placeholders})
                    )
                """
                params.extend(negative_tags)

            params.append(len(positive_tags))
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT i.id, i.filename, i.filepath, i.split,
                       COUNT(DISTINCT t.tag_name) AS match_count,
                       AVG(it.confidence) AS avg_confidence
                FROM images i
                JOIN image_tags it ON it.image_id = i.id
                JOIN tags t ON t.id = it.tag_id
                WHERE t.tag_name IN ({pos_placeholders})
                {split_clause}
                {neg_clause}
                GROUP BY i.id
                HAVING match_count = ?
                ORDER BY avg_confidence DESC, i.filename ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def suggestions(self, query: str, limit: int = 30) -> list[dict]:
        q = f"%{normalize_tag(query)}%"
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.tag_name, COUNT(it.image_id) AS count
                FROM tags t
                LEFT JOIN image_tags it ON it.tag_id = t.id
                WHERE t.tag_name LIKE ?
                GROUP BY t.id
                ORDER BY count DESC, t.tag_name ASC
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
        return [{"tag": row["tag_name"], "count": int(row["count"])} for row in rows]


def tag_missing_images(
    dataset_dir: Path,
    image_items: list[tuple[str, Path]],
    base_dir: Path,
    limit: int | None = None,
    progress_callback=None,
) -> dict:
    db = TagSearchDatabase(dataset_dir)
    db.sync_current_images(image_items)
    missing = db.missing_tag_images(image_items, limit=limit)
    total = len(missing)
    if progress_callback:
        progress_callback(0, total, "タグ処理開始")
    print(f"[tagging] タグ処理開始: dataset={dataset_dir.name} total={total}", flush=True)
    if not missing:
        print(f"[tagging] progress: 0/0 no missing images", flush=True)
        return {"tagged": 0, "missing_before": 0, "errors": []}

    tagger = get_tagger(base_dir)
    tagged = 0
    errors = []
    for processed, (split, path) in enumerate(missing, start=1):
        try:
            result = tagger.tag_image(path)
            db.upsert_tags(split, path, result.tags, result.inference_ms)
            tagged += 1
            message = f"Tagging {processed}/{total}: {path.name}"
        except Exception as e:
            errors.append({"filename": path.name, "error": str(e)})
            message = f"Tagging error {processed}/{total}: {path.name}"
        if progress_callback:
            progress_callback(processed, total, message)
        if processed <= 100:
            should_log = processed % 10 == 0
        elif processed <= 200:
            should_log = processed % 50 == 0
        else:
            should_log = processed % 100 == 0
        if should_log or processed == total:
            print(f"[tagging] progress: {processed}/{total} tagged={tagged} ({path.name})", flush=True)
    return {"tagged": tagged, "missing_before": len(missing), "errors": errors}
