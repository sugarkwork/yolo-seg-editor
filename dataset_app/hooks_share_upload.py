"""Dataset archive/upload hook for Sugar Knight Share.

This module contains no secrets. dataset_app/hooks_local.py imports and
registers it, while credentials live in dataset_app/.env.
"""
import os
import shutil
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, BackgroundTasks

_ENV_PATH = Path(__file__).parent / ".env"


def _load_env_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    _load_env_fallback(_ENV_PATH)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_7zip() -> Optional[str]:
    configured = os.environ.get("UPLOAD_7ZIP_PATH", "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        # If configured path doesn't exist, don't fail immediately. Fallback to auto-detection.

    for name in ("7z", "7zz", "7za"):
        exe = shutil.which(name)
        if exe:
            return exe

    candidates = [
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files (x86)/7-Zip/7z.exe"),
        Path("C:/Program Files/7-Zip/7zz.exe"),
        Path("C:/Program Files (x86)/7-Zip/7zz.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _create_archive(dataset_dir: Path, archive_path: Path) -> str:
    """Create an archive. Try 7z first; fallback to zip using Python standard zipfile if 7z is missing."""
    exe = _find_7zip()
    if exe:
        timeout = int(os.environ.get("UPLOAD_ARCHIVE_TIMEOUT_SECONDS", "3600"))
        dict_size = os.environ.get("UPLOAD_7ZIP_DICT_SIZE", "256m").strip() or "256m"
        cmd = [
            exe,
            "a",
            "-t7z",
            "-mx=9",
            "-m0=lzma2",
            f"-md={dict_size}",
            "-mfb=273",
            "-ms=on",
            "-y",
            "-bd",
            str(archive_path.with_suffix(".7z")),
            dataset_dir.name,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(dataset_dir.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if completed.returncode == 0 and archive_path.with_suffix(".7z").exists():
                return "7z"
        except Exception:
            pass # fallback to zip

    # ZIP Fallback using Python standard library (no external dependency, fully cross-platform)
    import zipfile
    zip_path = archive_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = Path(dataset_dir.name) / file_path.relative_to(dataset_dir)
                zipf.write(file_path, rel_path)
    return "zip"


def _upload_base_url() -> str:
    base_url = os.environ.get("UPLOAD_BASE_URL", "").strip()
    if not base_url:
        legacy_url = os.environ.get("UPLOAD_URL", "").strip()
        if legacy_url.endswith("/api/uploads/reserve"):
            base_url = legacy_url[: -len("/api/uploads/reserve")]
        else:
            base_url = legacy_url
    return base_url.rstrip("/")


def _auth_headers() -> dict:
    token = os.environ.get("UPLOAD_TOKEN", "").strip()
    scheme = os.environ.get("UPLOAD_AUTH_SCHEME", "Bearer").strip()
    if not token:
        return {}
    return {"Authorization": f"{scheme} {token}".strip()}


def _response_detail(resp) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:500]
    detail = data.get("detail") if isinstance(data, dict) else None
    return str(detail or data)[:500]


def _check_response(resp, action: str) -> None:
    if resp.status_code < 400:
        return
    raise HTTPException(
        status_code=resp.status_code if resp.status_code < 500 else 502,
        detail=f"{action} failed: HTTP {resp.status_code}: {_response_detail(resp)}",
    )


def _absolute_url(base_url: str, url):
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{base_url}{url}"
    return f"{base_url}/{url}"


def _upload_to_share_server(archive_path: Path, filename: str, content_type: str) -> dict:
    base_url = _upload_base_url()
    if not base_url:
        raise HTTPException(status_code=500, detail="UPLOAD_BASE_URL not configured in .env")

    try:
        import requests
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="requests not installed. Run: pip install requests python-dotenv",
        )

    public = _env_bool("UPLOAD_PUBLIC", True)
    archive_size = archive_path.stat().st_size
    headers = _auth_headers()

    reserve_payload = {
        "filename": filename,
        "size": archive_size,
        "content_type": content_type,
        "public": public,
    }

    try:
        reserve_resp = requests.post(
            f"{base_url}/api/uploads/reserve",
            headers=headers,
            json=reserve_payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upload reservation failed: {e}")
    _check_response(reserve_resp, "Upload reservation")

    reserve_data = reserve_resp.json()
    file_id = reserve_data.get("id")
    if not file_id:
        raise HTTPException(status_code=502, detail="Upload reservation response did not include an id")
    upload_token = reserve_data.get("upload_token")
    if not upload_token:
        raise HTTPException(status_code=502, detail="Upload reservation response did not include an upload_token")

    upload_url = reserve_data.get("upload_url") or f"{base_url}/api/uploads/{file_id}/content"
    upload_timeout = int(os.environ.get("UPLOAD_TIMEOUT_SECONDS", "3600"))
    put_headers = {
        "Authorization": f"Bearer {upload_token}",
        "Content-Type": content_type,
    }
    try:
        with open(archive_path, "rb") as f:
            upload_resp = requests.put(upload_url, headers=put_headers, data=f, timeout=upload_timeout)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upload failed: {e}")
    _check_response(upload_resp, "Upload")

    upload_data = upload_resp.json()
    download_url = (
        _absolute_url(base_url, upload_data.get("url"))
        or _absolute_url(base_url, reserve_data.get("url"))
    )
    if public and not download_url:
        download_url = f"{base_url}/f/{file_id}/{quote(filename)}"

    return {
        "file_id": file_id,
        "download_url": download_url,
        "public": public,
        "reserve_response": reserve_data,
        "upload_response": upload_data,
    }


def bg_archive_upload(dataset_name: str, ds_dir: Path, datasets_dir: Path, status_file: Path):
    try:
        # 1. Compressing Phase
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status": "compressing", "progress": 0.0}, f)

        filename = f"{dataset_name}.7z"
        with tempfile.TemporaryDirectory(prefix="dataset_archive_", dir=datasets_dir) as tmpdir:
            base_archive_path = Path(tmpdir) / dataset_name
            fmt = _create_archive(ds_dir, base_archive_path)
            
            if fmt == "7z":
                archive_path = base_archive_path.with_suffix(".7z")
                filename = f"{dataset_name}.7z"
                content_type = "application/x-7z-compressed"
            else:
                archive_path = base_archive_path.with_suffix(".zip")
                filename = f"{dataset_name}.zip"
                content_type = "application/zip"
                
            archive_size = archive_path.stat().st_size
            
            # 2. Uploading Phase
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": "uploading", "progress": 0.5, "filename": filename, "format": fmt}, f)
                
            upload_result = _upload_to_share_server(archive_path, filename, content_type)
            
        # 3. Complete Phase
        payload = {
            "status": "complete",
            "progress": 1.0,
            "archive_filename": filename,
            "archive_format": fmt,
            "bytes_uploaded": archive_size,
            "file_id": upload_result["file_id"],
            "download_url": upload_result["download_url"],
            "url": upload_result["download_url"],
            "public": upload_result["public"],
            "upload_state": upload_result["upload_response"].get("state"),
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        # Print the link loudly in the console so it's impossible to lose
        print("\n" + "="*80)
        print(f"[UPLOAD SUCCESS] Dataset '{dataset_name}' archived and uploaded successfully!")
        print(f"File Name:     {filename}")
        print(f"Download URL:  {upload_result['download_url']}")
        print("="*80 + "\n", flush=True)
            
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"\n[UPLOAD ERROR] Failed to upload dataset '{dataset_name}': {err_msg}", flush=True)
        traceback.print_exc()
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "error_msg": err_msg}, f)
        except Exception:
            pass


def register(app: FastAPI, datasets_dir: Path):
    @app.post("/api/dataset/{dataset_name}/archive_upload")
    def api_archive_upload(dataset_name: str, bg_tasks: BackgroundTasks):
        ds_dir = datasets_dir / dataset_name
        if not ds_dir.exists() or not ds_dir.is_dir():
            raise HTTPException(status_code=404, detail="Dataset not found")

        status_file = datasets_dir / dataset_name / "upload_status.json"
        
        # Avoid duplicate runs if already compressing or uploading
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    curr = json.load(f)
                    if curr.get("status") in ("compressing", "uploading"):
                        return {"status": "running", "msg": "Already running"}
            except Exception:
                pass

        # Start non-blocking background task
        bg_tasks.add_task(bg_archive_upload, dataset_name, ds_dir, datasets_dir, status_file)
        return {"status": "started"}

    @app.get("/api/dataset/{dataset_name}/upload_status")
    def api_upload_status(dataset_name: str):
        status_file = datasets_dir / dataset_name / "upload_status.json"
        if not status_file.exists():
            return {"status": "idle"}
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "error_msg": str(e)}
