import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import requests

import hooks_share_upload as hook


def write_status(path: Path, data: dict) -> None:
    payload = dict(data)
    payload["updated_at"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class ProgressFile:
    def __init__(self, path: Path, total_bytes: int, status_path: Path, base_status: dict):
        self._fp = open(path, "rb")
        self._total_bytes = total_bytes
        self._status_path = status_path
        self._base_status = dict(base_status)
        self._sent_bytes = 0
        self._last_write = 0.0

    def __len__(self):
        return self._total_bytes

    def read(self, size=-1):
        chunk = self._fp.read(size)
        if chunk:
            self._sent_bytes += len(chunk)
            now = time.time()
            if now - self._last_write >= 5:
                status = dict(self._base_status)
                status["phase"] = "uploading"
                status["sent_bytes"] = self._sent_bytes
                status["progress"] = self._sent_bytes / self._total_bytes if self._total_bytes else 0.0
                write_status(self._status_path, status)
                self._last_write = now
        return chunk

    def close(self):
        self._fp.close()


def poll_remote_status(status_path: Path, status_url: str, upload_token: str, base_status: dict, stop_event: threading.Event) -> None:
    headers = {"Authorization": f"Bearer {upload_token}"}
    while not stop_event.wait(2):
        try:
            resp = requests.get(status_url, headers=headers, timeout=15)
            if not resp.ok:
                continue
            remote = resp.json()
            status = dict(base_status)
            status["phase"] = "uploading"
            status["sent_bytes"] = remote.get("bytes_received", 0)
            status["progress"] = remote.get("progress", 0.0) or 0.0
            status["remote_state"] = remote.get("state")
            write_status(status_path, status)
        except Exception:
            continue


def upload_with_curl(archive_path: Path, upload_url: str, upload_token: str) -> tuple[int, str]:
    curl_exe = shutil.which("curl")
    if not curl_exe:
        raise RuntimeError("curl not found")
    cmd = [
        curl_exe,
        "--http1.1",
        "--silent",
        "--show-error",
        "--output",
        "-",
        "--write-out",
        "\n%{http_code}",
        "-X",
        "PUT",
        "-H",
        f"Authorization: Bearer {upload_token}",
        "-H",
        "Content-Type: application/x-7z-compressed",
        "-T",
        str(archive_path),
        upload_url,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[:1000] or f"curl failed: {completed.returncode}")
    output = completed.stdout
    if "\n" not in output:
        raise RuntimeError(f"Unexpected curl response: {output[:1000]}")
    body, code = output.rsplit("\n", 1)
    return int(code.strip()), body.strip()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: upload_dataset_worker.py <dataset_name> <status_json>", file=sys.stderr)
        return 2

    dataset_name = sys.argv[1]
    status_path = Path(sys.argv[2]).resolve()
    datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    dataset_dir = datasets_dir / dataset_name

    if not dataset_dir.exists():
        write_status(status_path, {"phase": "failed", "error": f"Dataset not found: {dataset_name}"})
        return 1

    try:
        write_status(status_path, {"phase": "compressing", "dataset": dataset_name})
        with tempfile.TemporaryDirectory(prefix="dataset_archive_", dir=datasets_dir) as tmpdir:
            archive_path = Path(tmpdir) / f"{dataset_name}.7z"
            hook._create_7z_archive(dataset_dir, archive_path)
            archive_bytes = archive_path.stat().st_size

            headers = hook._auth_headers()
            base_url = hook._upload_base_url()
            public = hook._env_bool("UPLOAD_PUBLIC", True)
            reserve_payload = {
                "filename": archive_path.name,
                "size": archive_bytes,
                "content_type": "application/x-7z-compressed",
                "public": public,
            }
            reserve_resp = requests.post(
                f"{base_url}/api/uploads/reserve",
                headers=headers,
                json=reserve_payload,
                timeout=60,
            )
            hook._check_response(reserve_resp, "Upload reservation")
            reserve_data = reserve_resp.json()
            file_id = reserve_data["id"]
            upload_token = reserve_data["upload_token"]
            status_url = reserve_data.get("status_url") or f"{base_url}/api/uploads/{file_id}"
            upload_url = reserve_data.get("upload_url") or f"{base_url}/api/uploads/{file_id}/content"
            status_base = {
                "dataset": dataset_name,
                "archive_filename": archive_path.name,
                "archive_bytes": archive_bytes,
                "file_id": file_id,
                "status_url": status_url,
                "public": public,
            }
            write_status(status_path, {**status_base, "phase": "uploading", "sent_bytes": 0, "progress": 0.0})
            stop_event = threading.Event()
            poller = threading.Thread(
                target=poll_remote_status,
                args=(status_path, status_url, upload_token, status_base, stop_event),
                daemon=True,
            )
            poller.start()
            try:
                http_code, body = upload_with_curl(archive_path, upload_url, upload_token)
            finally:
                stop_event.set()
                poller.join(timeout=5)
            if http_code >= 400:
                raise RuntimeError(f"Upload failed: HTTP {http_code}: {body[:1000]}")
            upload_data = json.loads(body)
            download_url = (
                hook._absolute_url(base_url, upload_data.get("url"))
                or hook._absolute_url(base_url, reserve_data.get("url"))
            )
            if public and not download_url:
                download_url = f"{base_url}/f/{file_id}/{hook.quote(archive_path.name)}"

            write_status(
                status_path,
                {
                    **status_base,
                    "phase": "complete",
                    "sent_bytes": archive_bytes,
                    "progress": 1.0,
                    "download_url": download_url,
                    "upload_state": upload_data.get("state"),
                },
            )
        return 0
    except Exception as exc:
        write_status(
            status_path,
            {
                "phase": "failed",
                "dataset": dataset_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
