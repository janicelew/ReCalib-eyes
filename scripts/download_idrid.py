"""Resume-friendly download for IDRiD B. Disease Grading.zip (Zenodo mirror)."""

from __future__ import annotations

import time
from pathlib import Path

import requests

URL = "https://zenodo.org/api/records/17219542/files/B.%20Disease%20Grading.zip/content"
OUT = Path("data/idrid/B_Disease_Grading.zip")
EXPECTED_BYTES = 212405123
CHUNK = 1024 * 1024


def download_once() -> bool:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    start = OUT.stat().st_size if OUT.exists() else 0
    if start >= EXPECTED_BYTES:
        return True

    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(URL, headers=headers, stream=True, timeout=300) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

        total_header = response.headers.get("Content-Length")
        total_bytes = None
        if total_header:
            total_bytes = int(total_header) + (start if response.status_code == 206 else 0)

        mode = "ab" if start else "wb"
        downloaded = start
        with OUT.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = downloaded * 100 / total_bytes
                    print(
                        f"\r{downloaded / 1e6:.1f} MB / {total_bytes / 1e6:.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

    print()
    return OUT.stat().st_size >= EXPECTED_BYTES


def main() -> None:
    attempt = 0
    while True:
        attempt += 1
        current = OUT.stat().st_size if OUT.exists() else 0
        if current >= EXPECTED_BYTES:
            print(f"Already complete: {OUT} ({current / 1e6:.1f} MB)")
            return
        try:
            if download_once():
                print(f"Done: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
                return
        except Exception as exc:
            print(f"Attempt {attempt} failed at {current / 1e6:.1f} MB: {exc}")
        time.sleep(min(60, 5 + attempt))


if __name__ == "__main__":
    main()
