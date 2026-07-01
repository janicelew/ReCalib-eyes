"""Download EyeCLIP checkpoint via local HTTP proxy (Clash 7890)."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import requests

FILE_ID = "1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr"
OUT = Path("D:/Projects/EyeCLIP/eyeclip_visual.pt")
PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
MIN_BYTES = 2_000_000_000
CHUNK = 1 << 20


def download() -> None:
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    url = "https://drive.google.com/uc?export=download&id=" + FILE_ID
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        url = url + "&confirm=" + token
        response = session.get(url, stream=True, timeout=120)
        response.raise_for_status()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with OUT.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded % (50 * CHUNK) == 0:
                print(f"\r{downloaded / 1e6:.1f} MB", end="", flush=True)

    size = OUT.stat().st_size
    print(f"\nSaved {OUT} ({size / 1e6:.1f} MB)")
    if size < MIN_BYTES:
        raise SystemExit(f"File too small ({size} bytes); download likely failed.")

    md5 = hashlib.md5()
    with OUT.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            md5.update(block)
    print("md5:", md5.hexdigest())


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:
        print("Download failed:", exc, file=sys.stderr)
        raise
