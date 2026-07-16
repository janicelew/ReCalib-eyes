"""Prepare MESSIDOR-2 for ReCalib-Eye (flat labels.csv + images/).

Supports:
  1. Kaggle-style folder (same layout as configs/messidor2_zeroshot.yaml):
       raw/messidor_2.csv + raw/images/
  2. Hugging Face mirror (OctoMed/Messidor2) when raw files are missing.

Output:
  data/messidor2/labels.csv   (columns: image_id, grade)
  data/messidor2/images/      (JPG files)
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from recalib_eye.data import read_csv_auto

DEFAULT_RAW = Path("data/messidor2/raw")
DEFAULT_OUT = Path("data/messidor2")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    normalised = {str(col).strip().lower().replace("_", " "): col for col in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    raise ValueError(f"Could not find any of {candidates} in columns: {columns}")


def _copy_or_link(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_from_kaggle_raw(raw_dir: Path, out_dir: Path, copy_images: bool = True) -> pd.DataFrame:
    csv_candidates = list(raw_dir.glob("messidor*.csv")) + list(raw_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found under {raw_dir}")
    csv_path = csv_candidates[0]

    image_dir = raw_dir / "images"
    if not image_dir.exists():
        image_dir = raw_dir

    df = read_csv_auto(csv_path)
    id_col = _pick_column(list(df.columns), ("image path", "image_path", "image_id", "filename", "image"))
    grade_col = _pick_column(list(df.columns), ("diagnosis", "grade", "label", "retinopathy grade"))

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        image_name = Path(str(row[id_col])).name
        src = image_dir / image_name
        if not src.exists():
            matches = list(image_dir.rglob(image_name))
            if not matches:
                raise FileNotFoundError(f"Missing image for {image_name} under {image_dir}")
            src = matches[0]
        dst = out_images / image_name
        _copy_or_link(src, dst, copy_images=copy_images)
        rows.append({"image_id": image_name, "grade": int(float(row[grade_col]))})

    labels = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "labels.csv", index=False)
    return labels


def _parse_hf_grade(answer: str, options: list[str] | None = None) -> int:
    text = str(answer).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    if options:
        for idx, option in enumerate(options):
            if str(option).strip() == text:
                return idx
    raise ValueError(f"Cannot parse grade from answer={answer!r}")


def prepare_from_huggingface(out_dir: Path) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets: pip install datasets") from exc

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for split in ("train", "test"):
        stream = load_dataset("OctoMed/Messidor2", split=split, streaming=True)
        for example in stream:
            image_name = f"messidor2_{idx:04d}.jpg"
            image_path = out_images / image_name
            if not image_path.exists():
                image = example["image"]
                if getattr(image, "mode", None) != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, quality=95)
            grade = _parse_hf_grade(example["answer"], example.get("options"))
            rows.append({"image_id": image_name, "grade": grade})
            idx += 1

    labels = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "labels.csv", index=False)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MESSIDOR-2 for ReCalib-Eye.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW, help="Kaggle-style raw folder.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Prepared output folder.")
    parser.add_argument("--from-huggingface", action="store_true", help="Download OctoMed/Messidor2 mirror.")
    parser.add_argument("--no-copy", action="store_true", help="Symlink images instead of copying.")
    args = parser.parse_args()

    if args.from_huggingface or not args.raw_dir.exists():
        labels = prepare_from_huggingface(args.out_dir)
        source = "Hugging Face OctoMed/Messidor2"
    else:
        labels = prepare_from_kaggle_raw(args.raw_dir, args.out_dir, copy_images=not args.no_copy)
        source = str(args.raw_dir)

    print(f"Prepared MESSIDOR-2 from {source}")
    print(f"Samples: {len(labels)}")
    print(f"Grade range: {labels['grade'].min()}-{labels['grade'].max()}")
    print(f"Referable rate (grade>=2): {(labels['grade'] >= 2).mean():.3f}")
    print(f"Wrote {args.out_dir / 'labels.csv'} and {args.out_dir / 'images/'}")


if __name__ == "__main__":
    main()
