"""Prepare IDRiD for ReCalib-Eye (harmonised labels + flat image folder).

Usage (after downloading B. Disease Grading.zip from IEEE DataPort):

  1. Unzip to e.g. D:\\cursor\\datasets\\idrid\\raw
  2. Run:
     python scripts/prepare_idrid.py --raw-dir D:/cursor/datasets/idrid/raw

Output:
  D:/cursor/datasets/idrid/images/   (all JPGs, train + test)
  D:/cursor/datasets/idrid/labels.csv  (columns: image_id, grade)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from recalib_eye.data import read_csv_auto

DEFAULT_RAW = Path("data/idrid/raw")
DEFAULT_OUT = Path("data/idrid")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

ID_CANDIDATES = (
    "image name",
    "image no",
    "image_id",
    "image",
    "id",
    "filename",
)
GRADE_CANDIDATES = (
    "retinopathy grade",
    "dr grade",
    "grade",
    "diagnosis",
    "label",
)


def _normalise_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalised = {_normalise_col(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    return None


def _find_label_csvs(raw_dir: Path) -> list[Path]:
    found: list[Path] = []
    for path in raw_dir.rglob("*.csv"):
        name = path.name.lower()
        if "diseasegrading" in name.replace("_", "").replace(" ", "") and "label" in name:
            found.append(path)
    return sorted(found)


def _find_image_roots(raw_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if "original" in name or "training set" in name or "testing set" in name:
            roots.append(path)
    if not roots:
        roots = [raw_dir]
    return roots


def _split_from_path(path: Path) -> str:
    text = str(path).lower()
    if "testing set" in text or "test set" in text:
        return "test"
    if "training set" in text or "train set" in text:
        return "train"
    return ""


def _split_from_csv(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if "testing" in name:
        return "test"
    if "training" in name:
        return "train"
    return ""


def _collect_images(search_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in search_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            split = _split_from_path(path)
            stem = path.stem.lower()
            if split:
                index.setdefault(f"{split}_{stem}", path)
            index.setdefault(stem, path)
            index.setdefault(path.name.lower(), path)
    return index


def _load_labels(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = read_csv_auto(csv_path)
        id_col = _pick_column(list(df.columns), ID_CANDIDATES)
        grade_col = _pick_column(list(df.columns), GRADE_CANDIDATES)
        if id_col is None or grade_col is None:
            raise ValueError(
                f"Cannot map columns in {csv_path}. "
                f"Columns: {list(df.columns)}. "
                f"Expected image id + DR grade columns."
            )
        split = _split_from_csv(csv_path)
        subset = df[[id_col, grade_col]].copy()
        subset.columns = ["image_id", "grade"]
        subset["image_id"] = subset["image_id"].astype(str).str.strip()
        if split:
            subset["image_id"] = split + "_" + subset["image_id"]
        frames.append(subset)
    if not frames:
        raise FileNotFoundError("No IDRiD label CSV files found under raw directory.")
    merged = pd.concat(frames, ignore_index=True)
    merged["grade"] = pd.to_numeric(merged["grade"], errors="raise").astype(int)
    return merged.drop_duplicates(subset=["image_id"], keep="first")


def prepare_idrid(raw_dir: Path, out_dir: Path, copy_images: bool = True) -> pd.DataFrame:
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    label_csvs = _find_label_csvs(raw_dir)
    if not label_csvs:
        raise FileNotFoundError(
            f"No IDRiD_DiseaseGrading_*Labels*.csv under {raw_dir}. "
            "Unzip B. Disease Grading.zip into this folder first."
        )

    labels = _load_labels(label_csvs)
    image_index = _collect_images(_find_image_roots(raw_dir))

    missing: list[str] = []
    copied = 0
    for image_id in labels["image_id"]:
        image_id_text = str(image_id).strip()
        key = Path(image_id_text).stem.lower()
        source = (
            image_index.get(key)
            or image_index.get(image_id_text.lower())
            or image_index.get(Path(image_id_text).name.lower())
        )
        if source is None:
            missing.append(image_id_text)
            continue
        if copy_images:
            target = images_dir / f"{image_id_text}{source.suffix.lower()}"
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1

    labels_path = out_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)

    referable = (labels["grade"] >= 2).mean()
    print(f"Label CSVs: {len(label_csvs)}")
    print(f"Labels written: {labels_path} ({len(labels)} rows)")
    print(f"Referable rate (grade>=2): {referable:.3f}")
    if copy_images:
        print(f"Images in {images_dir}: {len(list(images_dir.glob('*')))} (new copies: {copied})")
    if missing:
        print(f"WARNING: {len(missing)} label rows without matching images (first 5): {missing[:5]}")
    else:
        print("All label rows matched an image file.")

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IDRiD for ReCalib-Eye.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="Folder containing unzipped B. Disease Grading content.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output folder for images/ and labels.csv.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only write labels.csv; do not copy images.",
    )
    args = parser.parse_args()
    prepare_idrid(args.raw_dir, args.out_dir, copy_images=not args.no_copy)


if __name__ == "__main__":
    main()
