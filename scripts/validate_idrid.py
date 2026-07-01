"""Validate local IDRiD layout for ReCalib-Eye (no APTOS required)."""

from __future__ import annotations

import argparse
from pathlib import Path

from recalib_eye.data import FundusDRDataset, REFERABLE_GRADE_THRESHOLD

DEFAULT_CSV = Path("data/idrid/labels.csv")
DEFAULT_IMAGES = Path("data/idrid/images")


def validate(csv_path: Path, image_dir: Path) -> None:
    dataset = FundusDRDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=None,
        id_col="image_id",
        grade_col="grade",
        preferred_ext=".jpg",
    )
    n = len(dataset)
    referable = sum(row["label"] for row in dataset.rows) / n
    grades = [row["grade"] for row in dataset.rows]
    bad_grades = [g for g in grades if g < 0 or g > 4]
    train_n = sum(1 for row in dataset.rows if str(row["image_id"]).startswith("train_"))
    test_n = sum(1 for row in dataset.rows if str(row["image_id"]).startswith("test_"))

    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")
    print(f"Samples: {n}")
    print(f"Train IDs: {train_n}")
    print(f"Test IDs: {test_n}")
    print(f"Grade range: {min(grades)}-{max(grades)}")
    print(f"Referable rate (grade>={REFERABLE_GRADE_THRESHOLD}): {referable:.3f}")
    if bad_grades:
        raise SystemExit(f"Invalid grades found: {bad_grades[:10]}")
    if n != 516:
        raise SystemExit(f"Expected 516 samples, got {n}")
    print("IDRiD validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IDRiD labels and image paths.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args()
    validate(args.csv_path.resolve(), args.image_dir.resolve())


if __name__ == "__main__":
    main()
