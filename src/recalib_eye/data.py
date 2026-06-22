from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
REFERABLE_GRADE_THRESHOLD = 2


def read_csv_auto(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", engine="python")

    return df.rename(columns={col: str(col).strip() for col in df.columns})


def parse_dr_grade(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Missing DR grade")

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    label_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "r0": 0,
        "r1": 1,
        "r2": 2,
        "r3": 3,
        "no dr": 0,
        "none": 0,
        "normal": 0,
        "mild": 1,
        "mild dr": 1,
        "moderate": 2,
        "moderate dr": 2,
        "severe": 3,
        "severe dr": 3,
        "proliferative": 4,
        "proliferative dr": 4,
        "pdr": 4,
    }
    if text in label_map:
        return label_map[text]

    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot parse DR grade: {value!r}") from exc


def dr_grade_to_referable(value: Any) -> int:
    grade = parse_dr_grade(value)
    if grade < 0 or grade > 4:
        raise ValueError(f"DR grade must be in 0-4, got {value!r}")
    return int(grade >= REFERABLE_GRADE_THRESHOLD)


def iter_image_files(root: str | Path):
    root = Path(root)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_image_index(image_dir: str | Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in iter_image_files(image_dir):
        index.setdefault(image_path.stem.lower(), image_path)
        index.setdefault(image_path.name.lower(), image_path)
    return index


def resolve_image_path(
    image_dir: str | Path,
    image_id: Any,
    preferred_ext: str = "",
    image_index: dict[str, Path] | None = None,
) -> Path:
    image_dir = Path(image_dir)
    image_id_text = str(image_id).strip()
    raw_path = Path(image_id_text)

    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    candidates: list[Path] = []
    if raw_path.suffix:
        candidates.append(image_dir / raw_path)
        stem = raw_path.stem
    else:
        stem = image_id_text

    suffixes = [preferred_ext, *IMAGE_SUFFIXES] if preferred_ext else list(IMAGE_SUFFIXES)
    for suffix in suffixes:
        candidates.append(image_dir / f"{stem}{suffix}")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    if image_index is not None:
        for key in (image_id_text.lower(), raw_path.name.lower(), stem.lower()):
            match = image_index.get(key)
            if match is not None:
                return match

    checked = "\n".join(str(candidate) for candidate in candidates[:12])
    raise FileNotFoundError(f"Image not found for id={image_id_text}. Checked:\n{checked}")


class FundusDRDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform,
        id_col: str,
        grade_col: str,
        preferred_ext: str = "",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.id_col = id_col
        self.grade_col = grade_col
        self.preferred_ext = preferred_ext

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        df = read_csv_auto(self.csv_path)
        missing = [col for col in (id_col, grade_col) if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {self.csv_path}: {missing}. Available: {list(df.columns)}")

        image_index = build_image_index(self.image_dir)
        rows = []
        for _, row in df.iterrows():
            image_path = resolve_image_path(
                self.image_dir,
                row[id_col],
                preferred_ext=preferred_ext,
                image_index=image_index,
            )
            grade = parse_dr_grade(row[grade_col])
            rows.append(
                {
                    "image_path": image_path,
                    "image_id": str(row[id_col]),
                    "grade": grade,
                    "label": int(grade >= REFERABLE_GRADE_THRESHOLD),
                }
            )

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        item = self.rows[index]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["label"], item["grade"], item["image_id"], str(item["image_path"])
