from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import apply_probability_calibration, fit_source_calibration
from .data import FundusDRDataset
from .eyeclip_loader import load_eyeclip
from .metrics import binary_classification_report


@dataclass
class EncodedDataset:
    name: str
    role: str
    rows: pd.DataFrame
    features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source image-prototype DR evaluation.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def class_values_for_mode(mode: str) -> tuple[list[int], list[int]]:
    if mode == "binary":
        return [0, 1], [1]
    if mode == "grade_sum":
        return [0, 1, 2, 3, 4], [2, 3, 4]
    raise ValueError(f"Unsupported prototype_mode: {mode!r}")


def labels_for_mode(rows: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "binary":
        return rows["label_referable"].to_numpy(dtype=int)
    if mode == "grade_sum":
        return rows["grade"].to_numpy(dtype=int)
    raise ValueError(f"Unsupported prototype_mode: {mode!r}")


def build_class_prototypes(features: np.ndarray, class_labels: np.ndarray, class_values: list[int]) -> np.ndarray:
    prototypes = []
    for class_value in class_values:
        mask = class_labels == class_value
        if not np.any(mask):
            raise ValueError(f"No source samples found for prototype class {class_value}")
        prototype = features[mask].mean(axis=0)
        prototypes.append(prototype)
    return l2_normalize(np.stack(prototypes, axis=0))


def prototype_probs(
    features: np.ndarray,
    prototypes: np.ndarray,
    positive_class_indices: list[int],
    logit_scale: float,
) -> np.ndarray:
    logits = float(logit_scale) * features @ prototypes.T
    class_probs = softmax(logits, axis=1)
    return class_probs[:, positive_class_indices].sum(axis=1)


def leave_one_out_source_probs(
    features: np.ndarray,
    class_labels: np.ndarray,
    class_values: list[int],
    positive_class_indices: list[int],
    logit_scale: float,
) -> np.ndarray:
    class_sums = []
    class_counts = []
    for class_value in class_values:
        mask = class_labels == class_value
        class_sums.append(features[mask].sum(axis=0))
        class_counts.append(int(mask.sum()))
    class_sums = np.stack(class_sums, axis=0)
    class_counts = np.asarray(class_counts, dtype=int)

    probs = np.zeros(len(features), dtype=np.float64)
    class_to_index = {class_value: index for index, class_value in enumerate(class_values)}
    for row_idx, feature in enumerate(features):
        prototypes = class_sums.copy()
        own_class_idx = class_to_index[int(class_labels[row_idx])]
        if class_counts[own_class_idx] > 1:
            prototypes[own_class_idx] = prototypes[own_class_idx] - feature
            counts = class_counts.copy()
            counts[own_class_idx] -= 1
            prototypes = prototypes / counts[:, None]
        else:
            prototypes = prototypes / class_counts[:, None]
        prototypes = l2_normalize(prototypes)
        probs[row_idx] = prototype_probs(feature[None, :], prototypes, positive_class_indices, logit_scale)[0]
    return probs


@torch.no_grad()
def encode_dataset(
    name: str,
    role: str,
    dataset: FundusDRDataset,
    dataloader: DataLoader,
    bundle,
    cache_path: Path | None,
    reuse_cache: bool,
) -> EncodedDataset:
    if cache_path is not None and reuse_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        rows = pd.DataFrame(
            {
                "dataset": cached["dataset"],
                "image_id": cached["image_id"],
                "image_path": cached["image_path"],
                "grade": cached["grade"].astype(int),
                "label_referable": cached["label_referable"].astype(int),
            }
        )
        return EncodedDataset(name=name, role=role, rows=rows, features=cached["features"].astype(np.float32))

    features = []
    rows = []
    bundle.model.eval()
    for images, labels, grades, image_ids, image_paths in tqdm(dataloader, desc=f"encode {name}"):
        images = images.to(bundle.device, non_blocking=True)
        image_features = bundle.model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1).detach().cpu().numpy()
        features.append(image_features)
        for image_id, image_path, grade, label in zip(image_ids, image_paths, grades.numpy(), labels.numpy()):
            rows.append(
                {
                    "dataset": name,
                    "image_id": image_id,
                    "image_path": image_path,
                    "grade": int(grade),
                    "label_referable": int(label),
                }
            )

    encoded = EncodedDataset(
        name=name,
        role=role,
        rows=pd.DataFrame(rows),
        features=np.concatenate(features, axis=0).astype(np.float32),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            dataset=encoded.rows["dataset"].to_numpy(dtype=str),
            image_id=encoded.rows["image_id"].to_numpy(dtype=str),
            image_path=encoded.rows["image_path"].to_numpy(dtype=str),
            grade=encoded.rows["grade"].to_numpy(dtype=int),
            label_referable=encoded.rows["label_referable"].to_numpy(dtype=int),
            features=encoded.features,
        )
    return encoded


def cache_path_for_dataset(config: dict, output_dir: Path, dataset_cfg: dict) -> Path | None:
    if not config.get("feature_cache", True):
        return None
    cache_dir = Path(config.get("feature_cache_dir", output_dir / "feature_cache"))
    if not cache_dir.is_absolute():
        cache_dir = output_dir / cache_dir
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in dataset_cfg["name"])
    return cache_dir / f"{safe_name}.npz"


def make_dataloader(config: dict, dataset: FundusDRDataset, bundle) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=(bundle.device.type == "cuda"),
    )


def evaluate_image_prototypes(config: dict, encoded_datasets: list[EncodedDataset], output_dir: Path) -> None:
    prototype_mode = config.get("prototype_mode", "binary")
    class_values, positive_values = class_values_for_mode(prototype_mode)
    positive_class_indices = [class_values.index(value) for value in positive_values]
    source_roles = set(config.get("source_roles", ["source"]))
    logit_scale = float(config.get("logit_scale", 100.0))

    source_datasets = [dataset for dataset in encoded_datasets if dataset.role in source_roles]
    if not source_datasets:
        raise ValueError(f"No source datasets found for source_roles={sorted(source_roles)}")

    source_features = np.concatenate([dataset.features for dataset in source_datasets], axis=0)
    source_rows = pd.concat([dataset.rows for dataset in source_datasets], ignore_index=True)
    source_class_labels = labels_for_mode(source_rows, prototype_mode)
    source_prototypes = build_class_prototypes(source_features, source_class_labels, class_values)

    source_loo_probs = leave_one_out_source_probs(
        source_features,
        source_class_labels,
        class_values,
        positive_class_indices,
        logit_scale,
    )
    source_prob_slices = {}
    offset = 0
    for dataset in source_datasets:
        next_offset = offset + len(dataset.rows)
        source_prob_slices[dataset.name] = source_loo_probs[offset:next_offset]
        offset = next_offset

    predictions_by_dataset = {}
    metrics_rows = []
    for dataset in encoded_datasets:
        predictions = dataset.rows.copy()
        if dataset.role in source_roles and dataset.name in source_prob_slices:
            probs = source_prob_slices[dataset.name]
            scoring = "source_leave_one_out"
        else:
            probs = prototype_probs(dataset.features, source_prototypes, positive_class_indices, logit_scale)
            scoring = "source_prototype"
        predictions["prob_referable"] = probs
        predictions["pred_referable"] = (probs >= 0.5).astype(int)
        predictions["prototype_scoring"] = scoring
        predictions_by_dataset[dataset.name] = predictions

        metrics = binary_classification_report(
            predictions["label_referable"].to_numpy(),
            probs,
            target_specificity=float(config.get("target_specificity", 0.90)),
        )
        metrics["dataset"] = dataset.name
        metrics["role"] = dataset.role
        metrics["prototype_mode"] = prototype_mode
        metrics["prototype_scoring"] = scoring
        metrics_rows.append(metrics)

    calibration = fit_source_calibration(
        source_rows["label_referable"].to_numpy(),
        source_loo_probs,
        source_dataset="+".join(dataset.name for dataset in source_datasets),
        threshold_metric=config.get("threshold_metric", "balanced_accuracy"),
        fit_probability_calibrator=bool(config.get("fit_probability_calibrator", True)),
    )

    calibrated_metrics_rows = []
    for dataset in encoded_datasets:
        predictions = predictions_by_dataset[dataset.name]
        calibrated_probs = apply_probability_calibration(
            predictions["prob_referable"].to_numpy(),
            temperature=calibration.temperature,
            bias=calibration.bias,
        )
        predictions["prob_referable_calibrated"] = calibrated_probs
        predictions["pred_referable_calibrated"] = (calibrated_probs >= calibration.threshold).astype(int)
        metrics = binary_classification_report(
            predictions["label_referable"].to_numpy(),
            calibrated_probs,
            threshold=calibration.threshold,
            target_specificity=float(config.get("target_specificity", 0.90)),
        )
        metrics["dataset"] = dataset.name
        metrics["role"] = dataset.role
        metrics["prototype_mode"] = prototype_mode
        metrics["threshold"] = calibration.threshold
        metrics["temperature"] = calibration.temperature
        metrics["bias"] = calibration.bias
        calibrated_metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["dataset", "role", *[col for col in metrics_df.columns if col not in {"dataset", "role"}]]]
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    calibrated_metrics_df = pd.DataFrame(calibrated_metrics_rows)
    calibrated_metrics_df = calibrated_metrics_df[
        ["dataset", "role", *[col for col in calibrated_metrics_df.columns if col not in {"dataset", "role"}]]
    ]
    calibrated_metrics_df.to_csv(output_dir / "metrics_calibrated.csv", index=False)

    for dataset_name, predictions in predictions_by_dataset.items():
        predictions.to_csv(output_dir / f"predictions_{dataset_name}.csv", index=False)

    np.savez_compressed(
        output_dir / "source_image_prototypes.npz",
        prototype_mode=np.asarray([prototype_mode], dtype=str),
        class_values=np.asarray(class_values, dtype=int),
        positive_class_values=np.asarray(positive_values, dtype=int),
        prototypes=source_prototypes.astype(np.float32),
    )
    write_json(
        output_dir / "source_calibration.json",
        {
            **calibration.to_dict(),
            "prototype_mode": prototype_mode,
            "source_roles": sorted(source_roles),
            "logit_scale": logit_scale,
            "source_scoring": "leave_one_out",
        },
    )

    print(metrics_df.to_string(index=False))
    print("\nSource-calibrated metrics:")
    print(calibrated_metrics_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    output_dir = Path(config.get("output_dir", "outputs/image_prototype_dr"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_eyeclip(
        repo_path=config.get("eyeclip_repo"),
        checkpoint_path=config.get("checkpoint"),
        device=config.get("device", "auto"),
    )

    encoded_datasets = []
    for dataset_cfg in config["datasets"]:
        dataset = FundusDRDataset(
            csv_path=dataset_cfg["csv_path"],
            image_dir=dataset_cfg["image_dir"],
            transform=bundle.preprocess,
            id_col=dataset_cfg["id_col"],
            grade_col=dataset_cfg["grade_col"],
            preferred_ext=dataset_cfg.get("preferred_ext", ""),
            max_samples=dataset_cfg.get("max_samples"),
        )
        dataloader = make_dataloader(config, dataset, bundle)
        encoded_datasets.append(
            encode_dataset(
                name=dataset_cfg["name"],
                role=dataset_cfg.get("role", ""),
                dataset=dataset,
                dataloader=dataloader,
                bundle=bundle,
                cache_path=cache_path_for_dataset(config, output_dir, dataset_cfg),
                reuse_cache=bool(config.get("reuse_feature_cache", True)),
            )
        )

    evaluate_image_prototypes(config, encoded_datasets, output_dir)


if __name__ == "__main__":
    main()
