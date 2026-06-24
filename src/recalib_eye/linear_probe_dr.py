from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_probability_calibration, fit_source_calibration
from .data import FundusDRDataset
from .eyeclip_loader import load_eyeclip
from .image_prototype_dr import cache_path_for_dataset, encode_dataset, make_dataloader, write_json
from .metrics import binary_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-only linear probe DR evaluation.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser.parse_args()


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_probe(config: dict):
    class_weight = config.get("class_weight", "balanced")
    if class_weight == "none":
        class_weight = None
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config.get("C", 1.0)),
            class_weight=class_weight,
            max_iter=int(config.get("max_iter", 2000)),
            solver=config.get("solver", "lbfgs"),
            random_state=int(config.get("random_seed", 42)),
        ),
    )


def fit_source_oof_probs(features: np.ndarray, labels: np.ndarray, config: dict) -> tuple[np.ndarray, int]:
    labels = np.asarray(labels, dtype=int)
    min_class_count = int(np.bincount(labels).min())
    requested_folds = int(config.get("cv_folds", 5))
    n_folds = max(2, min(requested_folds, min_class_count))
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=int(config.get("random_seed", 42)),
    )

    oof_probs = np.zeros(len(labels), dtype=float)
    for train_idx, valid_idx in splitter.split(features, labels):
        probe = make_probe(config)
        probe.fit(features[train_idx], labels[train_idx])
        oof_probs[valid_idx] = probe.predict_proba(features[valid_idx])[:, 1]
    return oof_probs, n_folds


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    output_dir = Path(config.get("output_dir", "outputs/linear_probe_dr"))
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

    source_roles = set(config.get("source_roles", ["source"]))
    source_datasets = [dataset for dataset in encoded_datasets if dataset.role in source_roles]
    if not source_datasets:
        raise ValueError(f"No source datasets found for source_roles={sorted(source_roles)}")

    source_features = np.concatenate([dataset.features for dataset in source_datasets], axis=0)
    source_rows = pd.concat([dataset.rows for dataset in source_datasets], ignore_index=True)
    source_labels = source_rows["label_referable"].to_numpy(dtype=int)

    source_oof_probs, n_folds = fit_source_oof_probs(source_features, source_labels, config)
    final_probe = make_probe(config)
    final_probe.fit(source_features, source_labels)

    source_prob_slices = {}
    offset = 0
    for dataset in source_datasets:
        next_offset = offset + len(dataset.rows)
        source_prob_slices[dataset.name] = source_oof_probs[offset:next_offset]
        offset = next_offset

    predictions_by_dataset = {}
    metrics_rows = []
    for dataset in encoded_datasets:
        predictions = dataset.rows.copy()
        if dataset.role in source_roles and dataset.name in source_prob_slices:
            probs = source_prob_slices[dataset.name]
            scoring = f"{n_folds}_fold_source_oof"
        else:
            probs = final_probe.predict_proba(dataset.features)[:, 1]
            scoring = "source_linear_probe"
        predictions["prob_referable"] = probs
        predictions["pred_referable"] = (probs >= 0.5).astype(int)
        predictions["linear_probe_scoring"] = scoring
        predictions_by_dataset[dataset.name] = predictions

        metrics = binary_classification_report(
            predictions["label_referable"].to_numpy(),
            probs,
            target_specificity=float(config.get("target_specificity", 0.90)),
        )
        metrics["dataset"] = dataset.name
        metrics["role"] = dataset.role
        metrics["linear_probe_scoring"] = scoring
        metrics_rows.append(metrics)

    calibration = fit_source_calibration(
        source_rows["label_referable"].to_numpy(),
        source_oof_probs,
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

    write_json(
        output_dir / "source_calibration.json",
        {
            **calibration.to_dict(),
            "source_roles": sorted(source_roles),
            "cv_folds": n_folds,
            "class_weight": config.get("class_weight", "balanced"),
            "C": float(config.get("C", 1.0)),
        },
    )

    print(metrics_df.to_string(index=False))
    print("\nSource-calibrated metrics:")
    print(calibrated_metrics_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
