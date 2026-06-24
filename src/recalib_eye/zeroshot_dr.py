from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import apply_probability_calibration, fit_source_calibration
from .data import FundusDRDataset
from .eyeclip_loader import load_eyeclip
from .metrics import binary_classification_report
from .prototypes import (
    DR_REFERABLE_CLASS_NAMES,
    DR_GRADE_CLASS_NAMES,
    build_text_prototypes,
    dr_referable_probs_from_binary_logits,
    dr_referable_probs_from_grade_logits,
    image_text_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EyeCLIP zero-shot DR evaluation.")
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


def build_score_prototypes(config: dict, bundle):
    score_mode = config.get("score_mode", "grade_sum")
    if score_mode == "grade_sum":
        prompts_by_grade = config["prompts_by_grade"]
        ordered_prompts = {name: prompts_by_grade[name] for name in DR_GRADE_CLASS_NAMES}
        return score_mode, build_text_prototypes(bundle.model, bundle.clip, ordered_prompts, bundle.device)
    if score_mode == "binary_referable":
        prompts_by_referable = config["prompts_by_referable"]
        ordered_prompts = {name: prompts_by_referable[name] for name in DR_REFERABLE_CLASS_NAMES}
        return score_mode, build_text_prototypes(bundle.model, bundle.clip, ordered_prompts, bundle.device)
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


def referable_probs_from_logits(score_mode: str, logits: torch.Tensor) -> torch.Tensor:
    if score_mode == "grade_sum":
        return dr_referable_probs_from_grade_logits(logits)
    if score_mode == "binary_referable":
        return dr_referable_probs_from_binary_logits(logits)
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


@torch.no_grad()
def evaluate_dataset(name: str, dataset: FundusDRDataset, dataloader: DataLoader, bundle, text_features, score_mode: str):
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    rows = []

    bundle.model.eval()
    for images, labels, grades, image_ids, image_paths in tqdm(dataloader, desc=name):
        images = images.to(bundle.device, non_blocking=True)
        labels_np = labels.numpy().astype(int)
        image_features = bundle.model.encode_image(images)
        logits = image_text_logits(image_features, text_features)
        probs = referable_probs_from_logits(score_mode, logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels_np)
        for image_id, image_path, grade, label, prob in zip(image_ids, image_paths, grades.numpy(), labels_np, probs):
            rows.append(
                {
                    "dataset": name,
                    "image_id": image_id,
                    "image_path": image_path,
                    "grade": int(grade),
                    "label_referable": int(label),
                    "prob_referable": float(prob),
                    "pred_referable": int(prob >= 0.5),
                }
            )

    labels = np.concatenate(all_labels).astype(int)
    probs = np.concatenate(all_probs).astype(float)
    return binary_classification_report(labels, probs), pd.DataFrame(rows)


def maybe_apply_source_calibration(config: dict, output_dir: Path, predictions_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    calibration_cfg = config.get("source_calibration", {})
    if not calibration_cfg.get("enabled", False):
        return None

    source_roles = set(calibration_cfg.get("source_roles", ["source"]))
    source_frames = []
    source_names = []
    for dataset_cfg in config["datasets"]:
        role = dataset_cfg.get("role", "")
        name = dataset_cfg["name"]
        if role in source_roles:
            source_frames.append(predictions_by_dataset[name])
            source_names.append(name)

    if not source_frames:
        raise ValueError(f"No source datasets found for source_roles={sorted(source_roles)}")

    source_predictions = pd.concat(source_frames, ignore_index=True)
    calibration = fit_source_calibration(
        source_predictions["label_referable"].to_numpy(),
        source_predictions["prob_referable"].to_numpy(),
        source_dataset="+".join(source_names),
        threshold_metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
        fit_probability_calibrator=bool(calibration_cfg.get("fit_probability_calibrator", True)),
    )

    target_specificity = float(config.get("target_specificity", 0.90))
    calibrated_metrics = []
    for dataset_cfg in config["datasets"]:
        name = dataset_cfg["name"]
        predictions = predictions_by_dataset[name]
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
            target_specificity=target_specificity,
        )
        metrics["dataset"] = name
        metrics["role"] = dataset_cfg.get("role", "")
        metrics["threshold"] = calibration.threshold
        metrics["temperature"] = calibration.temperature
        metrics["bias"] = calibration.bias
        calibrated_metrics.append(metrics)

    calibration_path = calibration_cfg.get("output_path", "source_calibration.json")
    if not Path(calibration_path).is_absolute():
        calibration_path = output_dir / calibration_path
    write_json(
        calibration_path,
        {
            **calibration.to_dict(),
            "score_mode": config.get("score_mode", "grade_sum"),
            "source_roles": sorted(source_roles),
        },
    )

    metrics_df = pd.DataFrame(calibrated_metrics)
    metric_cols = ["dataset", "role", *[col for col in metrics_df.columns if col not in {"dataset", "role"}]]
    metrics_df = metrics_df[metric_cols]
    metrics_df.to_csv(output_dir / "metrics_calibrated.csv", index=False)
    return metrics_df


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = read_config(config_path)

    output_dir = Path(config.get("output_dir", "outputs/zeroshot_dr"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_eyeclip(
        repo_path=config.get("eyeclip_repo"),
        checkpoint_path=config.get("checkpoint"),
        device=config.get("device", "auto"),
    )

    score_mode, text_features = build_score_prototypes(config, bundle)

    metrics_rows = []
    predictions_by_dataset = {}
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
        dataloader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=(bundle.device.type == "cuda"),
        )

        metrics, predictions = evaluate_dataset(
            dataset_cfg["name"],
            dataset,
            dataloader,
            bundle,
            text_features,
            score_mode,
        )
        metrics["dataset"] = dataset_cfg["name"]
        metrics["role"] = dataset_cfg.get("role", "")
        metrics_rows.append(metrics)
        predictions_by_dataset[dataset_cfg["name"]] = predictions

    calibrated_metrics_df = maybe_apply_source_calibration(config, output_dir, predictions_by_dataset)

    for dataset_name, predictions in predictions_by_dataset.items():
        predictions.to_csv(output_dir / f"predictions_{dataset_name}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metric_cols = ["dataset", "role", *[col for col in metrics_df.columns if col not in {"dataset", "role"}]]
    metrics_df = metrics_df[metric_cols]
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    if calibrated_metrics_df is not None:
        print("\nSource-calibrated metrics:")
        print(calibrated_metrics_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
