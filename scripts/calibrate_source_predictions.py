from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recalib_eye.calibration import apply_probability_calibration, fit_source_calibration
from recalib_eye.metrics import binary_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit source-only calibration from saved prediction CSVs.")
    parser.add_argument("--predictions", required=True, help="Source prediction CSV with label_referable and prob_referable.")
    parser.add_argument("--source-name", default="source", help="Name written to the calibration JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for calibrated predictions and JSON.")
    parser.add_argument(
        "--threshold-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "macro_f1", "youden_j", "f1"],
        help="Source metric used to choose the operating threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    required_cols = {"label_referable", "prob_referable"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns in {predictions_path}: {sorted(missing)}")

    calibration = fit_source_calibration(
        predictions["label_referable"].to_numpy(),
        predictions["prob_referable"].to_numpy(),
        source_dataset=args.source_name,
        threshold_metric=args.threshold_metric,
    )
    calibrated_probs = apply_probability_calibration(
        predictions["prob_referable"].to_numpy(),
        temperature=calibration.temperature,
        bias=calibration.bias,
    )
    predictions["prob_referable_calibrated"] = calibrated_probs
    predictions["pred_referable_calibrated"] = (calibrated_probs >= calibration.threshold).astype(int)

    raw_metrics = binary_classification_report(
        predictions["label_referable"].to_numpy(),
        predictions["prob_referable"].to_numpy(),
    )
    calibrated_metrics = binary_classification_report(
        predictions["label_referable"].to_numpy(),
        calibrated_probs,
        threshold=calibration.threshold,
    )

    predictions.to_csv(output_dir / f"predictions_{args.source_name}_calibrated.csv", index=False)
    pd.DataFrame(
        [
            {"setting": "raw", **raw_metrics},
            {
                "setting": "source_calibrated",
                **calibrated_metrics,
                "threshold": calibration.threshold,
                "temperature": calibration.temperature,
                "bias": calibration.bias,
            },
        ]
    ).to_csv(output_dir / "metrics_calibrated.csv", index=False)

    with open(output_dir / "source_calibration.json", "w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2)
        handle.write("\n")

    print(json.dumps(calibration.to_dict(), indent=2))
    print(f"Saved calibrated source outputs to {output_dir}")


if __name__ == "__main__":
    main()
