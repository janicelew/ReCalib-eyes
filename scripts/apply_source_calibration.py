"""Apply a pre-fitted APTOS source calibration to target prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recalib_eye.calibration import apply_probability_calibration
from recalib_eye.metrics import binary_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply existing source_calibration.json to target predictions."
    )
    parser.add_argument("--predictions", required=True, help="Target predictions CSV.")
    parser.add_argument("--calibration", required=True, help="source_calibration.json from APTOS.")
    parser.add_argument("--output-dir", required=True, help="Directory for calibrated outputs.")
    parser.add_argument("--dataset-name", default="IDRiD")
    parser.add_argument("--target-specificity", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)

    threshold = float(calibration["threshold"])
    temperature = float(calibration["temperature"])
    bias = float(calibration["bias"])

    calibrated_probs = apply_probability_calibration(
        predictions["prob_referable"].to_numpy(),
        temperature=temperature,
        bias=bias,
    )
    predictions = predictions.copy()
    predictions["prob_referable_calibrated"] = calibrated_probs
    predictions["pred_referable_calibrated"] = (calibrated_probs >= threshold).astype(int)

    metrics = binary_classification_report(
        predictions["label_referable"].to_numpy(),
        calibrated_probs,
        threshold=threshold,
        target_specificity=args.target_specificity,
    )
    metrics["dataset"] = args.dataset_name
    metrics["role"] = "target"
    metrics["threshold"] = threshold
    metrics["temperature"] = temperature
    metrics["bias"] = bias

    predictions.to_csv(output_dir / f"predictions_{args.dataset_name}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_calibrated.csv", index=False)

    dest_cal = output_dir / "source_calibration.json"
    if Path(args.calibration).resolve() != dest_cal.resolve():
        dest_cal.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved calibrated outputs to {output_dir}")


if __name__ == "__main__":
    main()
