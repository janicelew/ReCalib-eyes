# APTOS → IDRiD linear probe

Frozen EyeCLIP visual encoder + logistic regression trained on APTOS2019 (source), evaluated on IDRiD (target).

## Run

- Date: 2026-07-09
- Config: `configs/aptos_to_idrid_linear_probe.json`
- Checkpoint: `D:/Projects/EyeCLIP/eyeclip_visual.pt`
- Device: CPU (auto), batch_size=8
- Command: `PYTHONPATH=src py -3.13 -m recalib_eye.linear_probe_dr --config configs/aptos_to_idrid_linear_probe.json`

## Data

| Dataset | Role | n | Images |
|---------|------|---|--------|
| APTOS2019 | source | 3662 | train_images PNGs |
| IDRiD | target | 516 | images JPGs |

## Key metrics (IDRiD target)

| Metric | Value |
|--------|-------|
| AUROC | 0.704 |
| AUPR | 0.812 |
| Macro F1 | 0.612 |
| Balanced accuracy | 0.631 |
| Sens @ Spec 0.90 | 0.319 |

Source APTOS OOF AUROC ≈ 0.936. Source-calibrated metrics and `source_calibration.json` are included.

## Files

- `metrics.csv` — uncalibrated metrics
- `metrics_calibrated.csv` — after source temperature/bias calibration
- `predictions_IDRiD.csv` — 516 target predictions
- `source_calibration.json` — fitted calibrator params
