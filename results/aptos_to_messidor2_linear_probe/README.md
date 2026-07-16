# APTOS → MESSIDOR2 linear probe

Frozen EyeCLIP visual encoder + logistic regression trained on APTOS2019 (source), evaluated on MESSIDOR2 (target).

## Run

- Date: 2026-07-16
- Config: `configs/aptos_to_messidor2_linear_probe.json`
- Checkpoint: `/Users/lewjanice/Documents/EyeCLIP/eyeclip_visual.pt`
- Device: CPU (auto), batch_size=8
- Command: `PYTHONPATH=src python -m recalib_eye.linear_probe_dr --config configs/aptos_to_messidor2_linear_probe.json`
- MESSIDOR2 data: prepared from Hugging Face `OctoMed/Messidor2` via `scripts/prepare_messidor2.py`

## Data

| Dataset | Role | n | Notes |
|---------|------|---|-------|
| APTOS2019 | source | 3662 | train_images PNGs |
| MESSIDOR2 | target | 1744 | images JPGs, referable rate 0.262 |

## Key metrics (MESSIDOR2 target)

| Metric | Value |
|--------|-------|
| AUROC | 0.406 |
| AUPR | 0.226 |
| Macro F1 | 0.443 |
| Balanced accuracy | 0.496 |
| Sens @ Spec 0.90 | 0.055 |

Source APTOS OOF AUROC ≈ 0.936. Source-calibrated metrics and `source_calibration.json` are included.

## Files

- `metrics.csv` — uncalibrated metrics
- `metrics_calibrated.csv` — after source temperature/bias calibration
- `predictions_MESSIDOR2.csv` — 1744 target predictions
- `source_calibration.json` — fitted calibrator params
