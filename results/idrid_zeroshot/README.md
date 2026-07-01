# IDRiD Zero-Shot Deliverables

## Status

Steps 1–2 **done**. Steps 3–5 **blocked** until `D:/Projects/EyeCLIP/eyeclip_visual.pt` exists (~2.14 GB from Google Drive).

## Already in this folder

| File | Description |
|------|-------------|
| `idrid_zeroshot.json` | Config used for the run |
| `source_calibration.json` | APTOS source calibration (not refit on IDRiD) |
| `README.md` | This file |

## Still needed (after checkpoint download)

| File | Description |
|------|-------------|
| `predictions_IDRiD.csv` | 516 rows, zero-shot scores |
| `metrics.csv` | Raw IDRiD metrics |
| `metrics_calibrated.csv` | IDRiD metrics after APTOS calibration |

## Finish commands (VPN on, global mode)

Download checkpoint to `D:/Projects/EyeCLIP/eyeclip_visual.pt`, then:

```powershell
cd D:\Projects\ReCalib-eyes
powershell -ExecutionPolicy Bypass -File scripts\run_idrid_zeroshot_pipeline.ps1
```

Or manually:

```powershell
cd D:\Projects\ReCalib-eyes
$env:PYTHONPATH = "src"
py -3.13 -m recalib_eye.zeroshot_dr --config configs/idrid_zeroshot.json
py -3.13 scripts/apply_source_calibration.py `
  --predictions outputs/idrid_zeroshot/predictions_IDRiD.csv `
  --calibration outputs/aptos2019_source_calibration/source_calibration.json `
  --output-dir outputs/idrid_zeroshot_calibrated `
  --dataset-name IDRiD
```

Then copy the three CSV files into `results/idrid_zeroshot/` and push to GitHub.

## Source calibration applied (from APTOS)

- temperature: `20.0`
- bias: `-0.4`
- threshold: `0.4010364229230885` (balanced_accuracy on APTOS only)
- n_source: `3662`

## IDRiD data verified

- 516 samples, referable rate 0.626
- Path: `D:/Projects/ReCalib-eyes/data/idrid/` (junction to `D:/Projects/datasets/idrid/`)
