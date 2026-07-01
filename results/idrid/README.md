# IDRiD Target Dataset Setup

IDRiD is used as an **external target** dataset for APTOS2019 -> IDRiD cross-dataset evaluation.
Do not tune prompts, thresholds, temperatures, or model weights on IDRiD labels.

## Expected layout

Place files under the repo root (same style as APTOS2019):

```text
data/idrid/
├── labels.csv
└── images/
    ├── train_IDRiD_001.jpg
    ├── test_IDRiD_001.jpg
    └── ...
```

`labels.csv` columns:

```text
image_id,grade
```

- `grade` is DR severity 0-4 (ICDR).
- Referable mapping in code: grade 0-1 -> non-referable, grade 2-4 -> referable.
- Train and test splits reuse the same numeric IDs in the original release, so prepared
  IDs are prefixed: `train_IDRiD_001`, `test_IDRiD_001`, etc.

## Download

Official source: [IEEE DataPort IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

Download **B. Disease Grading.zip** (516 fundus images + grading CSVs).

Public mirror (same ZIP files):

```text
https://zenodo.org/records/17219542
```

Expected ZIP size: `212405123` bytes. Expected MD5: `b9239a4b956021a1cf0225522f11f58f`.

## Prepare

1. Unzip `B. Disease Grading.zip` to `data/idrid/raw/`.
2. Run:

```bash
PYTHONPATH=src python scripts/prepare_idrid.py
```

Optional download helper (resume-friendly Zenodo mirror):

```bash
python scripts/download_idrid.py
```

## Validate (no APTOS required)

```bash
PYTHONPATH=src python scripts/validate_idrid.py
```

Expected output: `516` samples, referable rate about `0.63`.

Local verification (2026-06-28):

```text
Samples: 516
Train IDs: 413
Test IDs: 103
Grade range: 0-4
Referable rate: 0.626
```

Smoke test on one IDRiD image (set local EyeCLIP paths):

```bash
PYTHONPATH=src python scripts/smoke_eyeclip.py \
  --eyeclip-repo /path/to/EyeCLIP \
  --checkpoint /path/to/EyeCLIP/eyeclip_visual.pt \
  --image data/idrid/images/train_IDRiD_001.jpg
```

## Run APTOS -> IDRiD evaluation

Edit `eyeclip_repo` and `checkpoint` in the config for your machine, then:

```bash
PYTHONPATH=src python -m recalib_eye.linear_probe_dr --config configs/aptos_to_idrid_linear_probe.json
```

Requires APTOS2019 source data at:

```text
data/aptos2019/train.csv
data/aptos2019/train_images/
```

IDRiD is evaluated with the source-trained linear probe and source-only calibration.
No target tuning should occur.

## Path overrides

If your data lives outside the repo, pass explicit paths to validation:

```bash
PYTHONPATH=src python scripts/validate_idrid.py \
  --csv-path /path/to/labels.csv \
  --image-dir /path/to/images
```

For experiments, edit `csv_path` and `image_dir` for the IDRiD block in
`configs/aptos_to_idrid_linear_probe.json`.
