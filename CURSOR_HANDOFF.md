# Cursor Handoff: ReCalib-Eye Current Situation

This note is for Cursor or any teammate joining the project without the chat
history. Read this first before changing code.

## Project Goal

ReCalib-Eye is testing how to adapt EyeCLIP for diabetic retinopathy under
cross-dataset shift.

Current task:

1. Use APTOS2019 as the source dataset.
2. Learn source-only adaptation/calibration on APTOS2019.
3. Later evaluate on target datasets such as IDRiD or MESSIDOR2.
4. Do not tune prompts, thresholds, temperatures, or model weights on target
   labels.

## Local Setup

Repo:

```text
/Users/lewjanice/Documents/recalib-eye
```

EyeCLIP repo:

```text
/Users/lewjanice/Documents/EyeCLIP
```

EyeCLIP checkpoint:

```text
/Users/lewjanice/Documents/EyeCLIP/eyeclip_visual.pt
```

Python environment:

```bash
/Users/lewjanice/miniconda3/envs/recalib-eye/bin/python
```

Use commands with:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python ...
```

Local machine is CPU-only for this project, so full image encoding can be slow.

## Dataset Status

APTOS2019 exists locally:

```text
data/aptos2019/train.csv
data/aptos2019/train_images/
```

APTOS2019 has 3,662 labelled training images.

Label mapping:

```text
DR grade 0-1 -> non-referable
DR grade 2-4 -> referable
```

APTOS2019 is the source dataset. Targets are not set up yet.

## What Failed

### 1. Prompt-only zero-shot is weak

Config:

```text
configs/aptos2019_zeroshot.json
```

Result:

```text
results/aptos2019_zeroshot/
```

Raw APTOS2019 zero-shot:

```text
AUROC = 0.4856
AUPR = 0.4038
balanced accuracy = 0.4950
ECE = 0.0838
Brier = 0.2501
```

Conclusion: text prompt zero-shot is basically random on APTOS2019.

### 2. Binary referable/non-referable prompts are also bad

Config:

```text
configs/aptos2019_binary_zeroshot.json
```

User ran full binary prompt result:

```text
AUROC = 0.3981
AUPR = 0.3647
calibrated balanced accuracy = 0.5023
ECE = 0.0049
Brier = 0.2414
```

Conclusion: do not continue prompt-only work as the main method.

### 3. Image prototypes are not strong enough

Configs:

```text
configs/aptos2019_image_prototypes_256.json
configs/aptos2019_image_prototypes_256_grade.json
```

256-image binary prototype result:

```text
AUROC = 0.5013
calibrated balanced accuracy = 0.5825
```

256-image grade prototype result:

```text
AUROC = 0.4656
calibrated balanced accuracy = 0.5792
```

Conclusion: image prototypes are better than bad prompts in threshold metrics,
but the ranking signal is still weak.

## What Worked

### APTOS source linear probe is strong

Main config:

```text
configs/aptos2019_linear_probe.json
```

Code:

```text
src/recalib_eye/linear_probe_dr.py
```

Tracked result:

```text
results/aptos2019_linear_probe/
```

Full APTOS2019 source-only 5-fold out-of-fold result:

```text
n = 3662
positive_rate = 0.4061
AUROC = 0.9360
AUPR = 0.9009
macro-F1 = 0.8588
balanced accuracy = 0.8650
sensitivity at specificity 0.90 = 0.7976
ECE = 0.0325
Brier = 0.1003
```

Source-calibrated result:

```text
threshold = 0.4187
temperature = 1.1205
bias = -0.3000
macro-F1 = 0.8606
balanced accuracy = 0.8672
ECE = 0.0195
Brier = 0.0991
```

Conclusion: EyeCLIP image features are useful. The text prompt route is weak,
but image features plus an APTOS source linear probe are strong.

## Current Main Method

Use this as the main ReCalib-Eye method for now:

```text
EyeCLIP image encoder
-> image embeddings
-> source-only APTOS linear probe
-> source-only calibration
-> target dataset evaluation without target tuning
```

## Important Files Added/Changed

Core new code:

```text
src/recalib_eye/calibration.py
src/recalib_eye/image_prototype_dr.py
src/recalib_eye/linear_probe_dr.py
scripts/calibrate_source_predictions.py
```

Important configs:

```text
configs/aptos2019_linear_probe.json
configs/aptos2019_linear_probe_256.json
configs/aptos2019_image_prototypes.json
configs/aptos2019_image_prototypes_256.json
configs/aptos2019_binary_zeroshot.json
```

Important result folders:

```text
results/aptos2019_zeroshot/
results/aptos2019_linear_probe/
```

Local output folders under `outputs/` are ignored by Git. They may contain
cached image features and prediction CSVs.

## Commands Already Used

Raw zero-shot:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.zeroshot_dr --config configs/aptos2019_zeroshot.json
```

Binary prompt zero-shot:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.zeroshot_dr --config configs/aptos2019_binary_zeroshot.json
```

Image prototype subset:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.image_prototype_dr --config configs/aptos2019_image_prototypes_256.json
```

Linear probe subset:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.linear_probe_dr --config configs/aptos2019_linear_probe_256.json
```

Full APTOS linear probe:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.linear_probe_dr --config configs/aptos2019_linear_probe.json
```

## What The Team Should Work On Next

### Priority 1: Add the first target dataset

Set up one target dataset first, preferably IDRiD or MESSIDOR2.

Need files like:

```text
data/idrid/labels.csv
data/idrid/images/
```

or:

```text
data/messidor2/labels.csv
data/messidor2/images/
```

Required CSV columns:

```text
image_id,grade
```

The grade must be parseable as DR grade 0-4 or compatible labels handled in
`src/recalib_eye/data.py`.

### Priority 2: Extend linear probe config to source + target

Create a config similar to:

```json
{
  "eyeclip_repo": "/Users/lewjanice/Documents/EyeCLIP",
  "checkpoint": "/Users/lewjanice/Documents/EyeCLIP/eyeclip_visual.pt",
  "device": "auto",
  "batch_size": 8,
  "num_workers": 0,
  "output_dir": "outputs/aptos_source_to_idrid_linear_probe",
  "source_roles": ["source"],
  "target_specificity": 0.9,
  "threshold_metric": "balanced_accuracy",
  "fit_probability_calibrator": true,
  "cv_folds": 5,
  "class_weight": "balanced",
  "C": 1.0,
  "feature_cache": true,
  "reuse_feature_cache": true,
  "datasets": [
    {
      "name": "APTOS2019",
      "role": "source",
      "csv_path": "data/aptos2019/train.csv",
      "image_dir": "data/aptos2019/train_images",
      "id_col": "id_code",
      "grade_col": "diagnosis",
      "preferred_ext": ".png"
    },
    {
      "name": "IDRiD",
      "role": "target",
      "csv_path": "data/idrid/labels.csv",
      "image_dir": "data/idrid/images",
      "id_col": "image_id",
      "grade_col": "grade",
      "preferred_ext": ".jpg"
    }
  ]
}
```

Run:

```bash
PYTHONPATH=src /Users/lewjanice/miniconda3/envs/recalib-eye/bin/python -m recalib_eye.linear_probe_dr --config configs/<new_source_to_target_config>.json
```

Expected behavior:

- APTOS source gets 5-fold out-of-fold metrics.
- Final linear probe trains on all APTOS source features.
- Target gets evaluated using that source-trained probe.
- Source calibration threshold/temperature/bias are applied to target.
- No target calibration or target tuning should happen.

### Priority 3: Summarise source-to-target results

For every target result, record:

```text
AUROC
AUPR
macro-F1
balanced accuracy
sensitivity at specificity 0.90
ECE
Brier
threshold
temperature
bias
```

Create a tracked result folder like:

```text
results/aptos_to_idrid_linear_probe/
```

Do not commit huge prediction CSVs unless needed. Metrics and README are enough.

## Rules For Cursor

1. Do not replace the strong linear-probe direction with prompt-only work.
2. Do not tune anything on target labels.
3. Keep APTOS as source.
4. Use target datasets only for final evaluation.
5. Keep generated `outputs/` files local unless a small summary metric file is
   worth tracking under `results/`.
6. If adding target support, follow existing dataset config style instead of
   inventing a new data pipeline.
7. Before changing model logic, compare against:

```text
results/aptos2019_linear_probe/metrics_calibrated.csv
```

Current target: beat or explain cross-dataset performance relative to the strong
APTOS source baseline.
