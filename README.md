# ReCalib-Eye

Source-Calibrated Prototype Adaptation of Ophthalmic Visual-Language Foundation Models under Cross-Dataset Shift.

ReCalib-Eye is a local reproduction and extension project for the EyeCLIP paper:
*A multimodal visual-language foundation model for computational ophthalmology*.

This repository is the experiment workspace. It does not duplicate the EyeCLIP
source code. The local EyeCLIP clone is expected at:

```text
/Users/lewjanice/Documents/EyeCLIP
```

The local EyeCLIP checkpoint is expected at:

```text
/Users/lewjanice/Documents/EyeCLIP/eyeclip_visual.pt
```

## Overview

ReCalib-Eye is a lightweight adaptation framework for ophthalmic vision-language
foundation models. The project investigates how released EyeCLIP checkpoints can
be adapted to new ophthalmic datasets using clinical text prototypes, image
prototypes, and source-domain calibration without expensive retraining.

The practical reproduction target is downstream evaluation, not full EyeCLIP
pretraining. The original model was pretrained on 2.77 million ophthalmic images,
which is not realistic to reproduce on this machine.

## Research Motivation

Although EyeCLIP demonstrates strong zero-shot and few-shot performance, reliable
adaptation under cross-dataset distribution shift remains underexplored. This
project proposes a source-calibrated prototype fusion strategy to improve both
classification performance and probability calibration.

## Research Questions

- Can EyeCLIP zero-shot performance be reproduced on public ophthalmic datasets?
- Do clinical text prototypes improve robustness over simple class-name prompts?
- Can prototype fusion outperform text-only and image-only baselines?
- Can source-only calibration improve reliability under dataset shift?

## Current Scope

This project currently focuses on:

- EyeCLIP zero-shot diabetic retinopathy classification.
- Harmonised referable vs non-referable DR labels.
- Few-shot text/image prototype adaptation.
- Source-only calibration experiments for ReCalib-Eye.

## Datasets

Planned datasets:

- APTOS2019
- IDRiD
- MESSIDOR2
- PAPILA
- Glaucoma Fundus

Start with APTOS2019 first. Do not try to set up every dataset at once.

## Baselines

1. EyeCLIP zero-shot
2. Clinical prompt ensemble
3. Image prototype
4. Fixed-alpha fusion
5. Linear probe
6. MLP head

## Environment

Use the conda environment already created for this project:

```bash
conda activate recalib-eye
pip install -r requirements.txt
```

The current machine has CPU-only PyTorch in this environment:

```text
CUDA available: False
MPS available: False
```

So local smoke tests are expected to run on CPU. Full dataset experiments will be
slow locally unless moved to a GPU machine or Kaggle.

## Smoke Test

After installing dependencies, verify that this project can import and load the
local EyeCLIP checkpoint:

```bash
PYTHONPATH=src python scripts/smoke_eyeclip.py
```

Optionally test one retinal image:

```bash
PYTHONPATH=src python scripts/smoke_eyeclip.py --image data/sample_retina.jpg
```

## Zero-Shot DR Evaluation

Edit the dataset paths in:

```text
configs/dr_zeroshot.example.json
```

Then run:

```bash
PYTHONPATH=src python -m recalib_eye.zeroshot_dr --config configs/dr_zeroshot.example.json
```

Expected dataset label mapping:

- DR grade 0-1: non-referable.
- DR grade 2-4: referable.

Target datasets must only be used for final evaluation. Do not tune prompts,
fusion weights, thresholds, or temperature on target datasets.

### Source Calibration

APTOS2019 is the current source dataset. Calibration settings are learned only
from datasets marked with:

```json
"role": "source"
```

When `source_calibration.enabled` is true, the evaluation writes:

```text
source_calibration.json
metrics_calibrated.csv
predictions_<dataset>.csv
```

The same source threshold, temperature, and bias should then be applied to target
datasets such as IDRiD or MESSIDOR2 without refitting on those target labels.

To calibrate from an existing APTOS prediction CSV without rerunning EyeCLIP:

```bash
PYTHONPATH=src python scripts/calibrate_source_predictions.py \
  --predictions results/aptos2019_zeroshot/predictions_APTOS2019.csv \
  --source-name APTOS2019 \
  --output-dir outputs/aptos2019_source_calibration
```

To test a direct binary referable/non-referable prompt setup:

```bash
PYTHONPATH=src python -m recalib_eye.zeroshot_dr --config configs/aptos2019_binary_zeroshot.json
```

If prompt-only scoring is weak, switch to source image prototypes:

```bash
PYTHONPATH=src python -m recalib_eye.image_prototype_dr --config configs/aptos2019_image_prototypes.json
```

For APTOS2019 source evaluation, this uses leave-one-out scoring so each image
is not included in its own class prototype.

On CPU, use the 256-image source subset first:

```bash
PYTHONPATH=src python -m recalib_eye.image_prototype_dr --config configs/aptos2019_image_prototypes_256.json
```

If image prototypes are still weak, test whether EyeCLIP image features are
usable with a source-only linear probe:

```bash
PYTHONPATH=src python -m recalib_eye.linear_probe_dr --config configs/aptos2019_linear_probe_256.json
```

For the full APTOS2019 source run:

```bash
PYTHONPATH=src python -m recalib_eye.linear_probe_dr --config configs/aptos2019_linear_probe.json
```

## Repository Structure

```text
configs/        Example experiment configuration files
data/           Local datasets, ignored by Git
scripts/        Small runnable helper scripts
src/            ReCalib-Eye Python package
outputs/        Local experiment outputs, ignored by Git
```

## Results

Current strongest APTOS2019 source result:

- Method: EyeCLIP image features + source-only linear probe.
- Protocol: 5-fold source out-of-fold evaluation on APTOS2019.
- AUROC: `0.9360`.
- AUPR: `0.9009`.
- Calibrated balanced accuracy: `0.8672`.
- Calibrated ECE: `0.0195`.

Prompt-only zero-shot remains a weak baseline on APTOS2019.

## Citation

To be added.
