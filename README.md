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

## Repository Structure

```text
configs/        Example experiment configuration files
data/           Local datasets, ignored by Git
scripts/        Small runnable helper scripts
src/            ReCalib-Eye Python package
outputs/        Local experiment outputs, ignored by Git
```

## Results

To be updated after dataset evaluation.

## Citation

To be added.
