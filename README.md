# ReCalib-eyes
Source-Calibrated Prototype Adaptation of Ophthalmic Visual-Language Foundation Models under Cross-Dataset Shift

## Overview

ReCalib-Eye is a lightweight adaptation framework for ophthalmic vision-language foundation models. The project investigates how released EyeCLIP checkpoints can be adapted to new ophthalmic datasets using clinical text prototypes, image prototypes, and source-domain calibration without expensive retraining.

## Research Motivation

Although EyeCLIP demonstrates strong zero-shot and few-shot performance, reliable adaptation under cross-dataset distribution shift remains underexplored. This project proposes a source-calibrated prototype fusion strategy to improve both classification performance and probability calibration.

## Research Questions

* Can EyeCLIP zero-shot performance be reproduced on public ophthalmic datasets?
* Do clinical text prototypes improve robustness over simple class-name prompts?
* Can prototype fusion outperform text-only and image-only baselines?
* Can source-only calibration improve reliability under dataset shift?

## Datasets

* APTOS2019
* IDRiD
* MESSIDOR2
* PAPILA
* Glaucoma Fundus

## Baselines

1. EyeCLIP Zero-shot
2. Clinical Prompt Ensemble
3. Image Prototype
4. Fixed-alpha Fusion
5. Linear Probe
6. MLP Head

## Proposed Method

ReCalib-Eye introduces reliability-guided fusion between text prototypes and image prototypes, followed by source-only temperature scaling.

## Repository Structure

data/
src/
notebooks/
experiments/
results/
docs/

## Results

(To be updated)

## Citation

(To be added)
