# APTOS2019 Source Linear Probe

This folder records the full APTOS2019 source-only linear probe result using
EyeCLIP image features.

Input data:

```text
data/aptos2019/train.csv
data/aptos2019/train_images/
```

Run command:

```bash
PYTHONPATH=src python -m recalib_eye.linear_probe_dr --config configs/aptos2019_linear_probe.json
```

Evaluation protocol:

- APTOS2019 is used as the source dataset.
- Source performance is measured with 5-fold out-of-fold predictions.
- Calibration is fitted only from source out-of-fold predictions.
- No target dataset labels are used.

Raw source result:

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

This is the strongest current APTOS2019 source result. Prompt-only zero-shot and
image prototype variants should be treated as weaker baselines.
