# APTOS2019 Zero-Shot Baseline

This folder stores the first EyeCLIP zero-shot diabetic retinopathy baseline on
APTOS2019.

Input data:

```text
data/aptos2019/train.csv
data/aptos2019/train_images/
```

Run command:

```bash
PYTHONPATH=src python -m recalib_eye.zeroshot_dr --config configs/aptos2019_zeroshot.json
```

Result summary:

```text
n = 3662
positive_rate = 0.4061
AUROC = 0.4856
AUPR = 0.4038
macro-F1 = 0.3879
balanced accuracy = 0.4950
ECE = 0.0838
Brier = 0.2501
```

This is a first zero-shot baseline only. It does not use few-shot prototypes or
source-only calibration yet.
