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

This is the raw first zero-shot baseline. A source-only calibration pass was
then fitted from these saved APTOS2019 predictions and written locally to:

```text
outputs/aptos2019_source_calibration/
```

That calibration improves thresholded source metrics and calibration error, but
does not fix the weak ranking signal: AUROC remains `0.4856`. The next
experiment should compare prompt/scoring variants before target-dataset testing.

Follow-up checks showed that prompt-only and image-prototype variants are weak.
The full APTOS2019 source-only linear probe is now the main source baseline,
reaching AUROC `0.9360` and calibrated balanced accuracy `0.8672` with 5-fold
source cross-validation.
