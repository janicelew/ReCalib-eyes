# MESSIDOR2 Linear Probe Baseline

This folder stores the EyeCLIP source-only linear probe baseline on MESSIDOR2,
run standalone on Kaggle (same setup as `results/messidor2_zeroshot/`, separate
from the local `recalib_eye` package pipeline used for APTOS2019/IDRiD).

Input data (Kaggle paths, see `configs/messidor2_linear_probe.yaml`):

```text
/kaggle/input/datasets/nadaol0/messidor2/messidor_2.csv
/kaggle/input/datasets/nadaol0/messidor2/images
```

Run command:

```bash
python scripts/eval_messidor2_linear_probe.py
```

The script loads the EyeCLIP checkpoint over a CLIP ViT-B/32 backbone, extracts
frozen image features for every MESSIDOR2 image, and fits a `StandardScaler` +
`LogisticRegression` linear probe with 5-fold stratified out-of-fold (OOF)
scoring, separately for two label formulations:

- `any_dr`: grade 0 (normal) vs grades 1-3 (any DR).
- `rdr`: referable DR split derived from `true_grade` (grade >= 2).

No cross-dataset calibration has been applied to these numbers; this is the
source-only linear probe result on MESSIDOR2 itself.

## Files

| File | Description |
|------|-------------|
| `messidor2_linear_probe.yaml` | Config used for the run (see `configs/`) |
| `predictions_MESSIDOR2.csv` | Per-image predictions: `image_id, true_grade, true_any_dr, prob_any_dr_linear_probe, true_rdr, prob_rdr_linear_probe` |
| `metrics.csv` | Metrics below, in CSV form |
| `messidor2_linear_probe_metrics.txt` | Raw metrics output from the Kaggle run |

## Result summary

```text
any_dr_auroc    = 0.7235
any_dr_aupr     = 0.6315
any_dr_f1       = 0.6273
any_dr_accuracy = 0.6669
rdr_auroc       = 0.7388
rdr_aupr        = 0.5140
rdr_f1          = 0.5193
rdr_accuracy    = 0.6858
```

The linear probe clearly improves over the prompt-only zero-shot baseline on
MESSIDOR2 (any-DR AUROC 0.7235 vs 0.6538; referable-DR AUROC 0.7388 vs 0.6149),
consistent with the pattern already observed on APTOS2019 where prompt-only
zero-shot is weak and a source-only linear probe on EyeCLIP features is much
stronger.
