# MESSIDOR2 Zero-Shot Baseline

This folder stores the first EyeCLIP zero-shot diabetic retinopathy baseline on
MESSIDOR2, run standalone on Kaggle (separate from the local `recalib_eye`
package pipeline used for APTOS2019/IDRiD).

Input data (Kaggle paths, see `configs/messidor2_zeroshot.yaml`):

```text
/kaggle/input/datasets/nadaol0/messidor2/messidor_2.csv
/kaggle/input/datasets/nadaol0/messidor2/images
```

Run command:

```bash
python scripts/eval_messidor2_zeroshot.py
```

The script loads the EyeCLIP checkpoint over a CLIP ViT-B/32 backbone, scores
each image against the prompts `"normal retina"` / `"diabetic retinopathy"`,
and writes predictions plus binary metrics for two label formulations:

- `any_dr`: grade 0 (normal) vs grades 1-3 (any DR).
- `rdr`: referable DR split derived from `true_grade`.

## Files

| File | Description |
|------|-------------|
| `messidor2_zeroshot.yaml` | Config used for the run (see `configs/`) |
| `predictions_MESSIDOR2.csv` | Per-image predictions: `image_id, true_grade, prob_normal, prob_dr, true_any_dr` |
| `metrics.csv` | Metrics below, in CSV form |
| `messidor2_metrics.txt` | Raw metrics output from the Kaggle run |

## Result summary

```text
any_dr_auroc    = 0.6538
any_dr_aupr     = 0.5519
any_dr_f1       = 0.5884
any_dr_accuracy = 0.4169
rdr_auroc       = 0.6149
rdr_aupr        = 0.3392
```

Prompt-only zero-shot on MESSIDOR2 is a weak-to-moderate baseline (AUROC ~0.65
for any-DR, ~0.61 for referable DR), consistent with the weak prompt-only
zero-shot baselines already observed on APTOS2019. No source calibration has
been applied to these numbers yet.
