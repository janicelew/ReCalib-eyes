from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_ece(probs, labels, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())

    return float(ece)


def sensitivity_at_specificity(labels, probs, target_specificity: float = 0.90) -> float:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(probs))[::-1], -np.inf]
    best_sensitivity = np.nan

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()

        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        if np.isfinite(specificity) and specificity >= target_specificity:
            if np.isnan(best_sensitivity) or sensitivity > best_sensitivity:
                best_sensitivity = sensitivity

    return float(best_sensitivity)


def safe_binary_metric(fn, labels, probs) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(fn(labels, probs))


def binary_classification_report(
    labels,
    probs,
    threshold: float = 0.5,
    target_specificity: float = 0.90,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    return {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": safe_binary_metric(roc_auc_score, labels, probs),
        "aupr": safe_binary_metric(average_precision_score, labels, probs),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "sensitivity_at_specificity_0_90": sensitivity_at_specificity(labels, probs, target_specificity),
        "ece": compute_ece(probs, labels),
        "brier": float(brier_score_loss(labels, probs)),
    }
