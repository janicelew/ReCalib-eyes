from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EPS = 1e-6


@dataclass
class SourceCalibration:
    source_dataset: str
    n_source: int
    positive_rate: float
    temperature: float
    bias: float
    threshold: float
    threshold_metric: str
    source_metric_value: float
    nll_before: float
    nll_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clip_probs(probs) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)


def logit(probs) -> np.ndarray:
    probs = clip_probs(probs)
    return np.log(probs / (1.0 - probs))


def sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def apply_probability_calibration(probs, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return sigmoid(logit(probs) / float(temperature) + float(bias))


def binary_nll(labels, probs) -> float:
    labels = np.asarray(labels, dtype=int)
    probs = clip_probs(probs)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1.0 - probs)).mean())


def confusion_counts(labels, probs, threshold: float) -> tuple[int, int, int, int]:
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(probs, dtype=float) >= threshold
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tn, fp, fn, tp


def threshold_metric_value(labels, probs, threshold: float, metric: str) -> float:
    tn, fp, fn, tp = confusion_counts(labels, probs, threshold)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1_pos = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    f1_neg = 2 * npv * specificity / (npv + specificity) if (npv + specificity) else 0.0

    if metric == "balanced_accuracy":
        return float((sensitivity + specificity) / 2.0)
    if metric == "macro_f1":
        return float((f1_pos + f1_neg) / 2.0)
    if metric == "youden_j":
        return float(sensitivity + specificity - 1.0)
    if metric == "f1":
        return float(f1_pos)
    raise ValueError(f"Unsupported threshold metric: {metric}")


def candidate_thresholds(probs) -> np.ndarray:
    unique_probs = np.unique(np.asarray(probs, dtype=float))
    if len(unique_probs) == 0:
        return np.asarray([0.5], dtype=float)
    if len(unique_probs) == 1:
        return np.asarray([unique_probs[0]], dtype=float)
    midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    return np.r_[unique_probs[0] - EPS, midpoints, unique_probs[-1] + EPS]


def find_best_threshold(labels, probs, metric: str = "balanced_accuracy") -> tuple[float, float]:
    thresholds = candidate_thresholds(probs)
    scored = [
        (threshold_metric_value(labels, probs, float(threshold), metric), -abs(float(threshold) - 0.5), float(threshold))
        for threshold in thresholds
    ]
    best_metric, _, best_threshold = max(scored)
    return best_threshold, best_metric


def find_threshold_at_specificity(labels, probs, target_specificity: float = 0.90) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    best_threshold = 1.0 + EPS
    best_sensitivity = -1.0
    for threshold in candidate_thresholds(probs):
        tn, fp, fn, tp = confusion_counts(labels, probs, float(threshold))
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        if specificity >= target_specificity and sensitivity > best_sensitivity:
            best_threshold = float(threshold)
            best_sensitivity = float(sensitivity)
    return best_threshold, best_sensitivity


def fit_platt_grid(labels, probs) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=int)
    raw_logits = logit(probs)
    temperatures = np.geomspace(0.05, 20.0, 80)
    biases = np.linspace(-5.0, 5.0, 201)

    best_nll = float("inf")
    best_temperature = 1.0
    best_bias = 0.0
    for temperature in temperatures:
        scaled_logits = raw_logits / temperature
        calibrated = sigmoid(scaled_logits[:, None] + biases[None, :])
        nlls = -(labels[:, None] * np.log(clip_probs(calibrated)) + (1 - labels[:, None]) * np.log(clip_probs(1.0 - calibrated))).mean(axis=0)
        idx = int(np.argmin(nlls))
        if float(nlls[idx]) < best_nll:
            best_nll = float(nlls[idx])
            best_temperature = float(temperature)
            best_bias = float(biases[idx])
    return best_temperature, best_bias, best_nll


def fit_source_calibration(
    labels,
    probs,
    source_dataset: str,
    threshold_metric: str = "balanced_accuracy",
    fit_probability_calibrator: bool = True,
) -> SourceCalibration:
    labels = np.asarray(labels, dtype=int)
    probs = clip_probs(probs)
    if len(labels) == 0:
        raise ValueError("Cannot fit source calibration with no source samples.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Source calibration needs both positive and negative labels.")

    nll_before = binary_nll(labels, probs)
    if fit_probability_calibrator:
        temperature, bias, nll_after = fit_platt_grid(labels, probs)
        calibrated_probs = apply_probability_calibration(probs, temperature=temperature, bias=bias)
    else:
        temperature, bias = 1.0, 0.0
        calibrated_probs = probs
        nll_after = nll_before

    threshold, metric_value = find_best_threshold(labels, calibrated_probs, metric=threshold_metric)
    return SourceCalibration(
        source_dataset=source_dataset,
        n_source=int(len(labels)),
        positive_rate=float(labels.mean()),
        temperature=float(temperature),
        bias=float(bias),
        threshold=float(threshold),
        threshold_metric=threshold_metric,
        source_metric_value=float(metric_value),
        nll_before=float(nll_before),
        nll_after=float(nll_after),
    )
