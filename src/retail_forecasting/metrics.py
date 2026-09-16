"""Forecast metrics shared by the baseline and future LightGBM evaluation."""

from __future__ import annotations

import numpy as np


def mae(actual, prediction) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    return float(np.mean(np.abs(actual_array - prediction_array)))


def rmse(actual, prediction) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(actual_array - prediction_array))))


def wape(actual, prediction) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    denominator = float(actual_array.sum())
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual_array - prediction_array).sum() / denominator)


def metric_row(actual, prediction) -> dict[str, float | int]:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if actual_array.shape != prediction_array.shape or actual_array.size == 0:
        raise ValueError("Metric inputs must be non-empty arrays with matching shapes")
    if not (np.isfinite(actual_array).all() and np.isfinite(prediction_array).all()):
        raise ValueError("Metric inputs must contain only finite values")
    return {
        "n": int(actual_array.size),
        "MAE": mae(actual_array, prediction_array),
        "RMSE": rmse(actual_array, prediction_array),
        "WAPE": wape(actual_array, prediction_array),
        "actual_total": float(actual_array.sum()),
    }
