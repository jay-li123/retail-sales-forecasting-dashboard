"""Fixed global LightGBM model used by the weekly walk-forward evaluation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecasting.features import MODEL_FEATURES, TARGET
from retail_forecasting.validation import require


CATEGORICAL_FEATURES = ["item_id", "weekday", "month"]
WEEKDAY_CATEGORIES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

LIGHTGBM_PARAMETERS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 31,
    "min_child_samples": 50,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def deterministic_category_mapping(item_ids) -> dict[str, list]:
    """Use known identifiers/calendar levels, never target-derived encodings."""

    return {
        "item_id": sorted(pd.Series(item_ids).astype(str).unique().tolist()),
        "weekday": WEEKDAY_CATEGORIES,
        "month": list(range(1, 13)),
    }


def predictor_matrix(frame: pd.DataFrame, mapping: dict[str, list]) -> pd.DataFrame:
    require(list(mapping) == CATEGORICAL_FEATURES, "Unexpected categorical mapping keys/order")
    matrix = frame.loc[:, MODEL_FEATURES].copy()
    matrix["item_id"] = pd.Categorical(
        matrix["item_id"].astype(str), categories=mapping["item_id"], ordered=False
    )
    matrix["weekday"] = pd.Categorical(
        matrix["weekday"].astype(str), categories=mapping["weekday"], ordered=False
    )
    matrix["month"] = pd.Categorical(
        matrix["month"].astype(int), categories=mapping["month"], ordered=False
    )
    require(matrix.columns.tolist() == MODEL_FEATURES, "Model feature order changed")
    require(not matrix[CATEGORICAL_FEATURES].isna().any().any(), "Unknown categorical value")
    return matrix


def fit_global_model(training: pd.DataFrame, mapping: dict[str, list]) -> lgb.Booster:
    require(training["date"].notna().all(), "Training dates are missing")
    require(np.isfinite(training[TARGET].to_numpy(dtype=float)).all(), "Training target is non-finite")
    parameters = dict(LIGHTGBM_PARAMETERS)
    rounds = int(parameters.pop("n_estimators"))
    dataset = lgb.Dataset(
        predictor_matrix(training, mapping),
        label=training[TARGET].to_numpy(dtype=float),
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    model = lgb.train(parameters, dataset, num_boost_round=rounds)
    require(model.feature_name() == MODEL_FEATURES, "LightGBM received the wrong feature list")
    return model


def predict_nonnegative(
    model: lgb.Booster,
    features: pd.DataFrame,
    mapping: dict[str, list],
) -> tuple[np.ndarray, int]:
    raw = model.predict(predictor_matrix(features, mapping))
    require(np.isfinite(raw).all(), "LightGBM produced non-finite predictions")
    clipped_count = int((raw < 0).sum())
    return np.maximum(raw, 0.0), clipped_count
