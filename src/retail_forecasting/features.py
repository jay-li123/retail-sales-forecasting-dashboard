"""Transparent, leakage-safe features for one-day-ahead demand forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.validation import require


HISTORY_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]

MODEL_FEATURES = [
    "item_id",
    "weekday",
    "month",
    "snap",
    "event_indicator",
    "sell_price",
    *HISTORY_FEATURES,
]

REPORTING_METADATA = ["date", "period", "volume_segment"]
TARGET = "units_sold"


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return sorted rows with history features built independently per item.

    Every rolling statistic first shifts sales by one day, so the target day's
    value can never enter its own predictors. The 28-day standard deviation is
    the population standard deviation (``ddof=0``).
    """

    required = {
        "item_id",
        "date",
        "units_sold",
        "weekday",
        "month",
        "snap",
        "event_indicator",
        "sell_price",
        "period",
        "volume_segment",
    }
    missing = required.difference(panel.columns)
    require(not missing, f"Feature input is missing columns: {sorted(missing)}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["item_id", "date"], kind="stable").reset_index(drop=True)
    require(not frame.duplicated(["item_id", "date"]).any(), "Feature input has duplicate item/date rows")

    grouped = frame.groupby("item_id", sort=False, observed=True)["units_sold"]
    frame["lag_7"] = grouped.shift(7)
    frame["lag_14"] = grouped.shift(14)
    frame["lag_28"] = grouped.shift(28)

    shifted = grouped.shift(1)
    shifted_by_item = shifted.groupby(frame["item_id"], sort=False, observed=True)
    frame["rolling_mean_7"] = shifted_by_item.transform(
        lambda values: values.rolling(7, min_periods=7).mean()
    )
    frame["rolling_mean_28"] = shifted_by_item.transform(
        lambda values: values.rolling(28, min_periods=28).mean()
    )
    frame["rolling_std_28"] = shifted_by_item.transform(
        lambda values: values.rolling(28, min_periods=28).std(ddof=0)
    )
    return frame


def eligible_history_rows(features: pd.DataFrame) -> pd.Series:
    """Rows with all six history predictors available; price may remain NaN."""

    require(set(HISTORY_FEATURES).issubset(features.columns), "History features have not been built")
    return features[HISTORY_FEATURES].notna().all(axis=1)


def validate_feature_values(features: pd.DataFrame) -> None:
    values = features[HISTORY_FEATURES].to_numpy(dtype=float)
    finite_or_nan = np.isfinite(values) | np.isnan(values)
    require(finite_or_nan.all(), "History features contain infinite values")
    require(
        features[HISTORY_FEATURES].dropna().ge(0).all().all(),
        "History features contain negative values",
    )
    price = features["sell_price"].to_numpy(dtype=float)
    require((np.isfinite(price) | np.isnan(price)).all(), "sell_price contains infinite values")
