"""Small, explicit validation helpers for the three M5 source tables."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


SALES_METADATA = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_REQUIRED = {
    "date",
    "d",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
}
PRICE_REQUIRED = {"store_id", "item_id", "wm_yr_wk", "sell_price"}


def require(condition: bool, message: str) -> None:
    """Raise a clear structural-data error instead of silently repairing data."""

    if not condition:
        raise ValueError(message)


def validate_calendar_frame(calendar: pd.DataFrame) -> pd.DataFrame:
    missing = CALENDAR_REQUIRED.difference(calendar.columns)
    require(not missing, f"calendar.csv is missing required columns: {sorted(missing)}")
    result = calendar.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    require(not result["d"].duplicated().any(), "calendar.csv contains duplicate d values")
    require(not result["date"].duplicated().any(), "calendar.csv contains duplicate dates")
    require(result["date"].is_monotonic_increasing, "calendar.csv dates are not increasing")
    require(
        result["date"].diff().dropna().eq(pd.Timedelta(days=1)).all(),
        "calendar.csv dates are not continuous daily dates",
    )
    expected_d = [f"d_{i}" for i in range(1, len(result) + 1)]
    require(result["d"].tolist() == expected_d, "calendar.csv d values do not match date order")
    required_nonnull = [
        "date",
        "d",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "snap_CA",
    ]
    require(not result[required_nonnull].isna().any().any(), "calendar.csv has missing required values")
    require(result["month"].eq(result["date"].dt.month).all(), "calendar.csv month/date mismatch")
    require(result["year"].eq(result["date"].dt.year).all(), "calendar.csv year/date mismatch")
    require(result["weekday"].eq(result["date"].dt.day_name()).all(), "calendar.csv weekday/date mismatch")
    require(result["snap_CA"].isin([0, 1]).all(), "calendar.csv snap_CA must contain only 0/1")
    return result


def validate_sales_header(columns: list[str], calendar_days: set[str]) -> list[str]:
    missing = [column for column in SALES_METADATA if column not in columns]
    require(not missing, f"sales_train_evaluation.csv is missing metadata columns: {missing}")
    day_columns = [column for column in columns if re.fullmatch(r"d_\d+", column)]
    require(columns == SALES_METADATA + day_columns, "sales_train_evaluation.csv has unexpected columns/order")
    require(bool(day_columns), "sales_train_evaluation.csv has no d_* columns")
    expected = [f"d_{i}" for i in range(1, len(day_columns) + 1)]
    require(day_columns == expected, "sales_train_evaluation.csv d_* columns are not consecutive")
    require(set(day_columns).issubset(calendar_days), "sales day columns are missing from calendar.csv")
    return day_columns


def validate_sales_values(frame: pd.DataFrame, day_columns: list[str]) -> None:
    values = frame[day_columns].to_numpy()
    require(np.issubdtype(values.dtype, np.number), "sales values are not numeric")
    require(np.isfinite(values).all(), "sales values contain missing or non-finite values")
    require((values >= 0).all(), "sales values contain negative values")
    require(np.equal(values, np.floor(values)).all(), "sales values contain non-integer values")
    require(not frame[SALES_METADATA].isna().any().any(), "sales metadata contains missing values")


def validate_price_frame(prices: pd.DataFrame) -> None:
    missing = PRICE_REQUIRED.difference(prices.columns)
    require(not missing, f"sell_prices.csv is missing required columns: {sorted(missing)}")
    keys = ["store_id", "item_id", "wm_yr_wk"]
    require(not prices[keys].isna().any().any(), "sell_prices.csv has missing key values")
    require(not prices.duplicated(keys).any(), "sell_prices.csv has duplicate store/item/week keys")
    present = prices["sell_price"].notna()
    require(
        np.isfinite(prices.loc[present, "sell_price"]).all()
        and prices.loc[present, "sell_price"].gt(0).all(),
        "sell_prices.csv contains a non-positive or non-finite price",
    )


def validate_processed_panel(panel: pd.DataFrame, product_count: int) -> None:
    require(panel["item_id"].nunique() == product_count, f"panel must contain exactly {product_count} products")
    require(not panel.duplicated(["item_id", "date"]).any(), "panel has duplicate item/date rows")
    require(panel["store_id"].nunique() == 1, "panel must contain exactly one store")
    require(panel["dept_id"].nunique() == 1, "panel must contain exactly one department")
    require(panel["units_sold"].ge(0).all(), "panel contains negative units_sold")
    require(
        np.equal(panel["units_sold"], np.floor(panel["units_sold"])).all(),
        "panel units_sold must be integer-valued",
    )
    calendar_fields = ["date", "d", "wm_yr_wk", "weekday", "wday", "month", "year", "snap"]
    require(not panel[calendar_fields].isna().any().any(), "panel has incomplete calendar joins")
    expected_dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    per_item = panel.groupby("item_id", observed=True)["date"]
    require(per_item.size().eq(len(expected_dates)).all(), "panel products have unequal date counts")
    require(per_item.nunique().eq(len(expected_dates)).all(), "panel products do not have continuous dates")
