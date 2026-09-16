"""Cached, read-only loading and reconciliation for dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUT = ROOT / "outputs" / "model"
REQUIRED_ARTIFACTS = {
    "predictions": MODEL_OUTPUT / "final_test_predictions.parquet",
    "metrics": MODEL_OUTPUT / "final_test_metrics.csv",
    "products": MODEL_OUTPUT / "product_error_summary.csv",
    "representative": MODEL_OUTPUT / "representative_errors.csv",
    "importance": MODEL_OUTPUT / "feature_importance.csv",
    "errors": MODEL_OUTPUT / "error_analysis.csv",
    "summary": MODEL_OUTPUT / "summary.json",
}


def missing_artifacts() -> list[Path]:
    return [path for path in REQUIRED_ARTIFACTS.values() if not path.is_file()]


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict:
    """Load only compact, previously generated model outputs."""

    missing = missing_artifacts()
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing dashboard artifacts: {names}")

    predictions = pd.read_parquet(REQUIRED_ARTIFACTS["predictions"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    return {
        "predictions": predictions,
        "metrics": pd.read_csv(REQUIRED_ARTIFACTS["metrics"]),
        "products": pd.read_csv(REQUIRED_ARTIFACTS["products"]),
        "representative": pd.read_csv(REQUIRED_ARTIFACTS["representative"], parse_dates=["date"]),
        "importance": pd.read_csv(REQUIRED_ARTIFACTS["importance"]),
        "errors": pd.read_csv(REQUIRED_ARTIFACTS["errors"]),
        "summary": json.loads(REQUIRED_ARTIFACTS["summary"].read_text(encoding="utf-8")),
    }


def model_predictions(data: dict, model: str = "lightgbm") -> pd.DataFrame:
    return data["predictions"].loc[data["predictions"]["model"].eq(model)].copy()


def overall_metric(data: dict, model: str) -> pd.Series:
    metrics = data["metrics"]
    return metrics.loc[
        metrics["model"].eq(model)
        & metrics["breakdown"].eq("overall")
        & metrics["group"].eq("all")
    ].iloc[0]


def product_detail(data: dict, item_id: str) -> tuple[pd.Series, pd.DataFrame]:
    summary = data["products"].loc[data["products"]["item_id"].eq(item_id)].iloc[0]
    predictions = data["predictions"].loc[data["predictions"]["item_id"].eq(item_id)].copy()
    return summary, predictions


def recompute_product_summary(lightgbm_predictions: pd.DataFrame) -> pd.DataFrame:
    """Reconcile displayed product totals from saved predictions only."""

    rows = []
    total_error = float(lightgbm_predictions["absolute_error"].sum())
    for (item_id, segment), frame in lightgbm_predictions.groupby(
        ["item_id", "volume_segment"], sort=True
    ):
        actual_total = float(frame["actual"].sum())
        error = frame["prediction"] - frame["actual"]
        rows.append(
            {
                "item_id": item_id,
                "volume_segment": segment,
                "actual_total": actual_total,
                "prediction_total": float(frame["prediction"].sum()),
                "MAE": float(error.abs().mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "WAPE": float(error.abs().sum() / actual_total) if actual_total else np.nan,
                "absolute_error_share": float(error.abs().sum() / total_error),
            }
        )
    return pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
