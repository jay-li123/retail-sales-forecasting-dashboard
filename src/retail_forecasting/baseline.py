"""Create Milestone 3 features and evaluate seasonal naive on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.constants import ROOT, TEST_START, VALIDATION_START
from retail_forecasting.features import (
    HISTORY_FEATURES,
    MODEL_FEATURES,
    REPORTING_METADATA,
    TARGET,
    build_features,
    eligible_history_rows,
    validate_feature_values,
)
from retail_forecasting.metrics import metric_row
from retail_forecasting.validation import require


def metric_breakdowns(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(breakdown: str, group: str, data: pd.DataFrame) -> None:
        rows.append(
            {
                "breakdown": breakdown,
                "group": str(group),
                **metric_row(data["actual"], data["prediction"]),
            }
        )

    add("overall", "all", predictions)
    for column in ("validation_week", "volume_segment", "weekday", "demand_status"):
        for label, data in predictions.groupby(column, sort=True, observed=True):
            add(column, label, data)
    return pd.DataFrame(rows)


def json_safe_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                record[key] = None
    return records


def feature_sanity(features: pd.DataFrame, model_ready: pd.DataFrame) -> dict:
    summaries = {}
    for name in [*HISTORY_FEATURES, "sell_price"]:
        values = features[name]
        finite = values.dropna()
        summaries[name] = {
            "rows": int(len(values)),
            "non_missing": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "minimum": float(finite.min()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "infinite": int(np.isinf(finite.to_numpy(dtype=float)).sum()),
        }
    validation = model_ready.loc[model_ready["period"].eq("validation")]
    return {
        "features": summaries,
        "validation_history_missing": {
            name: int(validation[name].isna().sum()) for name in HISTORY_FEATURES
        },
        "validation_sell_price_missing": int(validation["sell_price"].isna().sum()),
        "model_ready_sell_price_missing": int(model_ready["sell_price"].isna().sum()),
    }


def create_figures(predictions: pd.DataFrame, metrics: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    daily = predictions.groupby("date", as_index=False)[["actual", "prediction"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily["date"], daily["actual"], label="Actual", color="#111827", linewidth=1.8)
    ax.plot(daily["date"], daily["prediction"], label="Seasonal naive", color="#2563eb", linewidth=1.5)
    ax.set(title="Validation daily sales: actual vs seasonal naive", xlabel="Date", ylabel="Units sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "01_validation_actual_vs_seasonal_naive.png", dpi=160)
    plt.close(fig)

    weekly = metrics.loc[metrics["breakdown"].eq("validation_week")].copy()
    weekly["week"] = weekly["group"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(weekly["week"].astype(str), weekly["RMSE"], color="#2563eb")
    axes[0].set(title="RMSE by validation week", xlabel="Week", ylabel="RMSE (units)")
    axes[1].bar(weekly["week"].astype(str), 100 * weekly["WAPE"], color="#0f766e")
    axes[1].set(title="WAPE by validation week", xlabel="Week", ylabel="WAPE (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_validation_week_metrics.png", dpi=160)
    plt.close(fig)

    segment = metrics.loc[metrics["breakdown"].eq("volume_segment")].copy()
    order = pd.Categorical(segment["group"], categories=["low", "medium", "high"], ordered=True)
    segment = segment.assign(order=order).sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#93c5fd", "#3b82f6", "#1e3a8a"]
    axes[0].bar(segment["group"], segment["RMSE"], color=colors)
    axes[0].set(title="RMSE by volume segment", xlabel="Segment", ylabel="RMSE (units)")
    axes[1].bar(segment["group"], 100 * segment["WAPE"], color=colors)
    axes[1].set(title="WAPE by volume segment", xlabel="Segment", ylabel="WAPE (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_validation_segment_metrics.png", dpi=160)
    plt.close(fig)
    return sorted(path.name for path in figures_dir.glob("*.png"))


def run(root: Path = ROOT) -> dict:
    panel = pd.read_parquet(
        root / "data" / "processed" / "forecasting_panel.parquet",
        filters=[("period", "in", ["development", "validation"])],
    )
    metadata = pd.read_parquet(root / "data" / "processed" / "item_metadata.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # The Parquet predicate prevents test rows from entering this process.
    modeling_panel = panel.copy()
    require(modeling_panel["date"].max() < pd.Timestamp(TEST_START), "Test rows entered feature construction")
    modeling_panel = modeling_panel.merge(
        metadata[["item_id", "volume_segment"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    features = build_features(modeling_panel)
    validate_feature_values(features)
    eligible = eligible_history_rows(features)
    development = features["period"].eq("development")
    first_eligible_date = features.loc[eligible, "date"].min()
    model_ready = features.loc[eligible, [*REPORTING_METADATA, TARGET, *MODEL_FEATURES]].copy()
    require(model_ready["date"].max() < pd.Timestamp(TEST_START), "Test date entered model_features.parquet")

    validation = model_ready.loc[model_ready["period"].eq("validation")].copy()
    require(len(validation) == 2_800, f"Expected 2,800 validation rows, found {len(validation)}")
    require(not validation[HISTORY_FEATURES].isna().any().any(), "Validation history features are incomplete")
    validation["validation_week"] = (
        (validation["date"] - pd.Timestamp(VALIDATION_START)).dt.days // 7 + 1
    ).astype(int)
    predictions = validation[
        ["item_id", "date", "validation_week", "volume_segment", "weekday", "units_sold", "lag_7"]
    ].rename(columns={"units_sold": "actual", "lag_7": "prediction"})
    predictions["error"] = predictions["prediction"] - predictions["actual"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["demand_status"] = np.where(predictions["actual"].eq(0), "zero", "positive")
    require(predictions["prediction"].eq(validation["lag_7"]).all(), "Baseline differs from lag_7")
    require(predictions["date"].max() < pd.Timestamp(TEST_START), "Test date entered validation predictions")

    metrics = metric_breakdowns(predictions)
    sanity = feature_sanity(features, model_ready)
    output = root / "outputs" / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    figures = create_figures(predictions, metrics, output / "figures")
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    model_ready.to_parquet(root / "data" / "processed" / "model_features.parquet", index=False)

    overall = metrics.loc[
        metrics["breakdown"].eq("overall") & metrics["group"].eq("all")
    ].iloc[0]
    summary = {
        "model_features": MODEL_FEATURES,
        "reporting_metadata": REPORTING_METADATA,
        "target": TARGET,
        "first_feature_eligible_date": str(first_eligible_date.date()),
        "development_rows_before_eligibility": int(development.sum()),
        "development_rows_after_eligibility": int((development & eligible).sum()),
        "validation_rows": int(len(validation)),
        "validation_history_features_complete": bool(
            validation[HISTORY_FEATURES].notna().all().all()
        ),
        "overall_validation": {
            "n": int(overall["n"]),
            "MAE": float(overall["MAE"]),
            "RMSE": float(overall["RMSE"]),
            "WAPE": float(overall["WAPE"]),
        },
        "breakdowns": json_safe_records(metrics),
        "feature_sanity": sanity,
        "figures": figures,
        "test_rows_in_model_features": 0,
        "test_rows_in_validation_predictions": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
