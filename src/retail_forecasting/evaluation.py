"""Milestone 4 fixed-model validation, final test, and error analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.constants import (
    ROOT,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
)
from retail_forecasting.features import (
    HISTORY_FEATURES,
    MODEL_FEATURES,
    TARGET,
    build_features,
    eligible_history_rows,
)
from retail_forecasting.metrics import metric_row
from retail_forecasting.model import (
    CATEGORICAL_FEATURES,
    LIGHTGBM_PARAMETERS,
    deterministic_category_mapping,
    fit_global_model,
    predict_nonnegative,
)
from retail_forecasting.validation import require


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def weekly_windows(start: str, end: str) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    starts = pd.date_range(first, last, freq="7D")
    windows = [(index + 1, week_start, week_start + pd.Timedelta(days=6)) for index, week_start in enumerate(starts)]
    require(len(windows) == 4 and windows[-1][2] == last, "Expected exactly four complete evaluation weeks")
    return windows


def select_final_model(lightgbm_rmse: float, baseline_rmse: float) -> str:
    """Locked decision rule based on validation RMSE only."""

    return "lightgbm" if lightgbm_rmse < baseline_rmse else "seasonal_naive"


def walk_forward(
    features: pd.DataFrame,
    *,
    phase: str,
    start: str,
    end: str,
    mapping: dict[str, list],
) -> tuple[pd.DataFrame, pd.DataFrame, int, list[np.ndarray]]:
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    importances: list[np.ndarray] = []
    clipped_total = 0

    for week, week_start, week_end in weekly_windows(start, end):
        training = features.loc[features["date"].lt(week_start)].copy()
        forecast = features.loc[features["date"].between(week_start, week_end)].copy()
        require(not training.empty, f"{phase} week {week} has no training rows")
        require(len(forecast) == 700, f"{phase} week {week} must contain 700 forecast rows")
        require(training["date"].max() < forecast["date"].min(), "Training reaches forecast dates")
        require(not forecast[HISTORY_FEATURES].isna().any().any(), "Forecast history features are incomplete")

        model = fit_global_model(training, mapping)
        lightgbm_prediction, clipped = predict_nonnegative(model, forecast, mapping)
        clipped_total += clipped
        importances.append(model.feature_importance(importance_type="gain"))

        base = forecast[["item_id", "date", "volume_segment", "weekday", TARGET, "lag_7"]].copy()
        base = base.rename(columns={TARGET: "actual"})
        base[f"{phase}_week"] = week
        for model_name, values in (
            ("seasonal_naive", base["lag_7"].to_numpy(dtype=float)),
            ("lightgbm", lightgbm_prediction),
        ):
            part = base.drop(columns="lag_7").copy()
            part["model"] = model_name
            part["prediction"] = values
            part["error"] = part["prediction"] - part["actual"]
            part["absolute_error"] = part["error"].abs()
            part["demand_status"] = np.where(part["actual"].eq(0), "zero", "positive")
            prediction_parts.append(part)

        audit_rows.append(
            {
                "phase": phase,
                "week": week,
                "training_rows": int(len(training)),
                "training_start": training["date"].min(),
                "training_end": training["date"].max(),
                "forecast_start": week_start,
                "forecast_end": week_end,
                "forecast_rows": int(len(forecast)),
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["model", "date", "item_id"], kind="stable"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    require(
        predictions.loc[predictions["model"].eq("lightgbm")].shape[0] == 2_800,
        f"{phase} LightGBM prediction count is not 2,800",
    )
    require(np.isfinite(predictions["prediction"]).all(), f"{phase} predictions are non-finite")
    require(predictions["prediction"].ge(0).all(), f"{phase} predictions are negative after clipping")
    return predictions, audit, clipped_total, importances


def metrics_table(predictions: pd.DataFrame, *, phase: str, detailed_lightgbm: bool) -> pd.DataFrame:
    rows: list[dict] = []
    week_column = f"{phase}_week"

    def add(model: str, breakdown: str, group: str, data: pd.DataFrame) -> None:
        rows.append(
            {
                "model": model,
                "breakdown": breakdown,
                "group": str(group),
                **metric_row(data["actual"], data["prediction"]),
            }
        )

    for model_name, model_data in predictions.groupby("model", sort=True):
        add(model_name, "overall", "all", model_data)
        for label, data in model_data.groupby(week_column, sort=True):
            add(model_name, week_column, label, data)
        if detailed_lightgbm and model_name == "lightgbm":
            for dimension in ("volume_segment", "weekday", "demand_status"):
                for label, data in model_data.groupby(dimension, sort=True, observed=True):
                    add(model_name, dimension, label, data)
    return pd.DataFrame(rows)


def overall_metrics(metrics: pd.DataFrame, model: str) -> dict[str, float | int]:
    row = metrics.loc[
        metrics["model"].eq(model)
        & metrics["breakdown"].eq("overall")
        & metrics["group"].eq("all")
    ].iloc[0]
    return {name: (int(row[name]) if name == "n" else float(row[name])) for name in ["n", "MAE", "RMSE", "WAPE"]}


def relative_improvements(baseline: dict, lightgbm: dict) -> dict[str, float]:
    return {
        metric: float((baseline[metric] - lightgbm[metric]) / baseline[metric])
        for metric in ("MAE", "RMSE", "WAPE")
    }


def build_test_features(root: Path) -> pd.DataFrame:
    """Load test targets only after the validation-based specification exists."""

    panel = pd.read_parquet(root / "data" / "processed" / "forecasting_panel.parquet")
    metadata = pd.read_parquet(root / "data" / "processed" / "item_metadata.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(
        metadata[["item_id", "volume_segment"]], on="item_id", how="left", validate="many_to_one"
    )
    features = build_features(panel)
    return features.loc[eligible_history_rows(features)].copy()


def error_breakdowns(selected: pd.DataFrame, phase: str) -> pd.DataFrame:
    rows: list[dict] = []
    for dimension in ("volume_segment", "weekday", f"{phase}_week", "demand_status"):
        for label, data in selected.groupby(dimension, sort=True, observed=True):
            rows.append({"breakdown": dimension, "group": str(label), **metric_row(data["actual"], data["prediction"])})
    return pd.DataFrame(rows)


def product_errors(selected: pd.DataFrame) -> pd.DataFrame:
    total_absolute_error = float(selected["absolute_error"].sum())
    rows = []
    for (item_id, segment), data in selected.groupby(["item_id", "volume_segment"], sort=True, observed=True):
        metrics = metric_row(data["actual"], data["prediction"])
        rows.append(
            {
                "item_id": item_id,
                "volume_segment": segment,
                "actual_total": float(data["actual"].sum()),
                "prediction_total": float(data["prediction"].sum()),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
                "absolute_error": float(data["absolute_error"].sum()),
                "absolute_error_share": float(data["absolute_error"].sum() / total_absolute_error),
            }
        )
    return pd.DataFrame(rows).sort_values(["absolute_error_share", "item_id"], ascending=[False, True])


def representative_errors(selected: pd.DataFrame, product_summary: pd.DataFrame) -> pd.DataFrame:
    def labeled(case: str, row: pd.Series) -> dict:
        return {
            "case": case,
            "item_id": row["item_id"],
            "date": row["date"],
            "actual": float(row["actual"]),
            "prediction": float(row["prediction"]),
            "error": float(row["error"]),
            "volume_segment": row["volume_segment"],
        }

    highest_item = product_summary.iloc[0]["item_id"]
    highest_item_rows = selected.loc[selected["item_id"].eq(highest_item)]
    positive = selected.loc[selected["actual"].gt(0)]
    low = selected.loc[selected["volume_segment"].eq("low")]
    records = [
        labeled("largest_overprediction", selected.loc[selected["error"].idxmax()]),
        labeled("largest_underprediction", selected.loc[selected["error"].idxmin()]),
        labeled("smallest_absolute_error_positive_demand", positive.loc[positive["absolute_error"].idxmin()]),
        labeled("highest_error_product", highest_item_rows.loc[highest_item_rows["absolute_error"].idxmax()]),
        labeled("low_volume_example", low.loc[low["absolute_error"].idxmax()]),
    ]
    return pd.DataFrame(records)


def aggregate_feature_importance(importances: list[np.ndarray]) -> pd.DataFrame:
    gain = np.vstack(importances).sum(axis=0)
    total = float(gain.sum())
    result = pd.DataFrame({"feature": MODEL_FEATURES, "gain": gain})
    result["gain_share"] = result["gain"] / total if total else 0.0
    return result.sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)


def json_safe_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                record[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                record[key] = value.isoformat()
    return records


def create_figures(
    validation_metrics: pd.DataFrame,
    final_predictions: pd.DataFrame,
    error_analysis: pd.DataFrame,
    importance: pd.DataFrame,
    selected_model: str,
    product_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    overall = validation_metrics.loc[validation_metrics["breakdown"].eq("overall")].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, label, scale in zip(
        axes,
        ("MAE", "RMSE", "WAPE"),
        ("MAE (units)", "RMSE (units)", "WAPE (%)"),
        (1, 1, 100),
    ):
        values = [scale * overall.loc[model, metric] for model in ("seasonal_naive", "lightgbm")]
        ax.bar(["Seasonal naive", "LightGBM"], values, color=["#94a3b8", "#2563eb"])
        ax.set(title=f"Validation {metric}", ylabel=label)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "01_validation_model_comparison.png", dpi=160)
    plt.close(fig)

    selected = final_predictions.loc[final_predictions["model"].eq(selected_model)]
    daily = selected.groupby("date", as_index=False)[["actual", "prediction"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily["date"], daily["actual"], label="Actual", color="#111827", linewidth=1.8)
    display_model = "LightGBM" if selected_model == "lightgbm" else "Seasonal naive"
    ax.plot(daily["date"], daily["prediction"], label=display_model, color="#2563eb", linewidth=1.5)
    ax.set(title="Final-test total daily sales", xlabel="Date", ylabel="Units sold")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "02_final_test_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    segment = error_analysis.loc[error_analysis["breakdown"].eq("volume_segment")].copy()
    segment["order"] = pd.Categorical(
        segment["group"], categories=["low", "medium", "high"], ordered=True
    )
    segment = segment.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#93c5fd", "#3b82f6", "#1e3a8a"]
    axes[0].bar(segment["group"], segment["RMSE"], color=colors)
    axes[0].set(title="Final-test RMSE by volume", ylabel="RMSE (units)")
    axes[1].bar(segment["group"], 100 * segment["WAPE"], color=colors)
    axes[1].set(title="Final-test WAPE by volume", ylabel="WAPE (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_final_test_segment_metrics.png", dpi=160)
    plt.close(fig)

    ordered_importance = importance.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordered_importance["feature"], ordered_importance["gain_share"] * 100, color="#7c3aed")
    ax.set(title="LightGBM aggregate gain importance", xlabel="Share of gain (%)", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_feature_importance.png", dpi=160)
    plt.close(fig)

    worst_item = product_summary.iloc[0]["item_id"]
    worst = selected.loc[selected["item_id"].eq(worst_item)].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(worst["date"], worst["actual"], marker="o", label="Actual", color="#111827")
    ax.plot(worst["date"], worst["prediction"], marker="o", label="Prediction", color="#dc2626")
    ax.set(title=f"Highest error-contribution product: {worst_item}", xlabel="Date", ylabel="Units sold")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "05_worst_product_path.png", dpi=160)
    plt.close(fig)
    return sorted(path.name for path in figures_dir.glob("*.png"))


def run(root: Path = ROOT) -> dict:
    output = root / "outputs" / "model"
    output.mkdir(parents=True, exist_ok=True)
    specification_path = output / "model_specification.json"

    validation_features = pd.read_parquet(root / "data" / "processed" / "model_features.parquet")
    validation_features["date"] = pd.to_datetime(validation_features["date"])
    require(validation_features["date"].max() < pd.Timestamp(TEST_START), "Test rows entered validation fitting")
    mapping = deterministic_category_mapping(validation_features["item_id"])

    validation_predictions, validation_audit, validation_clipped, _ = walk_forward(
        validation_features,
        phase="validation",
        start=VALIDATION_START,
        end=VALIDATION_END,
        mapping=mapping,
    )
    validation_metrics = metrics_table(
        validation_predictions, phase="validation", detailed_lightgbm=False
    )
    baseline_validation = overall_metrics(validation_metrics, "seasonal_naive")
    lightgbm_validation = overall_metrics(validation_metrics, "lightgbm")
    validation_improvement = relative_improvements(baseline_validation, lightgbm_validation)
    selected_model = select_final_model(
        lightgbm_validation["RMSE"], baseline_validation["RMSE"]
    )

    specification = {
        "created_at_utc": utc_now(),
        "feature_list": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_mapping": mapping,
        "lightgbm_parameters": LIGHTGBM_PARAMETERS,
        "validation_dates": [VALIDATION_START, VALIDATION_END],
        "validation_metrics": {
            "seasonal_naive": baseline_validation,
            "lightgbm": lightgbm_validation,
            "relative_lightgbm_improvement": validation_improvement,
        },
        "selected_final_model": selected_model,
        "selection_rule": "Select LightGBM only when its pooled validation RMSE is lower than seasonal naive.",
        "random_seed": LIGHTGBM_PARAMETERS["random_state"],
        "final_test_scored": False,
    }
    specification_path.write_text(
        json.dumps(specification, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    require(specification_path.is_file(), "Model specification was not saved before test evaluation")

    final_test_started_at = utc_now()
    frozen = json.loads(specification_path.read_text(encoding="utf-8"))
    require(frozen["selected_final_model"] == selected_model, "Frozen model decision changed")
    require(frozen["feature_list"] == MODEL_FEATURES, "Frozen feature list changed")
    require(frozen["lightgbm_parameters"] == LIGHTGBM_PARAMETERS, "Frozen parameters changed")

    all_features = build_test_features(root)
    final_predictions, test_audit, test_clipped, test_importances = walk_forward(
        all_features,
        phase="test",
        start=TEST_START,
        end=TEST_END,
        mapping=mapping,
    )
    final_metrics = metrics_table(final_predictions, phase="test", detailed_lightgbm=True)
    baseline_test = overall_metrics(final_metrics, "seasonal_naive")
    lightgbm_test = overall_metrics(final_metrics, "lightgbm")
    test_improvement = relative_improvements(baseline_test, lightgbm_test)

    selected_predictions = final_predictions.loc[final_predictions["model"].eq(selected_model)].copy()
    analysis = error_breakdowns(selected_predictions, "test")
    product_summary = product_errors(selected_predictions)
    representative = representative_errors(selected_predictions, product_summary)
    importance = aggregate_feature_importance(test_importances)
    figures = create_figures(
        validation_metrics,
        final_predictions,
        analysis,
        importance,
        selected_model,
        product_summary,
        output / "figures",
    )

    validation_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    validation_metrics.to_csv(output / "validation_metrics.csv", index=False)
    final_predictions.to_parquet(output / "final_test_predictions.parquet", index=False)
    final_metrics.to_csv(output / "final_test_metrics.csv", index=False)
    analysis.to_csv(output / "error_analysis.csv", index=False)
    product_summary.to_csv(output / "product_error_summary.csv", index=False)
    representative.to_csv(output / "representative_errors.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    fit_audit = pd.concat([validation_audit, test_audit], ignore_index=True)
    fit_audit.to_csv(output / "fit_audit.csv", index=False)

    summary = {
        "features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "parameters": LIGHTGBM_PARAMETERS,
        "validation": {
            "seasonal_naive": baseline_validation,
            "lightgbm": lightgbm_validation,
            "relative_lightgbm_improvement": validation_improvement,
            "lightgbm_predictions_clipped": validation_clipped,
        },
        "selected_final_model": selected_model,
        "selection_basis": "validation RMSE only",
        "specification_created_at_utc": specification["created_at_utc"],
        "final_test_started_at_utc": final_test_started_at,
        "final_test": {
            "seasonal_naive": baseline_test,
            "lightgbm": lightgbm_test,
            "relative_lightgbm_improvement": test_improvement,
            "lightgbm_predictions_clipped": test_clipped,
        },
        "final_test_lightgbm_breakdowns": json_safe_records(
            final_metrics.loc[
                final_metrics["model"].eq("lightgbm")
                & ~final_metrics["breakdown"].eq("overall")
            ]
        ),
        "top_products_by_mae": json_safe_records(
            product_summary.sort_values(["MAE", "item_id"], ascending=[False, True]).head(10)
        ),
        "top_products_by_absolute_error_share": json_safe_records(product_summary.head(10)),
        "representative_errors": json_safe_records(representative),
        "feature_importance": json_safe_records(importance),
        "figures": figures,
        "validation_lightgbm_prediction_count": int(
            validation_predictions["model"].eq("lightgbm").sum()
        ),
        "final_test_lightgbm_prediction_count": int(
            final_predictions["model"].eq("lightgbm").sum()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2, default=str))


if __name__ == "__main__":
    main()
