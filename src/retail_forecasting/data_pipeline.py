"""Build and describe the Milestone 2 M5 daily panel without training models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.constants import (
    DEPT_ID,
    DEVELOPMENT_END,
    EARLY_SELECTION_DAYS,
    EXPECTED_RAW,
    PRODUCT_COUNT,
    RAW_FILES,
    ROOT,
    STORE_ID,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
)
from retail_forecasting.validation import (
    SALES_METADATA,
    require,
    validate_calendar_frame,
    validate_price_frame,
    validate_processed_panel,
    validate_sales_header,
    validate_sales_values,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_files(raw_dir: Path) -> dict[str, dict[str, int | str]]:
    results: dict[str, dict[str, int | str]] = {}
    for name in RAW_FILES:
        path = raw_dir / name
        require(path.is_file(), f"Missing required raw file: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        require(actual == EXPECTED_RAW[name], f"Checksum or size mismatch for {name}: {actual}")
        results[name] = actual
    return results


def load_and_validate_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    checksums = verify_raw_files(raw_dir)
    calendar_raw = pd.read_csv(raw_dir / "calendar.csv")
    calendar = validate_calendar_frame(calendar_raw)

    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category"},
    )
    validate_price_frame(prices)
    require(prices["wm_yr_wk"].isin(calendar["wm_yr_wk"]).all(), "prices reference unknown calendar weeks")

    sales_path = raw_dir / "sales_train_evaluation.csv"
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    day_columns = validate_sales_header(header, set(calendar["d"]))
    scoped_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    sales_rows = 0
    for chunk in pd.read_csv(sales_path, chunksize=1_000):
        validate_sales_values(chunk, day_columns)
        metadata_parts.append(chunk[SALES_METADATA])
        sales_rows += len(chunk)
        scoped = chunk.loc[chunk["store_id"].eq(STORE_ID) & chunk["dept_id"].eq(DEPT_ID)]
        if not scoped.empty:
            scoped_parts.append(scoped.copy())

    metadata = pd.concat(metadata_parts, ignore_index=True)
    require(not metadata["id"].duplicated().any(), "sales contains duplicate id values")
    require(
        not metadata.duplicated(["item_id", "store_id"]).any(),
        "sales contains duplicate item/store series",
    )
    scoped_sales = pd.concat(scoped_parts, ignore_index=True)
    require(not scoped_sales.empty, f"No sales rows found for {STORE_ID} / {DEPT_ID}")
    source_info = {
        "checksums": checksums,
        "source_shapes": {
            "calendar.csv": [int(calendar_raw.shape[0]), int(calendar_raw.shape[1])],
            "sales_train_evaluation.csv": [int(sales_rows), int(len(header))],
            "sell_prices.csv": [int(prices.shape[0]), int(prices.shape[1])],
        },
    }
    return calendar, prices, scoped_sales, day_columns, source_info


def select_established_products(
    scoped_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    day_columns: list[str],
    *,
    count: int = PRODUCT_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select products using exactly the first 730 sales days and no later targets."""

    require(len(day_columns) >= EARLY_SELECTION_DAYS, "Fewer than 730 sales days are available")
    early_days = day_columns[:EARLY_SELECTION_DAYS]
    early_values = scoped_sales[early_days].to_numpy()
    positive = early_values > 0
    has_positive = positive.any(axis=1)
    first_index = np.where(has_positive, positive.argmax(axis=1), -1)
    early_dates = calendar.set_index("d").loc[early_days, "date"]
    first_dates = pd.Series(pd.NaT, index=scoped_sales.index, dtype="datetime64[ns]")
    first_dates.loc[has_positive] = early_dates.iloc[first_index[has_positive]].to_numpy()
    selection_cutoff = early_dates.iloc[-1]

    candidates = scoped_sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    candidates["first_positive_date"] = first_dates.to_numpy()
    candidates["positive_days_early"] = positive.sum(axis=1).astype(int)
    candidates["total_units_early"] = early_values.sum(axis=1).astype(int)
    candidates["established_days_early"] = (selection_cutoff - candidates["first_positive_date"]).dt.days + 1
    candidates["eligible"] = (
        candidates["established_days_early"].ge(365)
        & candidates["positive_days_early"].ge(28)
    )
    eligible = candidates.loc[candidates["eligible"]].sort_values(
        ["positive_days_early", "total_units_early", "item_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    require(len(eligible) >= count, f"Only {len(eligible)} products satisfy the establishment rules")
    selected = eligible.head(count).copy()
    selected["selection_rank"] = np.arange(1, count + 1)
    selected["selection_cutoff"] = selection_cutoff
    selected = selected.reset_index(drop=True)
    require(selected["item_id"].nunique() == count, "Product selection did not produce unique items")
    return selected, candidates


def assign_period(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    conditions = [
        dates.le(pd.Timestamp(DEVELOPMENT_END)),
        dates.between(VALIDATION_START, VALIDATION_END),
        dates.between(TEST_START, TEST_END),
    ]
    values = ["development", "validation", "test"]
    result = pd.Series(np.select(conditions, values, default="outside"), index=dates.index)
    require(not result.eq("outside").any(), "Panel contains dates outside the declared periods")
    return result


def assign_volume_segments(panel: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic tertiles using development targets only."""

    development = panel.loc[panel["period"].eq("development")]
    require(not development.empty, "No development rows are available for volume segments")
    stats = (
        development.groupby("item_id", as_index=False, observed=True)
        .agg(average_daily_sales=("units_sold", "mean"), development_total_units=("units_sold", "sum"))
        .sort_values(["average_daily_sales", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    stats["volume_rank"] = np.arange(1, len(stats) + 1)
    stats["volume_segment"] = ""
    for label, positions in zip(("low", "medium", "high"), np.array_split(np.arange(len(stats)), 3)):
        stats.loc[positions, "volume_segment"] = label
    return stats


def build_panel(
    scoped_sales: pd.DataFrame,
    selected: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    day_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    modeled_days = calendar.loc[calendar["date"].le(TEST_END), "d"].tolist()
    require(set(modeled_days).issubset(day_columns), "Modeled calendar dates are missing sales columns")
    selected_wide = scoped_sales.loc[scoped_sales["item_id"].isin(selected["item_id"])].copy()
    require(len(selected_wide) == PRODUCT_COUNT, "Selected product rows were not found exactly once")
    panel = selected_wide[SALES_METADATA[1:] + modeled_days].melt(
        id_vars=SALES_METADATA[1:],
        value_vars=modeled_days,
        var_name="d",
        value_name="units_sold",
    )
    calendar_fields = [
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
    ]
    panel = panel.merge(calendar[calendar_fields], on="d", how="left", validate="many_to_one")
    panel = panel.rename(columns={"snap_CA": "snap"})
    panel["event_indicator"] = (
        panel["event_name_1"].notna() | panel["event_name_2"].notna()
    ).astype("int8")
    scoped_prices = prices.loc[
        prices["store_id"].eq(STORE_ID) & prices["item_id"].isin(selected["item_id"]),
        ["store_id", "item_id", "wm_yr_wk", "sell_price"],
    ].copy()
    before_price_join = len(panel)
    panel = panel.merge(
        scoped_prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    require(len(panel) == before_price_join, "Price join changed the panel row count")
    panel["period"] = assign_period(panel["date"])
    panel = panel.sort_values(["item_id", "date"], kind="stable").reset_index(drop=True)
    validate_processed_panel(panel, PRODUCT_COUNT)
    require(panel["date"].max() == pd.Timestamp(TEST_END), "Panel does not end on the declared test date")

    volume = assign_volume_segments(panel)
    first_price = panel.loc[panel["sell_price"].notna()].groupby("item_id")["date"].min()
    metadata = selected.merge(volume, on="item_id", validate="one_to_one")
    metadata["first_recorded_price_date"] = metadata["item_id"].map(first_price)

    missing = panel["sell_price"].isna()
    first_price_by_row = panel["item_id"].map(first_price)
    before_first = missing & panel["date"].lt(first_price_by_row)
    on_or_after_first = missing & ~before_first
    development = panel["period"].eq("development")
    price_audit = {
        "missing_price_rows": int(missing.sum()),
        "missing_price_pct": float(100 * missing.mean()),
        "development_positive_sales_missing_price_rows": int(
            (development & missing & panel["units_sold"].gt(0)).sum()
        ),
        "missing_before_first_recorded_price_rows": int(before_first.sum()),
        "missing_on_or_after_first_recorded_price_rows": int(on_or_after_first.sum()),
        "missing_before_first_recorded_price_pct_of_missing": (
            float(100 * before_first.sum() / missing.sum()) if missing.any() else 0.0
        ),
    }
    return panel, metadata, price_audit


def development_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Single gate used by every target-based EDA calculation and figure."""

    result = panel.loc[panel["period"].eq("development")].copy()
    require(not result.empty, "Development-only EDA received no development rows")
    require(result["date"].max() <= pd.Timestamp(DEVELOPMENT_END), "EDA includes post-development targets")
    return result


def create_eda(panel: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path) -> dict:
    development = development_only(panel).merge(
        metadata[["item_id", "volume_segment"]], on="item_id", how="left", validate="many_to_one"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    daily = development.groupby("date", as_index=False)["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date"], daily["units_sold"], color="#2563eb", linewidth=0.8)
    ax.set(title="CA_3 / FOODS_3 total daily sales — development period", xlabel="Date", ylabel="Units sold")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_daily_sales.png", dpi=160)
    plt.close(fig)

    weekday = (
        development.groupby(["wday", "weekday"], as_index=False)["units_sold"].mean().sort_values("wday")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(weekday["weekday"], weekday["units_sold"], color="#2563eb")
    ax.set(title="Average item-day sales by weekday — development period", xlabel="Weekday", ylabel="Average units sold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "02_weekday_sales.png", dpi=160)
    plt.close(fig)

    zero_pct = 100 * development["units_sold"].eq(0).mean()
    positive_sales = development.loc[development["units_sold"].gt(0), "units_sold"]
    upper = max(1, float(positive_sales.quantile(0.99))) if not positive_sales.empty else 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(positive_sales.clip(upper=upper), bins=40, color="#0f766e", alpha=0.9)
    ax.set(
        title=f"Positive-sales distribution (99th percentile capped)\nZero-demand item-days: {zero_pct:.1f}%",
        xlabel="Units sold",
        ylabel="Item-days",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "03_sales_distribution.png", dpi=160)
    plt.close(fig)

    segment = (
        development.groupby("volume_segment", observed=True)["units_sold"]
        .mean()
        .reindex(["low", "medium", "high"])
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(segment.index, segment.values, color=["#93c5fd", "#3b82f6", "#1e3a8a"])
    ax.set(title="Average item-day sales by development-defined segment", xlabel="Volume segment", ylabel="Average units sold")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_volume_segments.png", dpi=160)
    plt.close(fig)

    prices = development.loc[development["sell_price"].notna(), "sell_price"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(prices, bins=40, color="#7c3aed", alpha=0.85)
    ax.set(
        title=f"Sell-price distribution — development period\nMissing price rows: {100 * development['sell_price'].isna().mean():.1f}%",
        xlabel="Sell price",
        ylabel="Item-days",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "05_sell_price.png", dpi=160)
    plt.close(fig)

    event_sales = development.groupby("event_indicator")["units_sold"].mean().reindex([0, 1])
    snap_sales = development.groupby("snap")["units_sold"].mean().reindex([0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(["No event", "Event"], event_sales.values, color=["#94a3b8", "#ea580c"])
    axes[0].set(title="Recorded calendar event", ylabel="Average units sold")
    axes[1].bar(["Non-SNAP", "SNAP"], snap_sales.values, color=["#94a3b8", "#16a34a"])
    axes[1].set(title="California SNAP day", ylabel="Average units sold")
    fig.suptitle("Descriptive development-period comparisons")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_event_snap_comparison.png", dpi=160)
    plt.close(fig)

    product_totals = development.groupby("item_id")["units_sold"].sum().sort_values(ascending=False)
    return {
        "target_periods_used": sorted(development["period"].unique().tolist()),
        "development_rows": int(len(development)),
        "development_total_units": int(development["units_sold"].sum()),
        "development_zero_demand_pct": float(zero_pct),
        "development_mean_units_per_item_day": float(development["units_sold"].mean()),
        "weekday_average_units": {row.weekday: float(row.units_sold) for row in weekday.itertuples()},
        "segment_average_units": {str(k): float(v) for k, v in segment.items()},
        "top_products_by_development_units": {str(k): int(v) for k, v in product_totals.head(10).items()},
        "event_average_units": {str(int(k)): float(v) for k, v in event_sales.items()},
        "snap_average_units": {str(int(k)): float(v) for k, v in snap_sales.items()},
        "development_price_missing_pct": float(100 * development["sell_price"].isna().mean()),
        "figure_files": sorted(path.name for path in figures_dir.glob("*.png")),
    }


def write_artifacts(
    root: Path,
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    summary: dict,
) -> None:
    processed = root / "data" / "processed"
    output = root / "outputs" / "eda"
    processed.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(processed / "selected_items.csv", index=False)
    panel.to_parquet(processed / "forecasting_panel.parquet", index=False)
    metadata.to_parquet(processed / "item_metadata.parquet", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )
    flat = {
        "development_rows": summary["eda"]["development_rows"],
        "development_total_units": summary["eda"]["development_total_units"],
        "development_zero_demand_pct": summary["eda"]["development_zero_demand_pct"],
        "development_mean_units_per_item_day": summary["eda"]["development_mean_units_per_item_day"],
        "development_price_missing_pct": summary["eda"]["development_price_missing_pct"],
    }
    pd.DataFrame([flat]).to_csv(output / "development_summary.csv", index=False)


def run(root: Path = ROOT) -> dict:
    raw_dir = root / "data" / "raw"
    calendar, prices, scoped_sales, day_columns, source_info = load_and_validate_raw(raw_dir)
    selected, candidates = select_established_products(scoped_sales, calendar, day_columns)
    panel, metadata, price_audit = build_panel(scoped_sales, selected, calendar, prices, day_columns)
    eda = create_eda(panel, metadata, root / "outputs" / "eda")
    period_counts = panel["period"].value_counts().reindex(["development", "validation", "test"]).astype(int)
    summary = {
        **source_info,
        "scope": {"store_id": STORE_ID, "dept_id": DEPT_ID},
        "candidate_products": int(len(candidates)),
        "eligible_products": int(candidates["eligible"].sum()),
        "selected_products": int(len(selected)),
        "selection": {
            "days_used": EARLY_SELECTION_DAYS,
            "cutoff": str(pd.Timestamp(selected["selection_cutoff"].iloc[0]).date()),
            "minimum_established_days": 365,
            "minimum_positive_days": 28,
            "ranking": [
                "positive_days_early descending",
                "total_units_early descending",
                "item_id ascending",
            ],
        },
        "panel_shape": [int(panel.shape[0]), int(panel.shape[1])],
        "panel_date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "period_rows": {str(k): int(v) for k, v in period_counts.items()},
        "price_audit": price_audit,
        "segment_counts": {
            str(k): int(v) for k, v in metadata["volume_segment"].value_counts().sort_index().items()
        },
        "eda": eda,
        "checks_passed": [
            "raw file sizes and SHA-256 checksums",
            "raw schemas, keys, date order, and value domains",
            "filter CA_3 / FOODS_3 before reshape",
            "selection uses only first 730 days",
            "100 deterministic established products",
            "complete unique daily item/date panel",
            "many-to-one calendar and price joins",
            "declared period boundaries",
            "development-only volume segments and target EDA",
        ],
    }
    write_artifacts(root, panel, metadata, selected, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = run(args.root.resolve())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
