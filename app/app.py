"""Three-page, artifact-only Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charts import (  # noqa: E402
    feature_importance,
    metric_comparison,
    overview_daily,
    product_errors,
    product_forecast,
    segment_metrics,
    weekday_metrics,
    weekly_rmse,
)
from data import (  # noqa: E402
    load_dashboard_data,
    missing_artifacts,
    model_predictions,
    overall_metric,
    product_detail,
)


st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 4rem; padding-bottom: 3rem; max-width: 1320px;}
      [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: .65rem;}
      [data-testid="stMetricLabel"] {font-weight: 600; color: #475569;}
      h1, h2, h3 {color: #172033;}
      .muted {color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_wape(value: float) -> str:
    return "Undefined" if pd.isna(value) else f"{100 * value:.2f}%"


def page_header(label: str, title: str, subtitle: str) -> None:
    st.caption(label.upper())
    st.title(title)
    st.markdown(f'<p class="muted">{subtitle}</p>', unsafe_allow_html=True)


def overview(data: dict) -> None:
    summary = data["summary"]
    lightgbm = summary["final_test"]["lightgbm"]
    baseline = summary["final_test"]["seasonal_naive"]
    improvement = summary["final_test"]["relative_lightgbm_improvement"]

    page_header(
        "Final test · 2015-09-28 to 2015-10-25",
        "Retail demand forecasting at a glance",
        "One global LightGBM model forecasts next-day demand for 100 Walmart products using leakage-safe historical, calendar, price, event, and SNAP features.",
    )
    columns = st.columns(4)
    columns[0].metric("LightGBM RMSE", f"{lightgbm['RMSE']:.2f}", help="Root mean squared error in units")
    columns[1].metric("LightGBM WAPE", f"{100 * lightgbm['WAPE']:.2f}%")
    columns[2].metric("RMSE improvement", f"{100 * improvement['RMSE']:.2f}%", delta=f"vs. baseline {baseline['RMSE']:.2f}", delta_color="normal")
    columns[3].metric("Final-test forecasts", f"{lightgbm['n']:,}")

    st.plotly_chart(overview_daily(data["predictions"]), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(metric_comparison(data["metrics"]), width="stretch")
    with right:
        st.plotly_chart(weekly_rmse(data["metrics"]), width="stretch")

    with st.container(border=True):
        st.subheader("Methodology in one minute")
        cols = st.columns(3)
        cols[0].markdown("**Scope**\n\n100 established products  \nCA_3 / FOODS_3")
        cols[1].markdown("**Forecast process**\n\nOne day ahead  \nWeekly model refit")
        cols[2].markdown("**Leakage control**\n\nHistory ends at prior day  \nFour-week untouched test")


def forecast_explorer(data: dict) -> None:
    page_header(
        "Product-level inspection",
        "Forecast Explorer",
        "Inspect every selected product—including difficult cases—not just aggregate performance.",
    )
    products = sorted(data["products"]["item_id"].tolist())
    default = products.index("FOODS_3_681") if "FOODS_3_681" in products else 0
    selected_item = st.selectbox("Product", products, index=default)
    product, predictions = product_detail(data, selected_item)

    columns = st.columns(3)
    columns[0].metric("Volume segment", str(product["volume_segment"]).title())
    columns[1].metric("Actual total", f"{product['actual_total']:,.0f}")
    columns[2].metric("Predicted total", f"{product['prediction_total']:,.1f}")
    columns = st.columns(3)
    columns[0].metric("MAE", f"{product['MAE']:.2f}")
    columns[1].metric("RMSE", f"{product['RMSE']:.2f}")
    columns[2].metric("WAPE", fmt_wape(product["WAPE"]))

    if selected_item == "FOODS_3_681":
        st.caption("Defaulted to the largest absolute-error contributor so the dashboard does not showcase only easy forecasts.")

    st.plotly_chart(product_forecast(predictions), width="stretch")
    st.plotly_chart(product_errors(predictions), width="stretch")

    lightgbm = predictions.loc[predictions["model"].eq("lightgbm")]
    over = lightgbm.loc[lightgbm["error"].idxmax()]
    under = lightgbm.loc[lightgbm["error"].idxmin()]
    left, right = st.columns(2)
    left.info(
        f"Largest overprediction: {over['date']:%b %d} — actual {over['actual']:.0f}, "
        f"predicted {over['prediction']:.1f} ({over['error']:+.1f})."
    )
    right.info(
        f"Largest underprediction: {under['date']:%b %d} — actual {under['actual']:.0f}, "
        f"predicted {under['prediction']:.1f} ({under['error']:+.1f})."
    )


def error_analysis(data: dict) -> None:
    page_header(
        "Final-model diagnostics",
        "Error Analysis",
        "Absolute and relative metrics reveal different strengths and weaknesses across the 2,800 final-test forecasts.",
    )
    st.subheader("Performance by product volume")
    st.plotly_chart(segment_metrics(data["errors"]), width="stretch")
    st.caption(
        "High-volume products have larger errors in units; low-volume products have worse relative error. "
        "RMSE and WAPE therefore tell different stories."
    )

    st.subheader("Weekday performance")
    st.plotly_chart(weekday_metrics(data["errors"]), width="stretch")
    st.caption("Tuesday had the lowest RMSE and Saturday the highest. These descriptive results cover only four test weeks and do not imply causality.")

    st.subheader("Highest-error products")
    table = data["products"][
        [
            "item_id",
            "volume_segment",
            "actual_total",
            "prediction_total",
            "MAE",
            "RMSE",
            "WAPE",
            "absolute_error_share",
        ]
    ].copy()
    table["WAPE"] = 100 * table["WAPE"]
    table["absolute_error_share"] = 100 * table["absolute_error_share"]
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "item_id": "Product",
            "volume_segment": "Segment",
            "actual_total": st.column_config.NumberColumn("Actual total", format="%.0f"),
            "prediction_total": st.column_config.NumberColumn("Predicted total", format="%.1f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "WAPE": st.column_config.NumberColumn("WAPE", format="%.2f%%"),
            "absolute_error_share": st.column_config.NumberColumn("Absolute-error share", format="%.2f%%"),
        },
    )
    st.caption("FOODS_3_681 is the largest absolute-error contributor at 7.64% of total absolute error.")

    st.subheader("Zero versus positive demand")
    demand = data["errors"].loc[data["errors"]["breakdown"].eq("demand_status")].set_index("group")
    left, right = st.columns(2)
    with left:
        st.markdown("**Positive-demand rows**")
        cols = st.columns(3)
        cols[0].metric("MAE", f"{demand.loc['positive', 'MAE']:.2f}")
        cols[1].metric("RMSE", f"{demand.loc['positive', 'RMSE']:.2f}")
        cols[2].metric("WAPE", fmt_wape(demand.loc["positive", "WAPE"]))
    with right:
        st.markdown("**Zero-demand rows**")
        cols = st.columns(3)
        cols[0].metric("MAE", f"{demand.loc['zero', 'MAE']:.2f}")
        cols[1].metric("RMSE", f"{demand.loc['zero', 'RMSE']:.2f}")
        cols[2].metric("WAPE", "N/A")
    st.caption("WAPE is undefined for zero-demand rows because total actual demand—the denominator—is zero.")

    st.subheader("Feature importance")
    st.plotly_chart(feature_importance(data["importance"]), width="stretch")
    st.warning("Gain importance describes how the fitted model used features. It does not measure causal business impact.")


st.sidebar.title("Retail Forecasting")
st.sidebar.caption("CA_3 · FOODS_3 · 100 products")
pages = ["Overview", "Forecast Explorer", "Error Analysis"]
requested_page = st.query_params.get("page", "Overview")
page_index = pages.index(requested_page) if requested_page in pages else 0
page = st.sidebar.radio("Navigate", pages, index=page_index)
st.sidebar.divider()
st.sidebar.caption("Read-only dashboard · Saved final-test artifacts")

missing = missing_artifacts()
if missing:
    st.error(
        "Dashboard artifacts are missing. Run the documented analytical pipeline before launching Streamlit. "
        "The dashboard will not regenerate or retrain anything."
    )
    st.code("\n".join(str(path.relative_to(ROOT)) for path in missing))
    st.stop()

try:
    dashboard_data = load_dashboard_data()
except Exception as exc:
    st.error(f"Unable to load saved dashboard artifacts: {exc}")
    st.stop()

if page == "Overview":
    overview(dashboard_data)
elif page == "Forecast Explorer":
    forecast_explorer(dashboard_data)
else:
    error_analysis(dashboard_data)
