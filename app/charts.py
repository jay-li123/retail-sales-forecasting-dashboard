"""Plotly charts built exclusively from saved dashboard artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BLUE = "#2563EB"
INK = "#172033"
SLATE = "#94A3B8"
TEAL = "#0F766E"
RED = "#DC2626"
SEGMENT_COLORS = {"low": "#93C5FD", "medium": "#3B82F6", "high": "#1E3A8A"}


def finish(fig: go.Figure, *, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial, sans-serif", color=INK),
        hovermode="x unified",
        legend_title_text="",
    )
    return fig


def overview_daily(predictions: pd.DataFrame) -> go.Figure:
    lightgbm = predictions.loc[predictions["model"].eq("lightgbm")]
    actual = lightgbm.groupby("date", as_index=False)["actual"].sum()
    by_model = predictions.groupby(["model", "date"], as_index=False)["prediction"].sum()
    fig = go.Figure()
    fig.add_scatter(x=actual["date"], y=actual["actual"], name="Actual", line=dict(color=INK, width=3))
    for model, color, dash in (
        ("lightgbm", BLUE, "solid"),
        ("seasonal_naive", SLATE, "dot"),
    ):
        frame = by_model.loc[by_model["model"].eq(model)]
        label = "LightGBM" if model == "lightgbm" else "Seasonal naive"
        fig.add_scatter(x=frame["date"], y=frame["prediction"], name=label, line=dict(color=color, width=2, dash=dash))
    fig.update_layout(title="Total daily sales across the final test", xaxis_title=None, yaxis_title="Units sold")
    return finish(fig, height=420)


def metric_comparison(metrics: pd.DataFrame) -> go.Figure:
    overall = metrics.loc[metrics["breakdown"].eq("overall")].copy()
    models = ["Seasonal naive", "LightGBM"]
    keys = ["seasonal_naive", "lightgbm"]
    colors = [SLATE, BLUE]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "WAPE"))
    for column, metric in enumerate(("MAE", "RMSE", "WAPE"), start=1):
        values = [float(overall.loc[overall["model"].eq(key), metric].iloc[0]) for key in keys]
        if metric == "WAPE":
            values = [100 * value for value in values]
        fig.add_bar(
            x=models,
            y=values,
            marker_color=colors,
            text=[f"{value:.2f}{'%' if metric == 'WAPE' else ''}" for value in values],
            textposition="outside",
            showlegend=False,
            row=1,
            col=column,
        )
        fig.update_yaxes(title_text="Percent" if metric == "WAPE" else "Units", row=1, col=column)
    fig.update_layout(title="Final-test metric comparison", barmode="group")
    return finish(fig)


def weekly_rmse(metrics: pd.DataFrame) -> go.Figure:
    frame = metrics.loc[metrics["breakdown"].eq("test_week")].copy()
    frame["Model"] = frame["model"].map({"lightgbm": "LightGBM", "seasonal_naive": "Seasonal naive"})
    frame["Week"] = frame["group"].astype(int)
    fig = px.bar(
        frame,
        x="Week",
        y="RMSE",
        color="Model",
        barmode="group",
        color_discrete_map={"LightGBM": BLUE, "Seasonal naive": SLATE},
        text_auto=".2f",
        title="RMSE by final-test week",
    )
    fig.update_xaxes(dtick=1)
    return finish(fig)


def product_forecast(predictions: pd.DataFrame) -> go.Figure:
    pivot = predictions.pivot(index="date", columns="model", values="prediction").reset_index()
    actual = predictions.loc[predictions["model"].eq("lightgbm"), ["date", "actual"]]
    frame = actual.merge(pivot, on="date", validate="one_to_one")
    fig = go.Figure()
    fig.add_scatter(x=frame["date"], y=frame["actual"], name="Actual", line=dict(color=INK, width=3), mode="lines+markers")
    fig.add_scatter(x=frame["date"], y=frame["lightgbm"], name="LightGBM", line=dict(color=BLUE, width=2), mode="lines+markers")
    fig.add_scatter(x=frame["date"], y=frame["seasonal_naive"], name="Seasonal naive", line=dict(color=SLATE, width=2, dash="dot"))
    fig.update_layout(title="Daily actual and predicted demand", xaxis_title=None, yaxis_title="Units sold")
    return finish(fig, height=430)


def product_errors(predictions: pd.DataFrame) -> go.Figure:
    frame = predictions.loc[predictions["model"].eq("lightgbm")].copy()
    fig = go.Figure()
    fig.add_bar(x=frame["date"], y=frame["error"], name="Signed error", marker_color=[RED if value < 0 else TEAL for value in frame["error"]])
    fig.add_scatter(x=frame["date"], y=frame["absolute_error"], name="Absolute error", line=dict(color=INK, width=2))
    fig.update_layout(title="LightGBM daily error", xaxis_title=None, yaxis_title="Units")
    return finish(fig, height=350)


def segment_metrics(errors: pd.DataFrame) -> go.Figure:
    frame = errors.loc[errors["breakdown"].eq("volume_segment")].copy()
    frame["group"] = pd.Categorical(frame["group"], ["low", "medium", "high"], ordered=True)
    frame = frame.sort_values("group")
    fig = go.Figure()
    fig.add_bar(x=frame["group"], y=frame["RMSE"], name="RMSE", marker_color=BLUE, text=frame["RMSE"].map(lambda value: f"{value:.2f}"))
    fig.add_bar(x=frame["group"], y=100 * frame["WAPE"], name="WAPE (%)", marker_color=TEAL, text=(100 * frame["WAPE"]).map(lambda value: f"{value:.1f}%"))
    fig.update_layout(title="Absolute and relative error by volume segment", barmode="group", xaxis_title=None, yaxis_title="Metric value")
    return finish(fig)


def weekday_metrics(errors: pd.DataFrame) -> go.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    frame = errors.loc[errors["breakdown"].eq("weekday")].copy()
    frame["group"] = pd.Categorical(frame["group"], order, ordered=True)
    frame = frame.sort_values("group")
    fig = go.Figure()
    fig.add_bar(x=frame["group"], y=frame["RMSE"], name="RMSE", marker_color=BLUE)
    fig.add_scatter(x=frame["group"], y=100 * frame["WAPE"], name="WAPE (%)", line=dict(color=TEAL, width=3), yaxis="y2")
    fig.update_layout(
        title="LightGBM error by weekday",
        xaxis_title=None,
        yaxis=dict(title="RMSE (units)"),
        yaxis2=dict(title="WAPE (%)", overlaying="y", side="right"),
    )
    return finish(fig)


def feature_importance(importance: pd.DataFrame) -> go.Figure:
    frame = importance.sort_values("gain_share", ascending=True)
    fig = px.bar(
        frame,
        x=100 * frame["gain_share"],
        y="feature",
        orientation="h",
        color_discrete_sequence=["#7C3AED"],
        title="LightGBM gain importance — all 12 features",
    )
    fig.update_xaxes(title="Share of aggregate gain (%)")
    fig.update_yaxes(title=None)
    return finish(fig, height=460)
