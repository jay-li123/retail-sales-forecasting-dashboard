"""Integrity tests for the artifact-only Streamlit dashboard."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.data import (
    REQUIRED_ARTIFACTS,
    load_dashboard_data,
    model_predictions,
    overall_metric,
    recompute_product_summary,
)


ROOT = Path(__file__).resolve().parents[1]


class DashboardArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dashboard_data.clear()
        cls.data = load_dashboard_data()

    def test_all_required_artifacts_exist(self) -> None:
        self.assertTrue(all(path.is_file() for path in REQUIRED_ARTIFACTS.values()))

    def test_final_test_shape_products_and_dates(self) -> None:
        predictions = model_predictions(self.data)
        self.assertEqual(len(predictions), 2_800)
        self.assertEqual(predictions["item_id"].nunique(), 100)
        self.assertEqual(predictions["date"].min(), pd.Timestamp("2015-09-28"))
        self.assertEqual(predictions["date"].max(), pd.Timestamp("2015-10-25"))

    def test_prediction_schema(self) -> None:
        required = {
            "item_id", "date", "volume_segment", "weekday", "actual",
            "test_week", "model", "prediction", "error", "absolute_error",
            "demand_status",
        }
        self.assertTrue(required.issubset(self.data["predictions"].columns))

    def test_headline_metrics_match_summary(self) -> None:
        for model in ("seasonal_naive", "lightgbm"):
            row = overall_metric(self.data, model)
            saved = self.data["summary"]["final_test"][model]
            for metric in ("MAE", "RMSE", "WAPE"):
                self.assertAlmostEqual(float(row[metric]), float(saved[metric]), places=12)

    def test_product_summary_reconciles_to_predictions(self) -> None:
        calculated = recompute_product_summary(model_predictions(self.data))
        saved = self.data["products"].sort_values("item_id").reset_index(drop=True)
        self.assertEqual(calculated["item_id"].tolist(), saved["item_id"].tolist())
        for column in (
            "actual_total", "prediction_total", "MAE", "RMSE", "WAPE",
            "absolute_error_share",
        ):
            np.testing.assert_allclose(
                calculated[column], saved[column], rtol=1e-10, atol=1e-10, equal_nan=True
            )

    def test_feature_importance_has_all_twelve_features(self) -> None:
        importance = self.data["importance"]
        self.assertEqual(len(importance), 12)
        self.assertEqual(importance["feature"].nunique(), 12)
        self.assertAlmostEqual(float(importance["gain_share"].sum()), 1.0, places=9)

    def test_zero_demand_wape_is_undefined(self) -> None:
        rows = self.data["errors"]
        zero = rows.loc[rows["breakdown"].eq("demand_status") & rows["group"].eq("zero")]
        self.assertEqual(len(zero), 1)
        self.assertTrue(pd.isna(zero.iloc[0]["WAPE"]))

    def test_dashboard_has_exactly_three_pages(self) -> None:
        source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
        navigation = '["Overview", "Forecast Explorer", "Error Analysis"]'
        self.assertEqual(source.count(navigation), 1)

    def test_dashboard_contains_no_training_entry_points(self) -> None:
        source = "\n".join(
            (ROOT / "app" / name).read_text(encoding="utf-8")
            for name in ("app.py", "data.py", "charts.py")
        ).lower()
        for forbidden in (
            "fit_global_model", "lgb.train", "lightgbm.train",
            "retail_forecasting.model", "retail_forecasting.evaluation",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
