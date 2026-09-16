from __future__ import annotations

from datetime import datetime
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecasting.evaluation import select_final_model  # noqa: E402
from retail_forecasting.features import MODEL_FEATURES  # noqa: E402
from retail_forecasting.metrics import metric_row  # noqa: E402
from retail_forecasting.model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    deterministic_category_mapping,
)


LOCKED_FEATURES = [
    "item_id",
    "weekday",
    "month",
    "snap",
    "event_indicator",
    "sell_price",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]


class ModelContractTests(unittest.TestCase):
    def test_exact_locked_feature_list(self):
        self.assertEqual(MODEL_FEATURES, LOCKED_FEATURES)
        self.assertEqual(CATEGORICAL_FEATURES, ["item_id", "weekday", "month"])

    def test_category_mapping_is_deterministic(self):
        first = deterministic_category_mapping(["z", "a", "m", "a"])
        second = deterministic_category_mapping(["m", "z", "a"])
        self.assertEqual(first, second)
        self.assertEqual(first["item_id"], ["a", "m", "z"])
        self.assertEqual(first["month"], list(range(1, 13)))

    def test_final_model_choice_uses_rmse_rule(self):
        self.assertEqual(select_final_model(8.0, 9.0), "lightgbm")
        self.assertEqual(select_final_model(9.0, 9.0), "seasonal_naive")
        self.assertEqual(select_final_model(10.0, 9.0), "seasonal_naive")


class EvaluationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "outputs" / "model"
        cls.specification = json.loads((cls.output / "model_specification.json").read_text(encoding="utf-8"))
        cls.summary = json.loads((cls.output / "summary.json").read_text(encoding="utf-8"))
        cls.validation = pd.read_parquet(cls.output / "validation_predictions.parquet")
        cls.test = pd.read_parquet(cls.output / "final_test_predictions.parquet")
        cls.fit_audit = pd.read_csv(cls.output / "fit_audit.csv", parse_dates=["training_start", "training_end", "forecast_start", "forecast_end"])
        cls.importance = pd.read_csv(cls.output / "feature_importance.csv")

    def test_prediction_counts_and_finite_nonnegative_values(self):
        for frame in (self.validation, self.test):
            for model in ("seasonal_naive", "lightgbm"):
                part = frame.loc[frame["model"].eq(model)]
                self.assertEqual(len(part), 2_800)
                self.assertTrue(np.isfinite(part["prediction"]).all())
                self.assertTrue(part["prediction"].ge(0).all())

    def test_validation_and_test_week_training_cutoffs(self):
        self.assertEqual(set(self.fit_audit["phase"]), {"validation", "test"})
        self.assertEqual(len(self.fit_audit), 8)
        self.assertTrue((self.fit_audit["training_end"] < self.fit_audit["forecast_start"]).all())
        for _, rows in self.fit_audit.groupby("phase"):
            self.assertEqual(rows["week"].tolist(), [1, 2, 3, 4])

    def test_validation_fitting_never_uses_final_test_dates(self):
        validation_audit = self.fit_audit.loc[self.fit_audit["phase"].eq("validation")]
        self.assertLess(validation_audit["training_end"].max(), pd.Timestamp("2015-09-28"))
        self.assertLess(validation_audit["forecast_end"].max(), pd.Timestamp("2015-09-28"))

    def test_baseline_equals_exact_lag_7_on_final_test(self):
        baseline = self.test.loc[self.test["model"].eq("seasonal_naive")]
        panel = pd.read_parquet(
            ROOT / "data" / "processed" / "forecasting_panel.parquet",
            columns=["item_id", "date", "units_sold"],
        ).sort_values(["item_id", "date"])
        panel["lag_7"] = panel.groupby("item_id")["units_sold"].shift(7)
        expected = baseline[["item_id", "date"]].merge(
            panel[["item_id", "date", "lag_7"]],
            on=["item_id", "date"],
            validate="one_to_one",
        )
        np.testing.assert_array_equal(baseline["prediction"], expected["lag_7"])

    def test_model_specification_precedes_final_test_evaluation(self):
        created = datetime.fromisoformat(self.summary["specification_created_at_utc"])
        started = datetime.fromisoformat(self.summary["final_test_started_at_utc"])
        self.assertLess(created, started)
        self.assertFalse(self.specification["final_test_scored"])

    def test_frozen_decision_matches_validation_only_rule(self):
        validation = self.summary["validation"]
        expected = select_final_model(
            validation["lightgbm"]["RMSE"], validation["seasonal_naive"]["RMSE"]
        )
        self.assertEqual(self.specification["selected_final_model"], expected)
        self.assertEqual(self.summary["selected_final_model"], expected)

    def test_saved_metrics_use_shared_metric_formula(self):
        lightgbm = self.test.loc[self.test["model"].eq("lightgbm")]
        calculated = metric_row(lightgbm["actual"], lightgbm["prediction"])
        saved = self.summary["final_test"]["lightgbm"]
        for metric in ("MAE", "RMSE", "WAPE"):
            self.assertAlmostEqual(calculated[metric], saved[metric])

    def test_clipping_counts_are_recorded(self):
        self.assertGreaterEqual(self.summary["validation"]["lightgbm_predictions_clipped"], 0)
        self.assertGreaterEqual(self.summary["final_test"]["lightgbm_predictions_clipped"], 0)
        self.assertTrue(self.validation.loc[self.validation["model"].eq("lightgbm"), "prediction"].ge(0).all())
        self.assertTrue(self.test.loc[self.test["model"].eq("lightgbm"), "prediction"].ge(0).all())

    def test_feature_importance_has_exactly_locked_features(self):
        self.assertEqual(len(self.importance), 12)
        self.assertEqual(set(self.importance["feature"]), set(LOCKED_FEATURES))
        self.assertTrue(self.importance["gain"].ge(0).all())


if __name__ == "__main__":
    unittest.main()
