from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecasting.constants import TEST_START  # noqa: E402
from retail_forecasting.features import (  # noqa: E402
    HISTORY_FEATURES,
    build_features,
    eligible_history_rows,
)
from retail_forecasting.metrics import mae, rmse, wape  # noqa: E402


def synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    pieces = []
    for item, offset in (("A", 0), ("B", 100)):
        pieces.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "date": dates,
                    "units_sold": np.arange(50, dtype=float) + offset,
                    "weekday": dates.day_name(),
                    "month": dates.month,
                    "snap": 0,
                    "event_indicator": 0,
                    "sell_price": 2.0,
                    "period": "development",
                    "volume_segment": "low" if item == "A" else "high",
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


class FeatureTimingTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel()
        self.features = build_features(self.panel)
        self.target_date = pd.Timestamp("2020-02-05")

    def row(self, frame: pd.DataFrame, item: str = "A") -> pd.Series:
        return frame.loc[
            frame["item_id"].eq(item) & frame["date"].eq(self.target_date)
        ].iloc[0]

    def test_exact_lag_timing(self):
        row = self.row(self.features)
        series = self.panel.loc[self.panel["item_id"].eq("A")].set_index("date")["units_sold"]
        self.assertEqual(row["lag_7"], series.loc[self.target_date - pd.Timedelta(days=7)])
        self.assertEqual(row["lag_14"], series.loc[self.target_date - pd.Timedelta(days=14)])
        self.assertEqual(row["lag_28"], series.loc[self.target_date - pd.Timedelta(days=28)])

    def test_exact_rolling_timing_excludes_target(self):
        row = self.row(self.features)
        series = self.panel.loc[self.panel["item_id"].eq("A")].set_index("date")["units_sold"]
        expected_7 = series.loc[
            self.target_date - pd.Timedelta(days=7) : self.target_date - pd.Timedelta(days=1)
        ]
        expected_28 = series.loc[
            self.target_date - pd.Timedelta(days=28) : self.target_date - pd.Timedelta(days=1)
        ]
        self.assertAlmostEqual(row["rolling_mean_7"], expected_7.mean())
        self.assertAlmostEqual(row["rolling_mean_28"], expected_28.mean())
        self.assertAlmostEqual(row["rolling_std_28"], expected_28.std(ddof=0))

    def test_future_and_target_perturbation_invariance(self):
        changed = self.panel.copy()
        mask = changed["item_id"].eq("A") & changed["date"].ge(self.target_date)
        changed.loc[mask, "units_sold"] = 999_999
        changed_features = build_features(changed)
        pd.testing.assert_series_equal(
            self.row(self.features)[HISTORY_FEATURES],
            self.row(changed_features)[HISTORY_FEATURES],
            check_names=False,
        )

    def test_item_isolation(self):
        changed = self.panel.copy()
        changed.loc[changed["item_id"].eq("A"), "units_sold"] = 999_999
        changed_features = build_features(changed)
        original_b = self.features.loc[self.features["item_id"].eq("B"), HISTORY_FEATURES].reset_index(drop=True)
        changed_b = changed_features.loc[changed_features["item_id"].eq("B"), HISTORY_FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_b, changed_b)

    def test_shuffled_input_is_deterministic(self):
        shuffled = self.panel.sample(frac=1, random_state=19)
        changed = build_features(shuffled)
        columns = ["item_id", "date", *HISTORY_FEATURES]
        pd.testing.assert_frame_equal(self.features[columns], changed[columns])

    def test_training_eligibility_and_no_infinite_features(self):
        eligible = eligible_history_rows(self.features)
        first_dates = self.features.loc[eligible].groupby("item_id")["date"].min()
        self.assertTrue(first_dates.eq(pd.Timestamp("2020-01-29")).all())
        self.assertTrue((np.isfinite(self.features[HISTORY_FEATURES]) | self.features[HISTORY_FEATURES].isna()).all().all())


class MetricTests(unittest.TestCase):
    def test_metric_formulas(self):
        actual = np.array([0.0, 2.0])
        prediction = np.array([1.0, 4.0])
        self.assertAlmostEqual(mae(actual, prediction), 1.5)
        self.assertAlmostEqual(rmse(actual, prediction), np.sqrt(2.5))
        self.assertAlmostEqual(wape(actual, prediction), 1.5)

    def test_zero_denominator_wape_is_nan(self):
        self.assertTrue(np.isnan(wape([0, 0], [1, 2])))


class SavedBaselineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_parquet(ROOT / "outputs" / "baseline" / "validation_predictions.parquet")
        cls.features = pd.read_parquet(ROOT / "data" / "processed" / "model_features.parquet")
        cls.summary = pd.read_json(ROOT / "outputs" / "baseline" / "summary.json", typ="series")

    def test_baseline_equals_lag_7_and_has_expected_count(self):
        self.assertEqual(len(self.predictions), 2_800)
        panel = pd.read_parquet(
            ROOT / "data" / "processed" / "forecasting_panel.parquet",
            columns=["item_id", "date", "units_sold"],
            filters=[("period", "in", ["development", "validation"])],
        ).sort_values(["item_id", "date"])
        panel["lag_7"] = panel.groupby("item_id")["units_sold"].shift(7)
        expected = self.predictions[["item_id", "date"]].merge(
            panel[["item_id", "date", "lag_7"]],
            on=["item_id", "date"],
            validate="one_to_one",
        )
        np.testing.assert_array_equal(self.predictions["prediction"], expected["lag_7"])

    def test_test_dates_are_absent_from_saved_artifacts(self):
        boundary = pd.Timestamp(TEST_START)
        self.assertLess(self.predictions["date"].max(), boundary)
        self.assertLess(self.features["date"].max(), boundary)
        self.assertNotIn("test", set(self.features["period"]))

    def test_validation_features_are_complete_and_finite(self):
        validation = self.features.loc[self.features["period"].eq("validation")]
        self.assertEqual(len(validation), 2_800)
        self.assertFalse(validation[HISTORY_FEATURES].isna().any().any())
        self.assertTrue(np.isfinite(validation[HISTORY_FEATURES].to_numpy()).all())

    def test_startup_missing_counts_are_expected(self):
        sanity = self.summary["feature_sanity"]["features"]
        self.assertEqual(sanity["lag_7"]["missing"], 700)
        self.assertEqual(sanity["lag_14"]["missing"], 1_400)
        for name in ("lag_28", "rolling_mean_28", "rolling_std_28"):
            self.assertEqual(sanity[name]["missing"], 2_800)
        self.assertEqual(sanity["rolling_mean_7"]["missing"], 700)


if __name__ == "__main__":
    unittest.main()
