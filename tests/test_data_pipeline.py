from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecasting.constants import (  # noqa: E402
    DEVELOPMENT_END,
    PRODUCT_COUNT,
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
)
from retail_forecasting.data_pipeline import (  # noqa: E402
    assign_volume_segments,
    development_only,
    select_established_products,
)
from retail_forecasting.validation import (  # noqa: E402
    validate_calendar_frame,
    validate_price_frame,
    validate_sales_header,
)


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2011-01-29", periods=3, freq="D")
        self.calendar = pd.DataFrame(
            {
                "date": dates,
                "d": ["d_1", "d_2", "d_3"],
                "wm_yr_wk": [11101, 11101, 11101],
                "weekday": dates.day_name(),
                "wday": [1, 2, 3],
                "month": dates.month,
                "year": dates.year,
                "event_name_1": [np.nan] * 3,
                "event_type_1": [np.nan] * 3,
                "event_name_2": [np.nan] * 3,
                "event_type_2": [np.nan] * 3,
                "snap_CA": [0, 1, 0],
            }
        )

    def test_source_schema_validation(self):
        validated = validate_calendar_frame(self.calendar)
        self.assertEqual(len(validated), 3)
        days = validate_sales_header(
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1", "d_2", "d_3"],
            {"d_1", "d_2", "d_3"},
        )
        self.assertEqual(days, ["d_1", "d_2", "d_3"])
        validate_price_frame(
            pd.DataFrame(
                {"store_id": ["CA_3"], "item_id": ["x"], "wm_yr_wk": [11101], "sell_price": [1.0]}
            )
        )

    def test_structural_failures_raise(self):
        bad = self.calendar.copy()
        bad.loc[1, "d"] = "d_1"
        with self.assertRaisesRegex(ValueError, "duplicate d"):
            validate_calendar_frame(bad)
        duplicate_price = pd.DataFrame(
            {
                "store_id": ["CA_3", "CA_3"],
                "item_id": ["x", "x"],
                "wm_yr_wk": [1, 1],
                "sell_price": [1.0, 1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_price_frame(duplicate_price)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days = [f"d_{i}" for i in range(1, 732)]
        dates = pd.date_range("2011-01-29", periods=731, freq="D")
        cls.calendar = pd.DataFrame({"d": cls.days, "date": dates})
        rows = []
        for i in range(101):
            row = {
                "id": f"item_{i}_CA_3_evaluation",
                "item_id": f"item_{i:03d}",
                "dept_id": "FOODS_3",
                "cat_id": "FOODS",
                "store_id": "CA_3",
                "state_id": "CA",
            }
            row.update({day: 0 for day in cls.days})
            positive_days = 28 + i
            for day in cls.days[:positive_days]:
                row[day] = 1
            rows.append(row)
        cls.sales = pd.DataFrame(rows)

    def test_deterministic_selection_and_exact_count(self):
        first, candidates = select_established_products(self.sales, self.calendar, self.days)
        second, _ = select_established_products(
            self.sales.sample(frac=1, random_state=7), self.calendar, self.days
        )
        self.assertEqual(len(candidates), 101)
        self.assertEqual(first["item_id"].tolist(), second["item_id"].tolist())
        self.assertEqual(first["item_id"].nunique(), PRODUCT_COUNT)

    def test_selection_uses_only_first_730_days(self):
        changed = self.sales.copy()
        changed["d_731"] = np.arange(len(changed)) * 1_000_000
        original, _ = select_established_products(self.sales, self.calendar, self.days)
        modified, _ = select_established_products(changed, self.calendar, self.days)
        self.assertEqual(original["item_id"].tolist(), modified["item_id"].tolist())


class ArtifactIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = pd.read_parquet(ROOT / "data" / "processed" / "forecasting_panel.parquet")
        cls.metadata = pd.read_parquet(ROOT / "data" / "processed" / "item_metadata.parquet")
        cls.selected = pd.read_csv(ROOT / "data" / "processed" / "selected_items.csv")
        cls.summary = json.loads((ROOT / "outputs" / "eda" / "summary.json").read_text(encoding="utf-8"))

    def test_exactly_100_products_and_deterministic_rows(self):
        expected_days = len(pd.date_range(self.panel["date"].min(), TEST_END, freq="D"))
        self.assertEqual(self.panel["item_id"].nunique(), PRODUCT_COUNT)
        self.assertEqual(len(self.selected), PRODUCT_COUNT)
        self.assertEqual(len(self.panel), PRODUCT_COUNT * expected_days)
        self.assertEqual(self.summary["panel_shape"][0], len(self.panel))

    def test_item_date_keys_and_continuous_dates(self):
        self.assertFalse(self.panel.duplicated(["item_id", "date"]).any())
        expected = len(pd.date_range(self.panel["date"].min(), self.panel["date"].max(), freq="D"))
        counts = self.panel.groupby("item_id")["date"].nunique()
        self.assertTrue(counts.eq(expected).all())

    def test_nonnegative_integer_target(self):
        self.assertTrue(self.panel["units_sold"].ge(0).all())
        self.assertTrue(np.equal(self.panel["units_sold"], np.floor(self.panel["units_sold"])).all())

    def test_calendar_and_price_joins_preserve_rows(self):
        fields = ["date", "d", "wm_yr_wk", "weekday", "wday", "month", "year", "snap"]
        self.assertFalse(self.panel[fields].isna().any().any())
        self.assertFalse(self.panel.duplicated(["item_id", "date"]).any())

    def test_period_boundaries(self):
        ranges = self.panel.groupby("period")["date"].agg(["min", "max"])
        self.assertEqual(ranges.loc["development", "max"], pd.Timestamp(DEVELOPMENT_END))
        self.assertEqual(ranges.loc["validation", "min"], pd.Timestamp(VALIDATION_START))
        self.assertEqual(ranges.loc["validation", "max"], pd.Timestamp(VALIDATION_END))
        self.assertEqual(ranges.loc["test", "min"], pd.Timestamp(TEST_START))
        self.assertEqual(ranges.loc["test", "max"], pd.Timestamp(TEST_END))

    def test_segments_use_development_only(self):
        recomputed = assign_volume_segments(self.panel).set_index("item_id")["volume_segment"].sort_index()
        saved = self.metadata.set_index("item_id")["volume_segment"].sort_index()
        pd.testing.assert_series_equal(recomputed, saved, check_names=False)
        modified = self.panel.copy()
        modified.loc[modified["period"].ne("development"), "units_sold"] = 999_999
        changed = assign_volume_segments(modified).set_index("item_id")["volume_segment"].sort_index()
        pd.testing.assert_series_equal(recomputed, changed, check_names=False)

    def test_target_eda_excludes_validation_and_test(self):
        eda_frame = development_only(self.panel)
        self.assertEqual(set(eda_frame["period"]), {"development"})
        self.assertLessEqual(eda_frame["date"].max(), pd.Timestamp(DEVELOPMENT_END))
        self.assertEqual(self.summary["eda"]["target_periods_used"], ["development"])


if __name__ == "__main__":
    unittest.main()
