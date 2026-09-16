"""Project-wide data scope and temporal boundaries."""

from pathlib import Path

STORE_ID = "CA_3"
DEPT_ID = "FOODS_3"
PRODUCT_COUNT = 100
EARLY_SELECTION_DAYS = 730

DEVELOPMENT_END = "2015-08-30"
VALIDATION_START = "2015-08-31"
VALIDATION_END = "2015-09-27"
TEST_START = "2015-09-28"
TEST_END = "2015-10-25"

RAW_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
)

EXPECTED_RAW = {
    "calendar.csv": {
        "bytes": 103_469,
        "sha256": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
    },
    "sales_train_evaluation.csv": {
        "bytes": 121_736_518,
        "sha256": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
    },
    "sell_prices.csv": {
        "bytes": 203_395_785,
        "sha256": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
    },
}

ROOT = Path(__file__).resolve().parents[2]
