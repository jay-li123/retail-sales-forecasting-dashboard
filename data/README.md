# Data access

Raw and processed data stay local and are excluded from Git.

## Required raw files

Milestone 2 placed independent copies of these files in `data/raw/`:

| File | Verified local size (bytes) | SHA-256 |
|---|---:|---|
| `sales_train_evaluation.csv` | 121,736,518 | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| `calendar.csv` | 103,469 | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `sell_prices.csv` | 203,395,785 | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

The verified source copies currently exist under:

```text
C:\!SCHOOL WORK\Project Building\demand-forecasting-decision-optimization\data\raw
```

That repository is reference-only and must not be modified. Copying the three raw files into this project's `data/raw/` in Milestone 2 keeps this project independently runnable. A symlink is not preferred because it would create a hidden runtime dependency on the old project.

The copies were verified again during the Milestone 2 pipeline run. They remain Git-ignored and the old repository remains read-only.
