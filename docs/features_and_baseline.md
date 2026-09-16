# Milestone 3 — Leakage-Safe Features and Seasonal Baseline

## Forecasting task

The project makes rolling one-day-ahead forecasts. For a target date `t`, a predictor may use known product, calendar, SNAP, event, and scheduled-price information for `t`, plus observed sales only through `t-1`.

The Milestone 3 pipeline uses a Parquet predicate to load only development and validation rows. Test rows never enter the process, and no final-test prediction or metric exists.

## Model inputs

The locked model feature list for Milestone 4 is:

1. `item_id`
2. `weekday`
3. `month`
4. `snap`
5. `event_indicator`
6. `sell_price`
7. `lag_7`
8. `lag_14`
9. `lag_28`
10. `rolling_mean_7`
11. `rolling_mean_28`
12. `rolling_std_28`

`date`, `period`, and `volume_segment` are reporting metadata, not model inputs. `units_sold` is the target.

Target-date selling price is assumed to be scheduled and known when the forecast is issued. Missing prices remain `NaN`; they are not imputed. No scaling is applied because tree models do not require standardized inputs.

## Feature timing

Lags are calculated independently for each product:

- `lag_7` is sales at `t-7`;
- `lag_14` is sales at `t-14`; and
- `lag_28` is sales at `t-28`.

Rolling features first apply `shift(1)` to each product's sales. They then summarize the 7 or 28 observations ending at `t-1`. This ordering matters: rolling directly over unshifted `units_sold` would include the answer for `t` in its own predictors.

`rolling_std_28` uses population standard deviation (`ddof=0`) over `t-28` through `t-1`.

## Feature eligibility and sanity

The first 28 dates per product do not have all six history features and are left missing. They are not imputed or included in the model-ready table.

| Check | Result |
|---|---:|
| Development rows before eligibility | 167,500 |
| Development rows after eligibility | 164,700 |
| First eligible date | 2011-02-26 |
| Eligible validation rows | 2,800 |
| Missing validation history features | 0 |
| Missing validation prices | 0 |
| Missing prices in all model-ready rows | 42 |

Startup missingness matches the intended windows: 700 rows for `lag_7` and `rolling_mean_7`, 1,400 for `lag_14`, and 2,800 for each 28-day feature. All non-missing history values are finite and nonnegative. The complete development-plus-validation feature calculation contains the same 147 missing prices reported in Milestone 2; applying the 28-day history eligibility rule leaves 42 of those rows in the saved table.

## Seasonal-naive baseline

The only baseline is:

```text
prediction(item, t) = units_sold(item, t - 7) = lag_7
```

This is appropriate for daily retail demand with visible weekly seasonality: a Sunday is compared with the previous Sunday, so weekday effects are preserved without fitted parameters.

The baseline is evaluated only on the four validation weeks from 2015-08-31 through 2015-09-27. It produces exactly 2,800 forecasts: 100 products × 28 days.

## Metrics

- **MAE** is the mean absolute difference between actual and predicted units. It describes the typical absolute miss.
- **RMSE** is the square root of mean squared error. Squaring makes large misses more influential.
- **WAPE** is total absolute error divided by total actual demand. It expresses aggregate error relative to observed volume. WAPE is undefined when a slice has zero actual demand; the implementation returns `NaN`, not zero.

## Overall validation result

| Forecasts | MAE | RMSE | WAPE |
|---:|---:|---:|---:|
| 2,800 | 4.6568 | 9.3735 | 50.83% |

These validation results establish the benchmark that LightGBM must improve upon. They are not final-test results.

## By validation week

| Week | Dates | MAE | RMSE | WAPE |
|---:|---|---:|---:|---:|
| 1 | Aug 31–Sep 6 | 3.7886 | 6.9863 | 47.27% |
| 2 | Sep 7–Sep 13 | 4.9543 | 9.8748 | 48.76% |
| 3 | Sep 14–Sep 20 | 5.5871 | 11.1740 | 61.00% |
| 4 | Sep 21–Sep 27 | 4.2971 | 8.9594 | 46.14% |

Week 3 is the weakest on all three metrics. The variation across only four weeks cautions against judging a model from one aggregate number.

## By development-defined volume segment

| Segment | Forecasts | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|
| Low | 952 | 1.7279 | 2.4893 | 76.23% |
| Medium | 924 | 2.8387 | 5.5443 | 66.04% |
| High | 924 | 9.4924 | 15.1369 | 44.93% |

High-volume products have the largest errors in units and dominate RMSE. Low-volume products have smaller absolute errors but the worst demand-relative WAPE. Milestone 4 should retain both absolute and relative metrics rather than claiming one segment is uniformly easier.

## By weekday

| Weekday | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Monday | 4.8350 | 9.3616 | 53.45% |
| Tuesday | 4.7500 | 10.2650 | 55.38% |
| Wednesday | 4.8300 | 9.2550 | 58.00% |
| Thursday | 3.8550 | 7.5611 | 48.22% |
| Friday | 4.4425 | 8.6381 | 51.69% |
| Saturday | 5.3225 | 10.9746 | 50.37% |
| Sunday | 4.5625 | 9.1729 | 41.39% |

By the primary RMSE metric, Thursday is strongest and Saturday is weakest. By WAPE, Sunday is strongest and Wednesday is weakest. These are descriptive validation-period comparisons, not stable causal weekday effects.

## Zero versus positive demand

| Actual-demand group | Forecasts | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|
| Positive | 2,160 | 5.6014 | 10.4102 | 47.17% |
| Zero | 640 | 1.4688 | 4.3178 | Undefined |

The baseline sometimes predicts positive sales on zero-demand days. WAPE cannot be computed for the all-zero slice because its denominator is zero.

## Leakage verification

Automated tests verify:

- exact `t-7`, `t-14`, and `t-28` lag values;
- exact rolling windows ending at `t-1`;
- target-day and future-sales perturbations do not change date `t` features;
- changing one product cannot affect another product's history features;
- shuffled input produces identical sorted features;
- the first eligible date follows exactly 28 prior observations;
- feature values contain no infinity;
- baseline predictions equal `lag_7` exactly;
- the validation prediction count is exactly 2,800; and
- final-test dates are absent from predictions and `model_features.parquet`.

All 23 project tests pass.

## Saved artifacts

- `data/processed/model_features.parquet`: 164,700 eligible development rows plus 2,800 validation rows; no test rows.
- `outputs/baseline/validation_predictions.parquet`: compact validation predictions and errors.
- `outputs/baseline/validation_metrics.csv`: overall and requested breakdown metrics.
- `outputs/baseline/summary.json`: feature eligibility, sanity, metrics, and artifact checks.
- `outputs/baseline/figures/`: three validation-only figures.

## Limitations and Milestone 4 implications

- Seasonal naive cannot adapt to level changes, events, SNAP, or price differences beyond what happened seven days earlier.
- Validation performance varies substantially by week and product volume.
- A simple global LightGBM should use the locked 12-feature list, native missing values, and categorical product/weekday handling. It should be compared against the same 2,800 validation rows with the same metric functions.
- High-volume items are likely to dominate RMSE, while low-volume items remain difficult under WAPE. Both views should remain visible.
- Validation may guide model choices; the final test must remain untouched until the simple LightGBM design is fixed.
