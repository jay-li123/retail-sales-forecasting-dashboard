# Milestone 2 — Data Quality and Development-Period EDA

## Result

The three local Walmart M5 source files passed checksum, schema, key, ordering, and value-domain validation. The pipeline selected 100 established `CA_3 / FOODS_3` products using only the first 730 sales days, then built a complete daily panel from 2011-01-29 through 2015-10-25.

No forecasting model was trained. All sales-based EDA and volume segmentation stop on the development boundary, 2015-08-30. Validation and final-test targets were stored for later evaluation but were not summarized or used to make data or feature decisions.

## Sources and integrity

| Source | Rows | Columns | SHA-256 verified |
|---|---:|---:|---|
| `calendar.csv` | 1,969 | 14 | Yes |
| `sales_train_evaluation.csv` | 30,490 | 1,947 | Yes |
| `sell_prices.csv` | 6,841,121 | 4 | Yes |

The hashes exactly match the values recorded in Milestone 1. Validation confirmed continuous ordered calendar dates and matching `d` labels, consecutive sales day columns, unique sales series, numeric nonnegative integer sales, unique store/item/week price keys, and positive finite recorded prices. Structural failures raise errors; the pipeline does not repair the source silently.

The wide sales source was filtered to `store_id = CA_3` and `dept_id = FOODS_3` before reshaping. The scoped source contained 823 candidate products.

## Product selection

Selection used only `d_1` through `d_730` (2011-01-29 through 2013-01-27):

1. Require at least 365 inclusive days from the first positive sale through the selection cutoff.
2. Require at least 28 positive-sales days in the early window.
3. Rank eligible products by positive-sales days descending, total early units descending, then `item_id` ascending.
4. Keep the first 100.

Of 823 candidates, 483 were eligible. The deterministic result and its early-history metadata are saved in `data/processed/selected_items.csv`. No validation or test outcome participates in the rules or ranking.

## Processed panel

| Property | Result |
|---|---:|
| Shape | 173,100 rows × 21 columns |
| Products | 100 |
| Date range | 2011-01-29 to 2015-10-25 |
| Store / department | `CA_3 / FOODS_3` |
| Development rows | 167,500 |
| Validation rows | 2,800 |
| Test rows | 2,800 |

There is exactly one row for every selected `item_id × date`. Every product has the same 1,731 continuous dates. Calendar joins are complete, target values are nonnegative integers, and the many-to-one price join preserves the expected row count.

Period boundaries are:

- development: 2011-01-29 through 2015-08-30;
- validation: 2015-08-31 through 2015-09-27; and
- final test: 2015-09-28 through 2015-10-25.

Raw source dates after 2015-10-25 remain available but are excluded from this modeling panel.

## Price missingness

The panel preserves missing prices rather than filling them:

- 147 missing-price rows, or 0.0849% of the complete panel;
- all 147 occur before the affected product's first recorded price;
- zero missing-price rows occur on or after the first recorded price; and
- zero development-period positive-sales rows have a missing price.

This is small, concentrated pre-selling missingness. The positive-sales check deliberately uses development targets only; test targets remain uninspected. No availability-state system or price imputation is warranted for Milestone 2. LightGBM can retain missing values later; the seasonal baseline will not use price.

For development-period recorded prices, the median is $2.28 and the interquartile range is $1.58–$3.28. These are repeated item-day prices, not a product-level price distribution.

## Development-only volume segments

Each product's average daily sales was calculated using development rows only. Products were sorted by average daily sales and `item_id`; deterministic rank positions were split into approximately equal tertiles:

| Segment | Products | Average item-day units within segment |
|---|---:|---:|
| Low | 34 | 2.94 |
| Medium | 33 | 6.39 |
| High | 33 | 22.16 |

The saved metadata contains average development sales, development total units, deterministic volume rank, and segment. Tests confirm that replacing every validation/test target with an extreme value leaves the segments unchanged.

## Development-period findings

- Development contains 167,500 item-days and 1,745,281 total units, averaging 10.42 units per item-day.
- Zero demand occurs on 12.07% of development item-days. Among positive-sales rows, the median is 6 units, the 90th percentile is 28, and the 99th percentile is 81, so the distribution is strongly right-skewed.
- Sunday has the highest descriptive weekday average at 12.58 units per item-day; Wednesday is lowest at 9.20. This supports including weekday information later but does not by itself establish a causal weekday effect.
- `FOODS_3_586` is the highest-development-volume selected product with 117,731 units. Product totals are concentrated enough that both pooled metrics and product-level error analysis will matter.
- Recorded event days average 10.13 units versus 10.45 on non-event days. SNAP days average 11.24 versus 10.02 on non-SNAP days. These are unadjusted associations with seasonality, weekday, product mix, and time trends left uncontrolled.
- Median total daily sales across the 100 products is 1,011 units, with an interquartile range of 858–1,209. The time series shows recurring weekly variation, longer shifts, spikes, and a few near-zero holiday closure days.

No validation- or test-period sales pattern is included in these findings.

## Figures

Six Matplotlib figures are saved under `outputs/eda/figures/`:

1. `01_daily_sales.png`
2. `02_weekday_sales.png`
3. `03_sales_distribution.png`
4. `04_volume_segments.png`
5. `05_sell_price.png`
6. `06_event_snap_comparison.png`

All target-based figures use development rows only.

## Assumptions and limitations

- Recorded sales are treated as the forecasting target, although M5 does not identify lost demand caused by stockouts.
- A missing price is left as missing; it is not interpreted as definitive proof that an item was unavailable.
- Calendar event and SNAP comparisons are descriptive, not causal.
- Selection favors products with frequent early positive sales. Results will describe this established-product sample, not all M5 products.
- The future modeling design assumes the target day's M5 price and calendar attributes are known when a one-day-ahead forecast is issued. That timing assumption will be tested and restated in Milestone 3.

## Reproduction

From the repository root with dependencies installed:

```powershell
$env:PYTHONPATH = "src"
python -m retail_forecasting.data_pipeline
python -m unittest discover -s tests -v
```

The pipeline rewrites the generated processed/EDA artifacts deterministically. Raw, processed, and output files remain Git-ignored.
