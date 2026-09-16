# Milestone 1 — Simplified Project Design

## 1. Scope and business question

This standalone project will answer:

> Can we forecast short-term retail demand for 100 Walmart products, understand where the model performs well or poorly, and present the results in an interactive dashboard?

The analysis unit is one `item_id` on one calendar date. The target is nonnegative daily `units_sold`. The scope is one Walmart store (`CA_3`), one department (`FOODS_3`), and exactly 100 established products selected before model evaluation.

The project has only two forecasting methods: a seven-day seasonal-naive baseline and one global LightGBM model. The only headline accuracy metrics are MAE, RMSE, and WAPE.

## 2. Chosen forecasting formulation

### Rolling one-day-ahead prediction

For each date `t`, predict `units_sold` separately for every product. Features derived from sales must stop at `t-1`. Calendar features, SNAP status, event status, and the scheduled selling price for `t` are treated as known before the forecast is made.

This formulation was chosen instead of a fixed seven-day direct forecast because it:

- matches the plain-language question, “what will sell tomorrow?”;
- makes lag and rolling-feature timing visible and easy to test;
- avoids recursive predictions, seven horizon-specific models, and multi-horizon leakage rules;
- still evaluates many consecutive future days, products, and business conditions; and
- produces `100 products × 56 evaluation days = 5,600` planned out-of-sample item-day forecasts across validation and test, subject to successful data validation.

The operational simulation is straightforward: refresh history after each observed day, construct tomorrow's features, and predict. The LightGBM model is refit at the start of each evaluation week using all labels available through the preceding day. Within that week, the fitted model stays fixed while lag/rolling inputs update with newly observed sales. This is expanding-window, walk-forward evaluation—not a random split.

### Feature timing

Planned features and their plain-English meaning:

| Feature | Meaning | Availability rule |
|---|---|---|
| `lag_7` | Sales for the same product seven days ago | Observed before `t` |
| `lag_14` | Sales two weeks ago | Observed before `t` |
| `lag_28` | Sales four weeks ago | Observed before `t` |
| `rolling_mean_7` | Average sales over the previous 7 days | Compute after shifting sales by one day |
| `rolling_mean_28` | Average sales over the previous 28 days | Compute after shifting sales by one day |
| `rolling_std_28` | Recent 28-day sales variability | Compute after shifting sales by one day |
| `weekday` | Day of week being predicted | Known calendar value |
| `month` | Month being predicted | Known calendar value |
| `item_id` | Product identity so one global model can learn product differences | Known identifier |
| `sell_price` | Scheduled price for the item in the target M5 week | Assumed known at forecast time; this assumption will be stated |
| `snap` | Whether California SNAP purchasing benefits are active | Known M5 calendar value |
| `event_indicator` | Whether either M5 event field is populated | Known M5 calendar value |

The initial implementation will not add features unless a concrete validation or modeling need appears. In particular, it will not bring over long feature lists, horizon indicators, price histories, or availability-state machinery from the old project.

## 3. Product selection

Product selection will use only the first 730 days of the M5 history, well before validation and test:

1. Filter sales rows to `store_id == "CA_3"` and `dept_id == "FOODS_3"` before reshaping.
2. Keep products with at least 365 days between first positive sale and the early-history cutoff.
3. Require at least 28 positive-sales days in that early window.
4. Rank eligible products by number of positive-sales days, then total units, then `item_id` for a deterministic tie-break.
5. Select the first 100 products.

This is simpler than the old project's eligibility, price-coverage, candidate-comparison, and rank-spacing procedure. It defines “established” in observable terms and prevents final-period outcomes from deciding which products enter the project. Volume segments for error analysis will be assigned later using training-period average sales only.

## 4. Chronological validation design

The source sales history runs from 2011-01-29 through 2016-05-22, but the modeling panel intentionally stops on 2015-10-25. This keeps the project methodologically independent from prior work that already analyzed later 2016 outcomes.

| Role | Dates | Use |
|---|---|---|
| Development history | 2011-01-29 to 2015-08-30 | Initial training, EDA, and product segments |
| Validation week 1 | 2015-08-31 to 2015-09-06 | Walk-forward model assessment |
| Validation week 2 | 2015-09-07 to 2015-09-13 | Walk-forward model assessment |
| Validation week 3 | 2015-09-14 to 2015-09-20 | Walk-forward model assessment |
| Validation week 4 | 2015-09-21 to 2015-09-27 | Walk-forward model assessment |
| Final test | 2015-09-28 to 2015-10-25 | Four untouched weeks; evaluate once after choices are fixed |

At each Monday weekly origin, models train on data ending the previous day. Every daily forecast inside the week uses actual sales history only through the preceding day. Validation is used for reasonable LightGBM settings and implementation checks. Once those choices are fixed, the same weekly walk-forward procedure runs over the final four weeks without tuning on test results. Final-test targets are excluded from product selection, EDA conclusions, feature and cleaning decisions, and parameter choices; structural integrity checks may span all periods.

Metrics will be reported both pooled across each role and by week. “Improvement versus baseline” means `(baseline metric - LightGBM metric) / baseline metric`, with the metric named explicitly. The dashboard headline will use final-test pooled RMSE improvement unless the implemented evaluation reveals a reason to change it; any change must be documented before test results are used.

## 5. Error analysis plan

After final-test predictions exist, descriptive summaries will cover:

- product volume segment (low, medium, high, assigned from training data);
- weekday;
- test week;
- zero-demand versus positive-demand observations;
- products with the largest MAE and contribution to total absolute error; and
- LightGBM gain-based feature importance.

Slices with zero total actual demand will show WAPE as undefined rather than zero. Error slices describe associations and model behavior; they are not causal explanations.

## 6. What is reused from the older project

The older repository at `C:\!SCHOOL WORK\Project Building\demand-forecasting-decision-optimization` was inspected as read-only reference material. The new repository does not import it or depend on it.

Reusable ideas, to be rewritten simply:

- the three official M5 inputs and their expected schemas;
- filtering the wide sales file to `CA_3 / FOODS_3` before melting, which avoids expanding all M5 series;
- chunked reading of the large sales file;
- early-history-only established-product selection;
- basic checks for required columns, unique keys, continuous dates, nonnegative integer sales, positive prices, and valid joins;
- shifted lag/rolling calculations with explicit leakage tests;
- the standard MAE, RMSE, and WAPE formulas;
- consistent actual-versus-predicted plotting with units and dates labeled; and
- a dashboard that consumes saved outputs instead of retraining.

These are design patterns, not copied modules. The old project's numerical results, trained models, processed tables, and dashboard artifacts will not be reused.

## 7. What will be rebuilt in simpler form

- A small loader dedicated to three CSV files and one fixed store/department.
- A transparent 100-product selection function with three eligibility/ranking rules.
- One tidy daily panel rather than multiple locked forecasting datasets.
- A one-day-ahead feature builder based on `groupby`, `shift`, and `rolling`.
- One seasonal-naive predictor (`lag_7`).
- One global LightGBM training path with a small, fixed, documented parameter set.
- One walk-forward evaluator for validation and final test.
- Small saved tables for predictions, metrics, error slices, and feature importance.
- A three-page Streamlit app that reads those saved tables.

## 8. Explicit exclusions from the older project

None of the following will be carried over:

- direct 1–7-day multi-horizon forecasting or horizon-specific examples;
- additive ETS or any model other than seasonal naive and LightGBM;
- conformal methods, empirical calibration, prediction intervals, coverage analysis, or uncertainty reports;
- calibration periods used for interval construction;
- demand scenarios, residual scenario generation, or simulation;
- optimization, MILP, SciPy/HiGHS, capacity constraints, shortage/excess penalties, or inventory allocation;
- retrospective inventory or replenishment simulation;
- artifact admission manifests, file locking, phase locks, or immutable pipeline architecture;
- MASE, forecast bias as a headline metric, or model-selection metric gates;
- large LightGBM candidate grids and horizon-dependent training logic;
- price-age, completed-week price summaries, days-since-positive, zero-rate, or extended rolling features;
- causal claims, business-savings claims, or the old project's resume bullets/results;
- the old Streamlit app and its optimization/uncertainty pages; and
- MLflow, Docker, Airflow, Spark, cloud deployment, or deep learning.

## 9. Dashboard contract for Milestone 5

The app will contain exactly three user-facing pages:

1. **Overview:** dataset size, final-test LightGBM RMSE and WAPE, RMSE improvement over seasonal naive, and actual-versus-predicted sales.
2. **Forecast Explorer:** product selector, historical sales, final-test predicted versus actual sales, and error over time.
3. **Error Analysis:** error by volume segment and weekday, worst products, test-week results, zero/positive-demand results, and feature importance.

The app will load saved artifacts. Missing artifacts will produce a clear setup message, never trigger training.

## 10. Milestone boundaries

Milestone 1 creates only repository structure and design documentation. Milestone 2 will validate/copy the local raw files, build the scoped panel, perform EDA, and record data-quality findings. It will not train the baseline or LightGBM model.
