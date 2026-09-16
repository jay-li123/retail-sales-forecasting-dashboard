# Milestone 4 — Global LightGBM and Final Evaluation

## Result

A single global LightGBM model improved validation RMSE over seasonal naive, so the validation-only decision rule selected LightGBM before final-test targets were loaded. On the untouched four-week final test, LightGBM achieved RMSE 6.0829 versus 8.2942 for seasonal naive, a 26.66% relative reduction.

This is a retrospective forecasting result for one store, one department, and 100 established products. It is not a production-impact or causal claim.

## What “global model” means

One LightGBM model is trained across eligible daily rows from all 100 products. `item_id` tells the model which product a row represents, while shared lag, rolling, calendar, price, event, and SNAP patterns can be learned across the product set. This is simpler to operate and explain than fitting 100 independent models.

The categorical predictors are `item_id`, `weekday`, and `month`. They use deterministic native LightGBM categories: sorted item identifiers, Monday–Sunday weekday order, and months 1–12. There is no target encoding or one-hot expansion.

## Fixed configuration

No grid search or validation-driven parameter tuning was performed.

| Parameter | Value |
|---|---:|
| Objective | regression |
| Metric | RMSE |
| Learning rate | 0.05 |
| Boosting rounds (`n_estimators`) | 300 |
| Leaves | 31 |
| Minimum child samples | 50 |
| Column sample / feature fraction | 0.9 |
| L2 regularization | 1.0 |
| Random seed | 42 |
| Threads | all available (`-1`) |

The installed environment does not include scikit-learn, so the implementation uses LightGBM’s native `train` API with the same configuration and 300 boosting rounds. Predictions below zero are clipped to zero; no other transformation is applied.

The feature list remains exactly the 12 inputs locked in Milestone 3.

## Weekly walk-forward procedure

For each validation or test week:

1. Use eligible labeled rows strictly before Monday as training data.
2. Fit the global model once.
3. Keep the fitted model fixed through Sunday.
4. Predict each day from target-date known attributes and actual sales history through the previous day.
5. Allow an observed day to enter later one-day-ahead features.
6. Refit only at the next Monday boundary.

The saved fit audit contains four validation and four test fits. Every `training_end` precedes its `forecast_start`.

## Validation comparison and model decision

Validation covers 2,800 identical item-days for both models.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal naive | 4.6568 | 9.3735 | 50.83% |
| LightGBM | **3.4724** | **6.7940** | **37.90%** |
| Relative LightGBM improvement | **25.43%** | **27.52%** | **25.43%** |

LightGBM improved RMSE in every validation week:

| Validation week | Seasonal-naive RMSE | LightGBM RMSE | LightGBM WAPE |
|---:|---:|---:|---:|
| 1 | 6.9863 | 4.9884 | 38.60% |
| 2 | 9.8748 | 7.7522 | 38.26% |
| 3 | 11.1740 | 7.8894 | 40.80% |
| 4 | 8.9594 | 6.1165 | 34.06% |

The locked rule selects LightGBM only when its pooled validation RMSE is lower. It was, so `lightgbm` became the final model. Fifty-one negative validation predictions were clipped to zero.

`outputs/model/model_specification.json` recorded the features, categories, parameters, validation metrics, seed, selection rule, and selected model before final-test scoring began.

## Final-test performance

The test covers 2,800 item-days from 2015-09-28 through 2015-10-25.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal naive | 4.0893 | 8.2942 | 50.97% |
| LightGBM | **3.1939** | **6.0829** | **39.81%** |
| Relative LightGBM improvement | **21.90%** | **26.66%** | **21.90%** |

These are final-test results observed only after the model specification was written. No parameters or features were changed afterward. Ninety-five negative LightGBM test predictions were clipped to zero.

### By test week

| Week | LightGBM MAE | LightGBM RMSE | LightGBM WAPE | Seasonal-naive RMSE |
|---:|---:|---:|---:|---:|
| 1 | 3.3243 | 6.7507 | 40.67% | 8.2974 |
| 2 | 3.6156 | 6.5436 | 44.08% | 10.1993 |
| 3 | 3.0376 | 5.6639 | 37.61% | 7.5236 |
| 4 | 2.7980 | 5.2473 | 36.62% | 6.7602 |

LightGBM improved RMSE in all four test weeks. Week 2 was weakest by MAE and WAPE; week 1 had the highest LightGBM RMSE.

## Error analysis for selected LightGBM model

### Volume segments

| Segment | Rows | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|
| Low | 952 | 1.2677 | 1.7361 | 63.62% |
| Medium | 924 | 2.2953 | 4.2775 | 54.24% |
| High | 924 | 6.0770 | 9.5248 | 33.71% |

High-volume products produce the largest unit errors, while low-volume products remain hardest relative to their observed demand.

### Weekdays

By RMSE, Tuesday was strongest at 4.9643 and Saturday weakest at 6.8376. By WAPE, Saturday was strongest at 36.72% and Wednesday weakest at 46.62%. These associations are specific to four test weeks and do not establish weekday effects.

### Zero and positive demand

| Actual-demand group | Rows | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|
| Positive | 2,043 | 3.6695 | 6.6638 | 33.37% |
| Zero | 757 | 1.9101 | 4.1250 | Undefined |

WAPE is undefined for the all-zero slice. Positive predictions on zero-demand rows remain an important dashboard diagnostic.

### Highest-error products

| Item | Segment | Actual total | Prediction total | MAE | RMSE | WAPE | Absolute-error share |
|---|---|---:|---:|---:|---:|---:|---:|
| FOODS_3_681 | High | 2,332 | 2,065.5 | 24.3862 | 28.5315 | 29.28% | 7.64% |
| FOODS_3_086 | Medium | 1,172 | 1,125.4 | 14.4584 | 19.2536 | 34.54% | 4.53% |
| FOODS_3_586 | High | 2,176 | 2,133.9 | 13.5721 | 15.2707 | 17.46% | 4.25% |
| FOODS_3_498 | High | 476 | 477.6 | 12.0181 | 16.2926 | 70.69% | 3.76% |
| FOODS_3_252 | High | 1,352 | 1,314.3 | 10.2181 | 12.5802 | 21.16% | 3.20% |
| FOODS_3_541 | High | 348 | 486.2 | 9.2854 | 15.8301 | 74.71% | 2.91% |
| FOODS_3_804 | High | 752 | 740.3 | 9.2043 | 11.8402 | 34.27% | 2.88% |
| FOODS_3_587 | High | 522 | 580.7 | 8.8072 | 11.6726 | 47.24% | 2.76% |
| FOODS_3_635 | High | 227 | 349.9 | 7.2478 | 11.0578 | 89.40% | 2.27% |
| FOODS_3_714 | High | 899 | 873.6 | 6.6269 | 8.0119 | 20.64% | 2.07% |

The same ten products lead both MAE and total absolute-error contribution in this test. Their error scales differ substantially, so MAE and WAPE tell complementary stories.

## Representative cases

- Largest overprediction: `FOODS_3_681`, 2015-09-30, actual 6, predicted 55.50, error +49.50.
- Largest underprediction: `FOODS_3_681`, 2015-10-04, actual 113, predicted 59.05, error −53.95.
- Closest positive-demand prediction: `FOODS_3_714`, 2015-10-23, actual 31, predicted 31.003.
- Highest-error product example: `FOODS_3_681` on 2015-10-04, the largest underprediction above.
- Low-volume example: `FOODS_3_460`, 2015-10-12, actual 13, predicted 3.70, error −9.30.

These cases include successes and failures for context; they were not used to revise the model.

## Gain feature importance

Gain importance is aggregated over the four weekly test fits. It describes how LightGBM allocated split gain; it does not identify causal business drivers.

| Feature | Gain share |
|---|---:|
| rolling_mean_7 | 87.06% |
| rolling_mean_28 | 4.72% |
| weekday | 2.19% |
| item_id | 1.16% |
| lag_28 | 0.99% |
| lag_7 | 0.93% |
| lag_14 | 0.90% |
| month | 0.75% |
| rolling_std_28 | 0.60% |
| snap | 0.39% |
| sell_price | 0.29% |
| event_indicator | 0.03% |

The model used `rolling_mean_7` heavily for gain in these fits. Correlated history features can divide or concentrate importance unpredictably, so these percentages should not be treated as isolated effects.

## Limitations

- The evaluation covers one store/department and eight total evaluation weeks.
- Recorded sales may differ from latent demand when stockouts occur.
- The fixed configuration is intentionally reasonable rather than optimized.
- Weekly refitting and actual prior-day history describe a one-day-ahead operating process; results do not apply directly to a fixed multi-day forecast.
- Test performance is one retrospective sample and is not evidence of deployment impact.
