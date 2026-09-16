# Evidence-based resume bullets

- Built a leakage-safe, one-day-ahead retail forecasting pipeline for 100 Walmart M5 products, engineering 12 interpretable calendar, price, event, SNAP, lag, and rolling features across 173,100 product-days.
- Trained and evaluated one global LightGBM model with weekly walk-forward refits and an untouched four-week test, producing 2,800 final forecasts without using future sales information.
- Reduced final-test error versus a weekly seasonal-naive benchmark by 21.90% on MAE/WAPE and 26.66% on RMSE; LightGBM beat the baseline RMSE in all four test weeks.
- Delivered a three-page Streamlit dashboard for aggregate results, product-level forecast inspection, and error diagnostics by volume, weekday, demand status, and feature importance.
