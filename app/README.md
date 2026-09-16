# Dashboard architecture

The Streamlit layer is intentionally read-only. `app.py` defines the three pages, `data.py` loads and reconciles saved final-test artifacts, and `charts.py` contains Plotly presentation logic. No training code is imported or executed when the app starts.

Required inputs are read from `outputs/model/`:

- `final_test_predictions.parquet`
- `final_test_metrics.csv`
- `product_error_summary.csv`
- `representative_errors.csv`
- `feature_importance.csv`
- `error_analysis.csv`
- `summary.json`

From the repository root, launch with:

```powershell
streamlit run app/app.py
```

If an input is missing, the app reports the exact missing files and stops. Generate the artifacts through the documented analytical pipeline; the dashboard never silently retrains or reconstructs results.
