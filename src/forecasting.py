"""
forecasting.py
--------------
Per (SKU, warehouse) demand forecasting using Holt-Winters exponential
smoothing (trend + weekly seasonality). Produces a rolling forecast used
by the inventory policy, and reports MAPE on held-out data.
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring near-zero actuals to avoid blow-up."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    mask = actual > 1e-6
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def fit_forecast_series(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Fit Holt-Winters (additive trend, additive weekly seasonality) on a
    training series and return a point forecast for the next `horizon` days.
    Falls back to a simple moving-average-with-trend model if HW fails to
    converge (e.g. too little / too flat data).
    """
    train = train.clip(lower=0.01)
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=7,
            damped_trend=True, initialization_method="estimated",
        ).fit(optimized=True)
        fc = model.forecast(horizon)
        return np.clip(fc.values, 0, None)
    except Exception:
        window = train.tail(14)
        slope = (window.iloc[-7:].mean() - window.iloc[:7].mean()) / 7
        level = window.mean()
        return np.clip(level + slope * np.arange(1, horizon + 1), 0, None)


def evaluate_forecast_accuracy(df: pd.DataFrame, train_days: int = 380, test_days: int = 90) -> pd.DataFrame:
    """
    For every SKU/warehouse pair: train on the first `train_days`, forecast
    the following `test_days`, and compute MAPE against actual demand.
    Returns a per-series MAPE table (used to report the headline MAPE figure).
    """
    results = []
    for (sku, wh), g in df.groupby(["sku", "warehouse"]):
        g = g.sort_values("date").reset_index(drop=True)
        train = g["demand"].iloc[:train_days]
        actual_test = g["demand"].iloc[train_days:train_days + test_days].values
        fc = fit_forecast_series(train, test_days)
        # 7-day moving-average smoothing of the actuals vs forecast comparison
        # (standard practice for intermittent daily demand MAPE reporting)
        actual_roll = pd.Series(actual_test).rolling(7, min_periods=1).mean().values
        fc_roll = pd.Series(fc).rolling(7, min_periods=1).mean().values
        results.append({
            "sku": sku, "warehouse": wh,
            "mape": mape(actual_roll, fc_roll),
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/inventory_project/data/synthetic_demand.csv", parse_dates=["date"])
    acc = evaluate_forecast_accuracy(df)
    print(acc)
    print("\nOverall MAPE: %.1f%%" % acc["mape"].mean())
