"""
model.py
========
Handles model training, evaluation, and future forecasting.

Model choice – Random Forest Regressor (Option B: Intermediate)
---------------------------------------------------------------
* Handles non-linear patterns without scaling
* Naturally captures feature interactions
* Provides feature importance (explainability)
* Robust to outliers
* No stationarity requirement (unlike ARIMA)

The module also exposes a simple recursive multi-step forecast so we can
project 30 days into the future from the last known data point.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection   import train_test_split, TimeSeriesSplit
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from typing                    import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET       = "consumption_kwh"

# Columns that are NOT features (target + derived from target already in lags)
NON_FEATURES = {TARGET}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Split data, train a Random Forest model, and return all artefacts.

    We use a time-ordered 80/20 split (no shuffling!) because shuffling a
    time series would cause data leakage.

    Returns
    -------
    model, X_train, X_test, y_train, y_test, feature_cols
    """
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].values
    y = df[TARGET].values

    # Time-ordered split (last 20 % as test set)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"      Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ── Train ──────────────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 15,
        min_samples_leaf = 4,
        max_features     = "sqrt",
        n_jobs           = 1,
        random_state     = 42
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[dict, np.ndarray]:
    """
    Compute RMSE, MAE, R², and MAPE on the held-out test set.

    Returns
    -------
    metrics : dict
    y_pred  : np.ndarray of predictions
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = _mape(y_test, y_pred)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    # Also save a metrics summary
    _save_metrics(metrics)

    return metrics, y_pred


def forecast_future(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    days: int = 30
) -> pd.DataFrame:
    """
    Recursive multi-step forecast for the next *days* × 24 hours.

    Strategy
    --------
    For each future hour we:
      1. Build a feature row using the newly predicted values as lags.
      2. Predict consumption.
      3. Append the prediction so subsequent steps can use it.

    This mirrors how companies run day-ahead and week-ahead forecasts.
    """
    # Work from a copy so we don't mutate the original
    history = df[TARGET].copy()
    last_ts = df.index[-1]

    future_index  = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=days * 24,
        freq="h"
    )

    predictions = []

    # We need the full feature frame to build future rows, so we'll simulate
    # the feature engineering for each new step using the extended series.
    temp_series = _extend_temperature(df, days)
    hum_series  = _extend_humidity(df, days)

    for ts in future_index:
        row = _build_future_row(ts, history, temp_series, hum_series)
        X_row = np.array([[row[c] for c in feature_cols]])
        pred  = float(model.predict(X_row)[0])
        pred  = max(0, pred)               # consumption cannot be negative
        history.loc[ts] = pred
        predictions.append({"timestamp": ts, "predicted_kwh": round(pred, 2)})

    forecast_df = pd.DataFrame(predictions).set_index("timestamp")
    return forecast_df


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list:
    """Return all columns that are not the target."""
    return [c for c in df.columns if c not in NON_FEATURES]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids divide-by-zero)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _save_metrics(metrics: dict):
    """Write metrics to a text file for GitHub proof."""
    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.txt", "w") as f:
        f.write("=" * 40 + "\n")
        f.write("  Model Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<6}: {v:.4f}\n")
        f.write("=" * 40 + "\n")


def _extend_temperature(df: pd.DataFrame, days: int) -> pd.Series:
    """
    Project temperature for future periods by repeating the last 30-day cycle.
    In production this would come from a weather API.
    """
    window = min(30 * 24, len(df))
    tail   = df["temperature_c"].iloc[-window:].values
    needed = days * 24
    repeats = (needed // len(tail)) + 2
    extended = np.tile(tail, repeats)[:needed]
    future_idx = pd.date_range(
        df.index[-1] + pd.Timedelta(hours=1),
        periods=needed, freq="h"
    )
    return pd.Series(extended, index=future_idx)


def _extend_humidity(df: pd.DataFrame, days: int) -> pd.Series:
    window = min(30 * 24, len(df))
    tail   = df["humidity_pct"].iloc[-window:].values
    needed = days * 24
    repeats = (needed // len(tail)) + 2
    extended = np.tile(tail, repeats)[:needed]
    future_idx = pd.date_range(
        df.index[-1] + pd.Timedelta(hours=1),
        periods=needed, freq="h"
    )
    return pd.Series(extended, index=future_idx)


def _build_future_row(
    ts: pd.Timestamp,
    history: pd.Series,
    temp_series: pd.Series,
    hum_series:  pd.Series
) -> dict:
    """Build a single feature row for a future timestamp."""
    hour        = ts.hour
    dow         = ts.dayofweek
    month       = ts.month
    quarter     = ts.quarter
    doy         = ts.dayofyear
    woy         = ts.isocalendar()[1]
    is_weekend  = int(dow >= 5)
    is_peak     = int(hour in [8, 9, 10, 18, 19, 20, 21])

    # Cyclical
    hour_sin   = np.sin(2 * np.pi * hour  / 24)
    hour_cos   = np.cos(2 * np.pi * hour  / 24)
    dow_sin    = np.sin(2 * np.pi * dow   / 7)
    dow_cos    = np.cos(2 * np.pi * dow   / 7)
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)

    # Lags (from the growing history series)
    def safe_lag(h):
        try:
            return float(history.iloc[-h])
        except IndexError:
            return float(history.mean())

    lag_1h  = safe_lag(1)
    lag_2h  = safe_lag(2)
    lag_3h  = safe_lag(3)
    lag_6h  = safe_lag(6)
    lag_12h = safe_lag(12)
    lag_24h = safe_lag(24)
    lag_48h = safe_lag(48)
    lag_168h= safe_lag(168)

    # Rolling
    roll_mean_6h   = float(history.iloc[-6:].mean())   if len(history) >= 6   else float(history.mean())
    roll_mean_24h  = float(history.iloc[-24:].mean())  if len(history) >= 24  else float(history.mean())
    roll_mean_168h = float(history.iloc[-168:].mean()) if len(history) >= 168 else float(history.mean())
    roll_std_6h    = float(history.iloc[-6:].std())    if len(history) >= 6   else 1.0
    roll_std_24h   = float(history.iloc[-24:].std())   if len(history) >= 24  else 1.0
    roll_std_168h  = float(history.iloc[-168:].std())  if len(history) >= 168 else 1.0

    # External
    temp = float(temp_series.get(ts, 25.0))
    hum  = float(hum_series.get(ts,  60.0))

    return {
        "temperature_c"  : temp,
        "humidity_pct"   : hum,
        "hour"           : hour,
        "day_of_week"    : dow,
        "month"          : month,
        "quarter"        : quarter,
        "day_of_year"    : doy,
        "week_of_year"   : woy,
        "is_weekend"     : is_weekend,
        "is_peak_hour"   : is_peak,
        "hour_sin"       : hour_sin,
        "hour_cos"       : hour_cos,
        "dow_sin"        : dow_sin,
        "dow_cos"        : dow_cos,
        "month_sin"      : month_sin,
        "month_cos"      : month_cos,
        "lag_1h"         : lag_1h,
        "lag_2h"         : lag_2h,
        "lag_3h"         : lag_3h,
        "lag_6h"         : lag_6h,
        "lag_12h"        : lag_12h,
        "lag_24h"        : lag_24h,
        "lag_48h"        : lag_48h,
        "lag_168h"       : lag_168h,
        "roll_mean_6h"   : roll_mean_6h,
        "roll_mean_24h"  : roll_mean_24h,
        "roll_mean_168h" : roll_mean_168h,
        "roll_std_6h"    : roll_std_6h,
        "roll_std_24h"   : roll_std_24h,
        "roll_std_168h"  : roll_std_168h,
        "temp_x_hour"    : temp * hour,
        "temp_sq"        : temp ** 2,
    }
