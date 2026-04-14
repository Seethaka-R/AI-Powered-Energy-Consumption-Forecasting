"""Feature engineering helpers for the forecasting pipeline."""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, lag, rolling, and interaction features."""
    df = df.copy()
    df = _add_calendar_features(df)
    df = _add_cyclical_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_interaction_features(df)

    before = len(df)
    df.dropna(inplace=True)
    print(f"        Dropped {before - len(df)} rows (lag warm-up)")
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["day_of_year"] = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 18, 19, 20, 21]).astype(int)
    return df


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    target = "consumption_kwh"
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    target = "consumption_kwh"
    for window in [6, 24, 168]:
        shifted = df[target].shift(1)
        df[f"roll_mean_{window}h"] = shifted.rolling(window=window, min_periods=1).mean()
        df[f"roll_std_{window}h"] = shifted.rolling(window=window, min_periods=1).std()
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_c" in df.columns:
        df["temp_x_hour"] = df["temperature_c"] * df["hour"]
        df["temp_sq"] = df["temperature_c"] ** 2
    return df
