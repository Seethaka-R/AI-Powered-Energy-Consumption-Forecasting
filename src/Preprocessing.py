"""Preprocessing utilities for energy time-series data."""

import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = df.copy()

    print("      -> Checking dtypes...")
    df = _ensure_dtypes(df)

    print("      -> Filling gaps (resample to 1h)...")
    df = _fill_time_gaps(df)

    print("      -> Handling missing values...")
    before_nan = int(df.isnull().sum().sum())
    df = _handle_missing(df)
    after_nan = int(df.isnull().sum().sum())
    print(f"        NaNs before: {before_nan}  |  after: {after_nan}")

    print("      -> Capping outliers (IQR x 3)...")
    capped, df = _cap_outliers(df, col="consumption_kwh", iqr_factor=3.0)
    print(f"        Outliers capped: {capped}")

    print("      -> Applying 3-hour rolling smooth...")
    return _smooth(df, col="consumption_kwh", window=3)


def _ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].astype(np.float32)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _fill_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("h").mean()


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.interpolate(method="time", limit=3)
    return df.ffill().bfill()


def _cap_outliers(
    df: pd.DataFrame,
    col: str = "consumption_kwh",
    iqr_factor: float = 3.0,
) -> tuple[int, pd.DataFrame]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr

    mask = (df[col] < lower) | (df[col] > upper)
    df[col] = df[col].clip(lower=lower, upper=upper)
    return int(mask.sum()), df


def _smooth(df: pd.DataFrame, col: str = "consumption_kwh", window: int = 3) -> pd.DataFrame:
    df[col] = (
        df[col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .round(2)
    )
    return df
