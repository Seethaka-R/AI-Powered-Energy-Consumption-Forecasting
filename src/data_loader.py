"""Load raw energy data or generate a synthetic fallback dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_or_generate_dataset(
    filepath: str | Path = "data/raw/energy_data.csv",
    generate_if_missing: bool = True,
) -> pd.DataFrame:
    """Load a dataset from disk, or generate one if it is missing."""
    path = Path(filepath)

    if path.exists():
        print(f"      Reading existing file: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    if not generate_if_missing:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Set generate_if_missing=True to create one."
        )

    print("      File not found - generating synthetic dataset...")
    df = _generate_synthetic_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"      Synthetic dataset saved -> {path}")
    return df


def _generate_synthetic_data(
    start: str = "2022-01-01",
    end: str = "2023-12-31 23:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly energy-consumption data."""
    np.random.seed(seed)
    idx = pd.date_range(start=start, end=end, freq="h")
    n = len(idx)

    hours = idx.hour.values
    day_of_week = idx.dayofweek.values
    day_of_year = idx.dayofyear.values

    temp_base = 25 + 13 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    temp_daily = 4 * np.sin(2 * np.pi * (hours - 6) / 24)
    temperature = temp_base + temp_daily + np.random.normal(0, 1.5, n)
    temperature = np.clip(temperature, 5, 45)

    humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    humidity += np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 100)

    base = np.full(n, 200.0)
    daily = (
        60 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)
        + 80 * np.exp(-0.5 * ((hours - 20) / 3) ** 2)
    )
    weekend_mask = (day_of_week >= 5).astype(float)
    weekly = -30 * weekend_mask
    seasonal = (
        50 * np.abs(np.sin(2 * np.pi * (day_of_year - 15) / 365))
        + 30 * np.abs(np.cos(2 * np.pi * (day_of_year - 15) / 365))
    )
    temp_effect = 2 * (temperature - 22) ** 2 / 10

    consumption = base + daily + weekly + seasonal + temp_effect
    consumption += np.random.lognormal(0, 0.05, n) * 10

    n_spikes = int(0.005 * n)
    spike_idx = np.random.choice(n, n_spikes, replace=False)
    consumption[spike_idx] *= np.random.uniform(1.5, 2.5, n_spikes)
    consumption = np.clip(consumption, 50, 900)

    df = pd.DataFrame(
        {
            "consumption_kwh": np.round(consumption, 2),
            "temperature_c": np.round(temperature, 2),
            "humidity_pct": np.round(humidity, 1),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
