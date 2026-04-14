# Architecture

## System Goal

Forecast future hourly energy consumption from historical usage and weather-related variables.

## Component Flow

```text
Raw CSV or Synthetic Generator
            |
            v
      Data Loader
            |
            v
      Preprocessing
            |
            v
   Feature Engineering
            |
            v
   Random Forest Model
            |
      +-----+-----+
      |           |
      v           v
 Evaluation    30-Day Forecast
      |           |
      +-----+-----+
            v
   Visualizations and Saved Outputs
```

## Module Responsibilities

### `data_loader.py`

- Reads `data/raw/energy_data.csv` if present.
- Generates a synthetic two-year hourly dataset if the raw file is missing.

### `preprocessing.py`

- Converts numeric fields to efficient numeric types.
- Resamples to hourly frequency.
- Interpolates short gaps and fills longer ones.
- Caps outliers using an IQR-based rule.
- Smooths the target series slightly with a rolling average.

### `feature_engineering.py`

- Adds calendar features such as hour, weekday, month, and quarter.
- Encodes periodic variables with sine and cosine transforms.
- Creates lag features and rolling summary statistics.
- Builds interaction terms such as `temp_x_hour` and `temp_sq`.

### `model.py`

- Trains a `RandomForestRegressor` using a chronological train/test split.
- Evaluates predictions on the holdout window.
- Produces recursive multi-step forecasts for future timestamps.

### `visualization.py`

- Saves overview, prediction, forecast, feature importance, and residual plots.

### `utils.py`

- Ensures project directories exist.
- Saves and loads trained models.
- Provides shared project-root path handling.

## Data Inputs and Outputs

### Input

- Optional raw dataset: `data/raw/energy_data.csv`

### Processed Data

- `data/processed/energy_clean.csv`
- `data/processed/energy_features.csv`

### Model Artefacts

- `models/rf_energy_model.pkl`
- `outputs/metrics.txt`

### Forecast and Charts

- `outputs/forecast_30days.csv`
- `outputs/graphs/*.png`

## Current Gaps That Were Filled

- Missing lowercase modules required by `main.py`
- Empty utility module
- Empty project documentation
- Empty notebook placeholders
