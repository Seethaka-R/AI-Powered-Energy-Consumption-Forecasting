"""Plotting utilities for the forecasting workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def plot_raw_data(df: pd.DataFrame, output_dir: str | Path = "outputs/graphs", save: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Energy Dataset Overview", fontsize=16, fontweight="bold", y=1.01)

    sample = df["consumption_kwh"].iloc[: 30 * 24]
    axes[0, 0].plot(sample.index, sample.values, lw=0.8, color="#2196F3")
    axes[0, 0].set_title("Hourly Consumption (first 30 days)")
    axes[0, 0].set_ylabel("kWh")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0, 0].tick_params(axis="x", rotation=30)

    daily = df["consumption_kwh"].resample("D").sum()
    axes[0, 1].fill_between(daily.index, daily.values, alpha=0.5, color="#4CAF50")
    axes[0, 1].plot(daily.index, daily.values, lw=0.8, color="#388E3C")
    axes[0, 1].set_title("Daily Total Consumption")
    axes[0, 1].set_ylabel("kWh / day")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0, 1].tick_params(axis="x", rotation=30)

    if "temperature_c" in df.columns:
        sample_sc = df.sample(min(3000, len(df)), random_state=42)
        axes[1, 0].scatter(
            sample_sc["temperature_c"],
            sample_sc["consumption_kwh"],
            alpha=0.2,
            s=5,
            color="#FF5722",
        )
        axes[1, 0].set_title("Temperature vs Consumption")
        axes[1, 0].set_xlabel("Temperature (C)")
        axes[1, 0].set_ylabel("Consumption (kWh)")

    axes[1, 1].hist(
        df["consumption_kwh"].dropna(),
        bins=60,
        color="#9C27B0",
        edgecolor="white",
        alpha=0.8,
    )
    axes[1, 1].set_title("Consumption Distribution")
    axes[1, 1].set_xlabel("kWh")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    _save(fig, Path(output_dir), "01_raw_data_overview.png", save)


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    index,
    output_dir: str | Path = "outputs/graphs",
    save: bool = True,
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Model Performance: Actual vs Predicted", fontsize=14, fontweight="bold")

    n = min(168, len(y_true))
    idx = list(index)[:n]
    axes[0].plot(idx, y_true[:n], label="Actual", lw=1.2, color="#2196F3")
    axes[0].plot(idx, y_pred[:n], label="Predicted", lw=1.2, color="#FF5722", linestyle="--")
    axes[0].set_title("Test Set - First 7 Days")
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].tick_params(axis="x", rotation=30)

    lim = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]
    axes[1].scatter(y_true, y_pred, alpha=0.15, s=4, color="#607D8B")
    axes[1].plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Actual (kWh)")
    axes[1].set_ylabel("Predicted (kWh)")
    axes[1].set_title("Actual vs Predicted Scatter")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, Path(output_dir), "02_actual_vs_predicted.png", save)


def plot_forecast(
    df_hist: pd.DataFrame,
    forecast_df: pd.DataFrame,
    history_days: int = 14,
    output_dir: str | Path = "outputs/graphs",
    save: bool = True,
):
    fig, ax = plt.subplots(figsize=(16, 6))

    tail = df_hist["consumption_kwh"].iloc[-(history_days * 24) :]
    ax.plot(tail.index, tail.values, color="#2196F3", lw=1.2, label="Historical")

    fc = forecast_df["predicted_kwh"]
    upper = fc * 1.10
    lower = fc * 0.90

    ax.plot(fc.index, fc.values, color="#FF5722", lw=1.5, linestyle="--", label="Forecast")
    ax.fill_between(fc.index, lower, upper, alpha=0.15, color="#FF5722", label="+/-10% band")
    ax.axvline(df_hist.index[-1], color="grey", linestyle=":", lw=1.2, label="Now")

    ax.set_title("30-Day Energy Consumption Forecast", fontsize=14, fontweight="bold")
    ax.set_ylabel("kWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    plt.tight_layout()
    _save(fig, Path(output_dir), "03_30day_forecast.png", save)


def plot_feature_importance(
    model,
    feature_cols: list,
    top_n: int = 20,
    output_dir: str | Path = "outputs/graphs",
    save: bool = True,
):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(fi))
    fi[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")

    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, Path(output_dir), "04_feature_importance.png", save)


def plot_error_distribution(
    y_true,
    y_pred,
    output_dir: str | Path = "outputs/graphs",
    save: bool = True,
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")

    axes[0].hist(errors, bins=60, color="#3F51B5", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", lw=1.5, linestyle="--", label="Zero error")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (kWh)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].scatter(y_pred, errors, alpha=0.15, s=4, color="#009688")
    axes[1].axhline(0, color="red", lw=1.2, linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted (kWh)")
    axes[1].set_ylabel("Residual (kWh)")

    plt.tight_layout()
    _save(fig, Path(output_dir), "05_residual_analysis.png", save)


def _save(fig, output_dir: Path, filename: str, save: bool) -> None:
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"        Saved: {path}")
    plt.close(fig)
