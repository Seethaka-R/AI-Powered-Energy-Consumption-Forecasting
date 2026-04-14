"""Shared utility helpers for the forecasting project."""

from __future__ import annotations

from pathlib import Path

import joblib


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_structure() -> None:
    root = get_project_root()
    for relative in [
        "data/raw",
        "data/processed",
        "models",
        "outputs",
        "outputs/graphs",
        "images",
        "docs",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def save_model(model, filepath: str | Path) -> Path:
    path = _resolve_project_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: str | Path):
    path = _resolve_project_path(filepath)
    return joblib.load(path)


def print_banner() -> None:
    banner = """
============================================================
      AI-Powered Energy Consumption Forecasting
============================================================
Synthetic data -> preprocessing -> features -> model -> forecast
"""
    print(banner.strip("\n"))


def _resolve_project_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    return path if path.is_absolute() else get_project_root() / path
