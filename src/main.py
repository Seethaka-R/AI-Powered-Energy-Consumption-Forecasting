"""Main entry point for the energy forecasting pipeline."""

from pathlib import Path

from data_loader import load_or_generate_dataset
from feature_engineering import engineer_features
from model import evaluate_model, forecast_future, train_model
from Preprocessing import preprocess_data
from Utils import ensure_project_structure, print_banner, save_model
from Visualization import (
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_feature_importance,
    plot_forecast,
    plot_raw_data,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    graph_dir = project_root / "outputs/graphs"
    ensure_project_structure()
    print_banner()

    print("\n[1/7] Loading dataset...")
    df = load_or_generate_dataset(
        filepath=project_root / "data/raw/energy_data.csv",
        generate_if_missing=True,
    )
    print(f"      Dataset shape: {df.shape}")
    print(f"      Date range   : {df.index.min()} -> {df.index.max()}")

    print("\n[2/7] Preprocessing data...")
    df_clean = preprocess_data(df)
    df_clean.to_csv(project_root / "data/processed/energy_clean.csv")
    print(f"      Clean shape  : {df_clean.shape}")

    print("\n[3/7] Engineering features...")
    df_feat = engineer_features(df_clean)
    df_feat.to_csv(project_root / "data/processed/energy_features.csv")
    print(f"      Feature shape: {df_feat.shape}")
    print(f"      Features     : {list(df_feat.columns)}")

    print("\n[4/7] Training Random Forest model...")
    model, X_train, X_test, y_train, y_test, feature_cols = train_model(df_feat)
    save_model(model, project_root / "models/rf_energy_model.pkl")
    print("      Model saved -> models/rf_energy_model.pkl")

    print("\n[5/7] Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print(f"      RMSE : {metrics['RMSE']:.4f} kWh")
    print(f"      MAE  : {metrics['MAE']:.4f} kWh")
    print(f"      R2   : {metrics['R2']:.4f}")
    print(f"      MAPE : {metrics['MAPE']:.2f}%")

    print("\n[6/7] Forecasting next 30 days...")
    forecast_df = forecast_future(model, df_feat, feature_cols, days=30)
    forecast_df.to_csv(project_root / "outputs/forecast_30days.csv")
    print("      Forecast saved -> outputs/forecast_30days.csv")

    print("\n[7/7] Generating visualizations...")
    plot_raw_data(df_clean, output_dir=graph_dir)
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        df_feat.index[-len(y_test) :],
        output_dir=graph_dir,
    )
    plot_forecast(df_feat, forecast_df, output_dir=graph_dir)
    plot_feature_importance(model, feature_cols, output_dir=graph_dir)
    plot_error_distribution(y_test, y_pred, output_dir=graph_dir)
    print("      All graphs saved -> outputs/graphs/")

    print("\nPipeline complete. Check outputs/ for results.\n")


if __name__ == "__main__":
    main()
