"""Interactive dashboard for the energy forecasting project."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data/raw/energy_data.csv"
CLEAN_PATH = PROJECT_ROOT / "data/processed/energy_clean.csv"
FEATURE_PATH = PROJECT_ROOT / "data/processed/energy_features.csv"
FORECAST_PATH = PROJECT_ROOT / "outputs/forecast_30days.csv"
METRICS_PATH = PROJECT_ROOT / "outputs/metrics.txt"

PALETTE = {
    "bg": "#0b1118",
    "panel": "rgba(15, 23, 33, 0.86)",
    "panel_soft": "rgba(23, 34, 48, 0.72)",
    "border": "rgba(164, 180, 199, 0.18)",
    "text": "#d9e2ec",
    "muted": "#93a4b8",
    "silver": "#c5ced8",
    "steel": "#7b8da1",
    "cyan": "#41c7c7",
    "blue": "#57a6ff",
    "amber": "#f0b35a",
    "red": "#ff6b6b",
    "green": "#5be7a9",
}


st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 20% 10%, rgba(87, 166, 255, 0.16), transparent 24%),
                radial-gradient(circle at 80% 0%, rgba(65, 199, 199, 0.18), transparent 20%),
                radial-gradient(circle at 50% 100%, rgba(240, 179, 90, 0.12), transparent 28%),
                linear-gradient(160deg, #070c12 0%, #0b1118 52%, #121a24 100%);
            color: {PALETTE["text"]};
        }}
        .block-container {{
            max-width: 1360px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3 {{
            color: {PALETTE["silver"]};
            letter-spacing: -0.02em;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(12, 18, 26, 0.98), rgba(18, 27, 38, 0.98));
            border-right: 1px solid {PALETTE["border"]};
        }}
        .hero {{
            padding: 1.6rem 1.7rem;
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(14, 21, 31, 0.94) 0%, rgba(20, 34, 48, 0.92) 42%, rgba(37, 49, 61, 0.95) 100%);
            border: 1px solid rgba(197, 206, 216, 0.18);
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .hero:before {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent),
                linear-gradient(180deg, transparent 0%, rgba(65, 199, 199, 0.05) 100%);
            pointer-events: none;
        }}
        .hero h1 {{
            color: white;
            margin: 0;
        }}
        .hero p {{
            color: rgba(217, 226, 236, 0.86);
            margin: 0.4rem 0 0 0;
            max-width: 760px;
        }}
        .section-label {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["cyan"]};
            margin-bottom: 0.5rem;
        }}
        .metric-card {{
            background: linear-gradient(180deg, rgba(19, 28, 39, 0.92), rgba(12, 19, 28, 0.9));
            border: 1px solid {PALETTE["border"]};
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 118px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 14px 28px rgba(0,0,0,0.24);
        }}
        .metric-label {{
            color: {PALETTE["muted"]};
            font-size: 0.88rem;
            margin-bottom: 0.3rem;
        }}
        .metric-value {{
            color: {PALETTE["silver"]};
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .metric-sub {{
            margin-top: 0.45rem;
            color: {PALETTE["steel"]};
            font-size: 0.84rem;
        }}
        .panel {{
            background: linear-gradient(180deg, rgba(16, 24, 34, 0.88), rgba(12, 18, 27, 0.88));
            border: 1px solid {PALETTE["border"]};
            border-radius: 22px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: 0 16px 34px rgba(0,0,0,0.22);
        }}
        .mini-note {{
            color: {PALETTE["muted"]};
            font-size: 0.86rem;
            margin-top: -0.2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


@st.cache_data
def load_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    pattern = re.compile(r"^\s*([A-Z0-9]+)\s*:\s*([0-9.]+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


@st.cache_data
def load_feature_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


def build_kpi_card(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="rgba(197,206,216,0.08)",
            zerolinecolor="rgba(197,206,216,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(197,206,216,0.08)",
            zerolinecolor="rgba(197,206,216,0.08)",
        ),
    )
    return fig


def detect_anomalies(history: pd.DataFrame) -> pd.DataFrame:
    enriched = history.copy()
    rolling = enriched["consumption_kwh"].rolling(window=24, min_periods=12)
    enriched["rolling_mean"] = rolling.mean()
    enriched["rolling_std"] = rolling.std().fillna(0)
    enriched["z_score"] = (
        (enriched["consumption_kwh"] - enriched["rolling_mean"])
        / enriched["rolling_std"].replace(0, pd.NA)
    ).fillna(0)
    anomalies = enriched.loc[enriched["z_score"].abs() >= 2.5, ["timestamp", "consumption_kwh", "z_score"]]
    return anomalies.sort_values("timestamp")


def make_history_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, show_band: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["consumption_kwh"],
            mode="lines",
            name="Historical",
            line=dict(color=PALETTE["cyan"], width=2.8),
        )
    )

    if show_band:
        upper = forecast["predicted_kwh"] * 1.08
        lower = forecast["predicted_kwh"] * 0.92
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(87, 166, 255, 0.14)",
                line=dict(width=0),
                name="Forecast Range",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["predicted_kwh"],
            mode="lines",
            name="Forecast",
            line=dict(color=PALETTE["amber"], width=3, dash="dash"),
        )
    )
    fig.add_vline(
        x=history["timestamp"].iloc[-1],
        line_width=2,
        line_dash="dot",
        line_color=PALETTE["silver"],
    )
    fig.update_layout(
        title="Historical Demand vs Forecast Horizon",
        xaxis_title="Timestamp",
        yaxis_title="Consumption (kWh)",
    )
    return apply_plot_theme(fig, height=500)


def make_daily_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    hist_daily = history.set_index("timestamp")["consumption_kwh"].resample("D").sum().reset_index()
    fc_daily = forecast.set_index("timestamp")["predicted_kwh"].resample("D").sum().reset_index()
    hist_daily["series"] = "Historical"
    fc_daily["series"] = "Forecast"
    combined = pd.concat(
        [
            hist_daily.rename(columns={"consumption_kwh": "value"}),
            fc_daily.rename(columns={"predicted_kwh": "value"}),
        ],
        ignore_index=True,
    )
    fig = px.bar(
        combined,
        x="timestamp",
        y="value",
        color="series",
        barmode="group",
        color_discrete_map={"Historical": PALETTE["blue"], "Forecast": PALETTE["amber"]},
    )
    fig.update_layout(title="Daily Energy Totals", xaxis_title="Date", yaxis_title="Energy (kWh)")
    return apply_plot_theme(fig)


def make_temperature_chart(history: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        history,
        x="temperature_c",
        y="consumption_kwh",
        color="humidity_pct",
        opacity=0.52,
        color_continuous_scale=[PALETTE["blue"], PALETTE["cyan"], PALETTE["amber"]],
    )
    fig.update_layout(title="Consumption vs Temperature", xaxis_title="Temperature (C)", yaxis_title="Consumption (kWh)")
    return apply_plot_theme(fig)


def make_hour_profile_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    hist_profile = history.assign(hour=history["timestamp"].dt.hour).groupby("hour", as_index=False)["consumption_kwh"].mean()
    fc_profile = forecast.assign(hour=forecast["timestamp"].dt.hour).groupby("hour", as_index=False)["predicted_kwh"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_profile["hour"],
            y=hist_profile["consumption_kwh"],
            mode="lines+markers",
            name="Historical Profile",
            line=dict(color=PALETTE["cyan"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_profile["hour"],
            y=fc_profile["predicted_kwh"],
            mode="lines+markers",
            name="Forecast Profile",
            line=dict(color=PALETTE["amber"], width=3, dash="dash"),
        )
    )
    fig.update_layout(title="Average Hourly Load Profile", xaxis_title="Hour of Day", yaxis_title="Average Consumption (kWh)")
    return apply_plot_theme(fig)


def make_weekday_chart(history: pd.DataFrame) -> go.Figure:
    weekday_profile = (
        history.assign(weekday=history["timestamp"].dt.day_name())
        .groupby("weekday", as_index=False)["consumption_kwh"]
        .mean()
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_profile["weekday"] = pd.Categorical(weekday_profile["weekday"], categories=weekday_order, ordered=True)
    weekday_profile = weekday_profile.sort_values("weekday")
    fig = px.area(weekday_profile, x="weekday", y="consumption_kwh", color_discrete_sequence=[PALETTE["green"]])
    fig.update_layout(title="Average Consumption by Weekday", xaxis_title="Weekday", yaxis_title="Average Consumption (kWh)", showlegend=False)
    return apply_plot_theme(fig)


def make_heatmap(history: pd.DataFrame) -> go.Figure:
    heatmap_df = history.copy()
    heatmap_df["hour"] = heatmap_df["timestamp"].dt.hour
    heatmap_df["weekday"] = heatmap_df["timestamp"].dt.day_name()
    pivot = heatmap_df.pivot_table(index="weekday", columns="hour", values="consumption_kwh", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(order)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[
                [0.0, "#12202d"],
                [0.35, "#2a5e87"],
                [0.65, "#41c7c7"],
                [1.0, "#f0b35a"],
            ],
            colorbar=dict(title="kWh"),
        )
    )
    fig.update_layout(title="Weekly Load Heatmap", xaxis_title="Hour of Day", yaxis_title="Weekday")
    return apply_plot_theme(fig)


def make_monthly_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    hist_month = history.set_index("timestamp")["consumption_kwh"].resample("M").sum().reset_index()
    fc_month = forecast.set_index("timestamp")["predicted_kwh"].resample("M").sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_month["timestamp"],
            y=hist_month["consumption_kwh"],
            mode="lines+markers",
            name="Historical Monthly",
            line=dict(color=PALETTE["blue"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_month["timestamp"],
            y=fc_month["predicted_kwh"],
            mode="lines+markers",
            name="Forecast Monthly",
            line=dict(color=PALETTE["amber"], width=3, dash="dash"),
        )
    )
    fig.update_layout(title="Monthly Energy Trend", xaxis_title="Month", yaxis_title="Energy (kWh)")
    return apply_plot_theme(fig)


def make_feature_importance_chart(feature_df: pd.DataFrame) -> go.Figure | None:
    if feature_df.empty:
        return None

    corr = (
        feature_df.select_dtypes(include="number")
        .corr(numeric_only=True)["consumption_kwh"]
        .drop("consumption_kwh")
        .abs()
        .sort_values(ascending=False)
        .head(12)
        .sort_values()
    )
    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        color=corr.values,
        color_continuous_scale=[PALETTE["blue"], PALETTE["cyan"], PALETTE["amber"]],
    )
    fig.update_layout(
        title="Top Feature Relationships to Consumption",
        xaxis_title="Absolute Correlation",
        yaxis_title="Feature",
        coloraxis_showscale=False,
    )
    return apply_plot_theme(fig)


def make_anomaly_chart(history: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["consumption_kwh"],
            mode="lines",
            name="Load",
            line=dict(color=PALETTE["cyan"], width=2.2),
        )
    )
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["consumption_kwh"],
                mode="markers",
                name="Anomalies",
                marker=dict(color=PALETTE["red"], size=8, line=dict(color="white", width=0.8)),
            )
        )
    fig.update_layout(title="Detected Demand Anomalies", xaxis_title="Timestamp", yaxis_title="Consumption (kWh)")
    return apply_plot_theme(fig)


def download_section(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Exports</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name="forecast_dashboard_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="history_dashboard_export.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Interactive Control Room</div>
            <h1>AI-Powered Energy Consumption Forecasting</h1>
            <p>A metallic, robotics-inspired dashboard for exploring demand behavior, anomaly zones, feature strength, and the 30-day consumption forecast from your generated model outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_paths = [path for path in [CLEAN_PATH, FORECAST_PATH, METRICS_PATH] if not path.exists()]
    if missing_paths:
        st.error("Some output files are missing. Run `python src/main.py` first, then reload this dashboard.")
        st.stop()

    history = load_time_series(CLEAN_PATH)
    forecast = load_time_series(FORECAST_PATH)
    metrics = load_metrics(METRICS_PATH)
    feature_df = load_feature_frame(FEATURE_PATH)

    with st.sidebar:
        st.header("Dashboard Controls")
        lookback_days = st.slider("Historical window (days)", 7, 365, 60, 1)
        show_raw = st.toggle("Use raw dataset", value=False)
        show_forecast_band = st.toggle("Show forecast confidence band", value=True)
        show_table = st.toggle("Show forecast table", value=True)
        selected_view = st.selectbox("Focus metric", ["Consumption", "Temperature", "Humidity"], index=0)

        source_path = RAW_PATH if show_raw and RAW_PATH.exists() else CLEAN_PATH
        history = load_time_series(source_path)
        history = history.sort_values("timestamp")

        start_cutoff = history["timestamp"].max() - pd.Timedelta(days=lookback_days)
        history_window = history.loc[history["timestamp"] >= start_cutoff].copy()

        st.caption(f"History source: `{source_path.relative_to(PROJECT_ROOT)}`")
        st.caption(f"Forecast source: `{FORECAST_PATH.relative_to(PROJECT_ROOT)}`")
        st.caption(f"Feature source: `{FEATURE_PATH.relative_to(PROJECT_ROOT)}`")

    anomalies = detect_anomalies(history_window)

    latest_actual = history["consumption_kwh"].iloc[-1]
    avg_forecast = forecast["predicted_kwh"].mean()
    peak_forecast = forecast["predicted_kwh"].max()
    total_forecast = forecast["predicted_kwh"].sum()
    peak_timestamp = forecast.loc[forecast["predicted_kwh"].idxmax(), "timestamp"]
    demand_delta = forecast["predicted_kwh"].iloc[0] - latest_actual

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        build_kpi_card("Latest Actual", f"{latest_actual:.2f} kWh", "Most recent historical value")
    with c2:
        build_kpi_card("Average Forecast", f"{avg_forecast:.2f} kWh", "Mean of next 30 days")
    with c3:
        build_kpi_card("Peak Forecast", f"{peak_forecast:.2f} kWh", peak_timestamp.strftime("%d %b %Y %H:%M"))
    with c4:
        build_kpi_card("30-Day Total", f"{total_forecast:,.0f} kWh", "Projected energy volume")
    with c5:
        build_kpi_card("Immediate Shift", f"{demand_delta:+.2f} kWh", "Forecast start vs latest actual")

    if metrics:
        st.markdown('<div class="section-label" style="margin-top: 1rem;">Model Quality</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            build_kpi_card("RMSE", f"{metrics.get('RMSE', 0.0):.4f}", "Lower is better")
        with m2:
            build_kpi_card("MAE", f"{metrics.get('MAE', 0.0):.4f}", "Average absolute error")
        with m3:
            build_kpi_card("R2", f"{metrics.get('R2', 0.0):.4f}", "Explained variance")
        with m4:
            build_kpi_card("MAPE", f"{metrics.get('MAPE', 0.0):.2f}%", "Percentage forecasting error")

    tabs = st.tabs(["Forecast Hub", "Patterns", "Intelligence", "Explorer"])

    with tabs[0]:
        st.plotly_chart(make_history_forecast_chart(history_window, forecast, show_forecast_band), use_container_width=True)
        left, right = st.columns([1.05, 0.95])
        with left:
            st.plotly_chart(make_daily_chart(history_window, forecast), use_container_width=True)
        with right:
            st.plotly_chart(make_monthly_chart(history_window, forecast), use_container_width=True)

    with tabs[1]:
        top_left, top_right = st.columns(2)
        with top_left:
            st.plotly_chart(make_hour_profile_chart(history_window, forecast), use_container_width=True)
        with top_right:
            st.plotly_chart(make_weekday_chart(history_window), use_container_width=True)
        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.plotly_chart(make_temperature_chart(history_window), use_container_width=True)
        with bottom_right:
            st.plotly_chart(make_heatmap(history_window), use_container_width=True)

    with tabs[2]:
        intel_left, intel_right = st.columns([1.05, 0.95])
        with intel_left:
            st.plotly_chart(make_anomaly_chart(history_window, anomalies), use_container_width=True)
        with intel_right:
            feature_chart = make_feature_importance_chart(feature_df)
            if feature_chart is not None:
                st.plotly_chart(feature_chart, use_container_width=True)
            else:
                st.info("Feature dataset not found yet. Run the pipeline to generate `energy_features.csv`.")

        st.markdown('<div class="section-label">Signal Summary</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            build_kpi_card("Detected Anomalies", str(len(anomalies)), "Events beyond 2.5 rolling z-score")
        with s2:
            peak_hour = history_window.assign(hour=history_window["timestamp"].dt.hour).groupby("hour")["consumption_kwh"].mean().idxmax()
            build_kpi_card("Dominant Peak Hour", f"{int(peak_hour):02d}:00", "Highest average load hour")
        with s3:
            metric_series = {
                "Consumption": history_window["consumption_kwh"],
                "Temperature": history_window["temperature_c"],
                "Humidity": history_window["humidity_pct"],
            }
            build_kpi_card(
                f"{selected_view} Range",
                f"{metric_series[selected_view].min():.1f} - {metric_series[selected_view].max():.1f}",
                "Within selected historical window",
            )

    with tabs[3]:
        download_section(history_window, forecast)
        st.markdown("#### History Window")
        st.dataframe(history_window.tail(250), use_container_width=True, height=250)
        if show_table:
            st.markdown("#### Forecast Output")
            st.dataframe(forecast, use_container_width=True, height=250)
        if not anomalies.empty:
            st.markdown("#### Anomaly Log")
            st.dataframe(anomalies.tail(100), use_container_width=True, height=220)


if __name__ == "__main__":
    main()
