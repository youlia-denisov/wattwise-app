import streamlit as st
import pandas as pd
import plotly.express as px

from src.outliers import detect_outliers_iqr


def render_trends(df_clean, consumption_col):
    st.header("Trends & Outliers")

    col1, col2 = st.columns(2)
    with col1:
        daily_sum = df_clean.groupby("date")[consumption_col].sum().reset_index(name="daily_kWh")
        daily_sum["rolling_7d"] = daily_sum["daily_kWh"].rolling(7, min_periods=1).mean()
        st.plotly_chart(
            px.line(daily_sum, x="date", y=["daily_kWh", "rolling_7d"],
                    title="Daily Consumption with 7-Day Rolling Average",
                    labels={"daily_kWh": "Daily Total (kWh)", "rolling_7d": "7-Day Rolling Avg"}),
            width="stretch",
        )
    with col2:
        sorted_data = (
            df_clean[[consumption_col]]
            .sort_values(consumption_col, ascending=False)
            .reset_index(drop=True)
        )
        sorted_data["percentile"] = (sorted_data.index + 1) / len(sorted_data) * 100
        st.plotly_chart(
            px.line(sorted_data, x="percentile", y=consumption_col, title="Load Duration Curve"),
            width="stretch",
        )

    st.caption(
        "**Left — 7-day rolling average:** instead of showing every noisy day individually, "
        "this line averages the last 7 days together as it moves forward in time, smoothing out "
        "one-off spikes so you can see the real long-term trend. "
        "**Right — Load Duration Curve:** all hourly readings sorted from highest to lowest. "
        "The left side shows your peak usage moments (e.g. top 5% of hours), the right your quietest. "
        "A steep drop means a few hours dominate your bill; a flat curve means usage is spread evenly."
    )

    _render_outliers(df_clean, consumption_col)


def _render_outliers(df_clean, consumption_col):
    """IQR-based outlier detection and visualisation."""
    st.subheader("Unusual Readings — IQR Method")

    # Rename to "kWh" so detect_outliers_iqr (which expects that column name) can work.
    # We rename back afterwards so the rest of the tab sees the original column name.
    df_norm = df_clean.copy()
    if consumption_col != "kWh":
        df_norm = df_norm.rename(columns={consumption_col: "kWh"})
    df_norm["kWh"] = pd.to_numeric(df_norm["kWh"], errors="coerce")
    df_norm = df_norm.dropna(subset=["kWh"])

    # Compute display stats (q1/q3/iqr are needed for the metrics cards below)
    kwh = df_norm["kWh"]
    q1, q3 = kwh.quantile(0.25), kwh.quantile(0.75)
    iqr = q3 - q1
    total_readings = len(kwh)

    # Outlier detection — no reimplementation; use the shared function
    df_out_raw = detect_outliers_iqr(df_norm)

    # Pull fences from the returned metadata columns (same values on every row)
    if not df_out_raw.empty:
        upper_fence = df_out_raw["upper_limit"].iloc[0]
        lower_fence = df_out_raw["lower_limit"].iloc[0]
    else:
        upper_fence = q3 + 1.5 * iqr
        lower_fence = q1 - 1.5 * iqr

    # Restore original column name and drop pipeline metadata columns before display
    df_out = (df_out_raw
              .drop(columns=["method", "lower_limit", "upper_limit"], errors="ignore")
              .rename(columns={"kWh": consumption_col}))
    total_readings = len(kwh)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Q1 — lower typical", f"{q1:.3f} kWh",
                  help="25% of your readings are below this value")
    with m2:
        st.metric("Q3 — upper typical", f"{q3:.3f} kWh",
                  help="75% of your readings are below this value")
    with m3:
        st.metric("Typical range (IQR)", f"{iqr:.3f} kWh",
                  help="The spread of your middle 50% of readings")
    with m4:
        st.metric("Unusual if above", f"{upper_fence:.3f} kWh",
                  help="Readings above this are flagged as unusually high")
    with m5:
        st.metric(
            "Unusual readings",
            f"{len(df_out)} of {total_readings:,}",
            help=f"{len(df_out) / total_readings * 100:.1f}% of all readings fall outside the IQR fences",
        )

    st.caption(
        "The **IQR (interquartile range)** method looks at the middle 50% of your readings "
        "and flags anything that falls far outside that range — "
        "think of it as: *'this hour looks nothing like a normal hour for you.'*"
    )

    if "weekday" in df_out.columns:
        by_day = (
            df_out.groupby("weekday")[consumption_col]
            .agg(count="count", mean_kWh="mean", max_kWh="max")
            .reset_index()
        )
        st.plotly_chart(
            px.bar(by_day, x="weekday", y="count", color="mean_kWh",
                   color_continuous_scale="Reds", title="IQR Outlier Count by Weekday",
                   labels={"count": "# Outliers", "mean_kWh": "Avg kWh"}),
            width="stretch",
        )
        st.caption(
            "Each bar shows how many flagged readings fall on that weekday. "
            "Color encodes the average kWh of those outliers — darker red means the unusual readings "
            "on that day tend to be higher-consumption events."
        )
