import streamlit as st
import pandas as pd
import plotly.express as px

from src.loader import detect_kwh_col
from app_loaders import safe_mean, safe_max
from src.features import night_baseline, _hourly

def render_overview(
    daily_totals, date_col, daily_value_col, df_clean,
    consumption_col,
    weekday_order=None, simple=False,
):
    """
    Render the Overview tab.

    Parameters
    ----------
    simple : bool
        True  → friendly labels, explanatory text, and hourly heatmap (Simple mode).
        False → concise technical labels only (Analyst mode).
    weekday_order : list, optional
        Required when simple=True for the heatmap row order.
    """
    if simple:
        st.header("Your Electricity at a Glance")
        st.markdown(
            "For the best results, make sure you have at least 30 weeks of measurements. "
            "The charts below will update as you add more data."
        )
        labels = ("Total used", "Days tracked", "Typical day", "Busiest day")
    else:
        st.header("Overview")
        labels = ("Total Consumption", "Days Analyzed", "Avg Daily", "Peak Day")

    # ── Key metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    total_kwh = pd.to_numeric(daily_totals[daily_value_col], errors="coerce").sum()
    with col1:
        st.metric(labels[0], f"{total_kwh:,.1f} kWh")
    with col2:
        st.metric(labels[1], len(daily_totals))
    with col3:
        st.metric(labels[2], f"{safe_mean(daily_totals[daily_value_col]):.1f} kWh")
    with col4:
        st.metric(labels[3], f"{safe_max(daily_totals[daily_value_col]):.1f} kWh")
    with col5:
        baseline_label = "Min. Consumption Baseline" if not simple else "Minimal Consumption"
        try:
            df_nb = df_clean.copy()
            df_nb["datetime"] = pd.to_datetime(df_nb["datetime"])
            if "kWh" not in df_nb.columns:
                kwh_col = detect_kwh_col(df_nb)
                if kwh_col:
                    df_nb = df_nb.rename(columns={kwh_col: "kWh"})
            h_nb = _hourly(df_nb)
            nb_series = night_baseline(h_nb)
            nb_val    = nb_series.get("min_consumption_baseline_kwh", float("nan"))
            nb_str    = f"{nb_val:.2f} kWh/h" if pd.notna(nb_val) else "N/A"
        except Exception as e:
            st.error(f"Feature computation failed: {e}")
            return None
        st.metric(
            baseline_label, nb_str,
            help=(
                "Average electricity used between 23:00 and 07:00, when the household "
                "is asleep. This is your unavoidable standby load — fridges, routers, "
                "and always-on devices. Below 0.10 kWh/h is excellent; above 0.20 is worth investigating."
            ),
        )

    # ── Daily line chart ──────────────────────────────────────────────────────
    if simple:
        chart = px.line(
            daily_totals, x=date_col, y=daily_value_col,
            title="Daily electricity use over time",
            labels={daily_value_col: "kWh", date_col: "Date"},
        )
    else:
        chart = px.line(daily_totals, x=date_col, y=daily_value_col,
                        title="Daily Consumption Trend")

    st.plotly_chart(chart, width="stretch")

    if simple:
        st.caption(
            "Each point is one day's total electricity use. "
            "Peaks show days you used noticeably more than usual."
        )

    # ── Heatmap (Simple mode only) ────────────────────────────────────────────
    if simple and weekday_order is not None:
        st.divider()
        st.subheader("When do you use the most electricity?")
        st.markdown(
            "The chart below shows your average electricity use by hour of the day and day of the week. "
            "Darker red means that time slot is typically busier."
        )

        if {"hour", "weekday"}.issubset(df_clean.columns):
            pivot_mean = _weekday_hour_pivot(df_clean, consumption_col, weekday_order)
            st.plotly_chart(
                px.imshow(
                    pivot_mean,
                    title="Average electricity use: day of week vs. hour of day",
                    color_continuous_scale="RdYlGn_r",
                    labels={"x": "Hour of day", "y": "Day of week", "color": "kWh"},
                    aspect="auto",
                ),
                width="stretch",
            )
            st.caption(
                "Each cell shows the average kWh for that hour and day combination. "
                "Darker red = higher usage."
            )


def _weekday_hour_pivot(df, consumption_col, weekday_order):
    """Return a pivot table: rows = weekday, columns = hour, values = mean consumption."""
    pivot = (
        df.groupby(["weekday", "hour"])[consumption_col]
        .mean()
        .unstack(level="hour")
    )
    # Reorder rows to match weekday order
    pivot = pivot.reindex([w for w in weekday_order if w in pivot.index])
    return pivot
