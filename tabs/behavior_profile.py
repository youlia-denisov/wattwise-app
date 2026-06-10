"""
Behavioral fingerprint tab.

Calls build_user_features() from src/features.py on the full cleaned dataset
and renders the results as an interactive dashboard.

The features are grouped into four sections that mirror the feature-engineering
module: time-of-day ratios, weekday/weekend, regularity, and peak behaviour.
Each number is shown with a plain-English interpretation so the tab is readable
even without a data-science background.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_behavior_profile(df_clean):
    st.header("Behavioural Fingerprint")
    st.html(
        "This tab distils your entire consumption history into a handful of "
        "interpretable numbers — each one capturing a distinct aspect of how "
        "your household uses electricity. Together they form your **behavioural "
        "fingerprint**: a compact portrait that distinguishes WFH workers from "
        "office commuters, early risers from night owls, and routine households "
        "from unpredictable ones."
    )

    features = _compute_features(df_clean)
    if features is None:
        return

    _render_time_of_day(features)
    st.divider()
    _render_weekday_weekend(features)
    st.divider()
    _render_regularity(features)
    st.divider()
    _render_peak_behaviour(features)


# ── feature computation ───────────────────────────────────────────────────────

@st.cache_data
def _compute_features(df_clean: pd.DataFrame) -> pd.Series | None:
    """
    Calls build_user_features from src/features.py.
    Renames the kWh column if needed so feature functions always receive
    a column literally called 'kWh'.
    """
    try:
        from src.features import build_user_features
    except ImportError as e:
        st.error(f"Could not import feature engineering module: {e}")
        return None

    df = df_clean.copy()
    if "datetime" not in df.columns:
        st.error("'datetime' column not found in cleaned data.")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])

    if "kWh" not in df.columns:
        kwh_col = next(
            (c for c in df.columns if any(w in c.lower() for w in ["kwh", "kwatt", "consumption"])),
            None,
        )
        if kwh_col is None:
            st.error("No kWh column found in cleaned data.")
            return None
        df = df.rename(columns={kwh_col: "kWh"})

    try:
        return build_user_features(df)
    except Exception as e:
        st.error(f"Feature computation failed: {e}")
        return None


# ── section renderers ─────────────────────────────────────────────────────────

def _render_time_of_day(f: pd.Series):
    st.subheader("Time-of-day ratios")
    st.caption(
        "What fraction of your total electricity falls in each part of the day? "
        "These three windows sum to 1, so they are scale-invariant: a frugal household "
        "and a heavy-use household with the same daily routine get the same ratios."
    )

    ratios = {
        "Day\n07–16":     f.get("ratio_day",     0),
        "Evening\n17–22": f.get("ratio_evening", 0),
        "Night\n23–06":   f.get("ratio_night",   0),
    }
    df_r = pd.DataFrame({"Period": list(ratios.keys()), "Fraction": list(ratios.values())})

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.bar(
            df_r, x="Period", y="Fraction",
            text=df_r["Fraction"].map(lambda v: f"{v:.1%}"),
            color="Period",
            color_discrete_sequence=["#FFB74D", "#70b8ff", "#9575CD"],
            title="Share of daily consumption by time window",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Fraction of daily total")
        st.plotly_chart(fig, width="stretch")

    with col2:
        periods = ["Day", "Evening", "Night"]
        values  = [f.get("ratio_day", 0), f.get("ratio_evening", 0), f.get("ratio_night", 0)]
        values_closed = values + [values[0]]
        periods_closed = periods + [periods[0]]

        fig_r = go.Figure(go.Scatterpolar(
            r=values_closed, theta=periods_closed,
            fill="toself", fillcolor="rgba(112,184,255,0.25)",
            line=dict(color="#70b8ff", width=2),
            name="Your profile",
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.3])),
            title="Usage shape (radar)",
            showlegend=False,
        )
        st.plotly_chart(fig_r, width="stretch")

    day     = f.get("ratio_day", 0)
    evening = f.get("ratio_evening", 0)
    if day > 0.45:
        st.info("**High daytime usage** — consistent with working from home or someone being home during the day.")
    elif evening > 0.40:
        st.info("**Prominent evening peak** — classic after-work pattern: heavy usage in the evening hours.")


def _render_weekday_weekend(f: pd.Series):
    st.subheader("Weekday vs. weekend")

    col1, col2 = st.columns(2)

    wr = f.get("weekend_ratio", None)
    ms = f.get("morning_shift_hours", None)

    with col1:
        if wr is not None:
            _metric_with_bar(
                label="Weekend ratio",
                value=wr,
                fmt=".2f",
                help_text=(
                    "Mean weekend hourly usage ÷ mean weekday hourly usage. "
                    "> 1 → more electricity on weekends (home on weekends). "
                    "< 1 → heavier weekday use (WFH or weekday-heavy appliances)."
                ),
                reference=1.0,
                lo=0.5, hi=1.8,
            )

    with col2:
        if ms is not None and not pd.isna(ms):
            st.metric(
                "Weekend morning shift",
                f"{ms:+.1f} h",
                help=(
                    "How many hours later the morning electricity peak falls on weekends vs. weekdays. "
                    "Positive = sleeping in on weekends."
                ),
            )
            if ms > 1.5:
                st.caption("☕ Weekends start significantly later — classic 'sleeping in' pattern.")
            elif ms < -0.5:
                st.caption("⏰ Earlier starts on weekends (sport, market, religious observance?).")
            else:
                st.caption("Morning routine is similar on weekdays and weekends.")


def _render_regularity(f: pd.Series):
    st.subheader("Regularity & variability")
    st.caption(
        "Coefficient of Variation (CV) = std ÷ mean. Low CV means predictable, "
        "clockwork usage. High CV means erratic — irregular schedule, guests, or "
        "appliances that run only occasionally."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        cv_d = f.get("cv_daily", None)
        if cv_d is not None:
            st.metric("CV (daily totals)", f"{cv_d:.2f}",
                      help="Day-to-day variability of total kWh.")
            st.caption("Low < 0.3 · Medium 0.3–0.6 · High > 0.6")

    with col2:
        cv_h = f.get("cv_same_hour", None)
        if cv_h is not None:
            st.metric("CV (same hour, cross-day)", f"{cv_h:.2f}",
                      help="Average variability of each hour-slot across different days.")

    with col3:
        rs = f.get("routine_score", None)
        if rs is not None:
            st.metric("Routine score", f"{rs:.2f}",
                      help="1 − CV(same hour). Closer to 1 = very routine household.")
            if rs > 0.7:
                st.caption("🕐 Very routine — predictable schedule.")
            elif rs > 0.4:
                st.caption("〰️ Moderately routine.")
            else:
                st.caption("🎲 Irregular — usage varies a lot from day to day.")


def _render_peak_behaviour(f: pd.Series):
    st.subheader("Peak behaviour")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hop = f.get("hour_of_peak", None)
        if hop is not None:
            hour_label = f"{int(hop):02d}:00"
            st.metric("Hour of peak", hour_label,
                      help="The hour (averaged across all days) with the highest electricity use.")
            if hop < 10:
                st.caption("🌅 Morning peak")
            elif hop < 15:
                st.caption("☀️ Midday peak")
            elif hop < 20:
                st.caption("🌆 Afternoon/evening peak")
            else:
                st.caption("🌙 Night peak")

    with col2:
        ptm = f.get("peak_to_mean", None)
        if ptm is not None:
            st.metric("Peak-to-mean ratio", f"{ptm:.2f}",
                      help="Peak hourly usage ÷ mean hourly usage. High → spiky; low → flat load.")
            if ptm > 3:
                st.caption("⚡ Very spiky — concentrated usage burst.")
            elif ptm > 2:
                st.caption("📈 Moderate spike.")
            else:
                st.caption("📉 Flat load — spread evenly through the day.")

    with col3:
        ecs = f.get("evening_chores_score", None)
        if ecs is not None:
            st.metric("Evening chores score", f"{ecs:.2f}",
                      help="Fri/Sat evening usage ÷ weekday evening usage. High → weekend chores/cooking.")
            if ecs > 1.3:
                st.caption("🧹 Weekend evenings are more active (cooking, cleaning).")
            elif ecs < 0.8:
                st.caption("📅 Weekday evenings dominate.")
            else:
                st.caption("Evenings are similar on weekdays and weekends.")

    with col4:
        nb = f.get("night_baseline_kwh", None)
        if nb is not None:
            st.metric("Night baseline", f"{nb:.3f} kWh/h",
                      help="Mean hourly consumption 00:00–05:00. Captures always-on devices.")
            if nb > 0.2:
                st.caption("🔌 High standby load — check always-on appliances.")
            else:
                st.caption("✅ Low standby load.")


# ── small helper ──────────────────────────────────────────────────────────────

def _metric_with_bar(label, value, fmt, help_text, reference, lo, hi):
    """Show a metric plus a tiny progress bar positioned relative to a reference."""
    st.metric(label, f"{value:{fmt}}", help=help_text)
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    ref_pct = max(0.0, min(1.0, (reference - lo) / (hi - lo)))
    st.progress(pct, text=f"reference (1.0) at {ref_pct:.0%} of scale")
