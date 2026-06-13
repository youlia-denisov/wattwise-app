"""
Behavioural fingerprint tab.

Calls build_user_features() from src/features.py on the full cleaned dataset
and renders the results as an interactive dashboard.

Two render modes:
  simple=False (default / Analyst) — full detail with charts and expanders.
  simple=True  (Simple mode)       — one-page summary card with persona + top insights.

Section renderers (_render_time_of_day etc.) live in behavior_components.py
to keep this file focused on orchestration only.
"""

import streamlit as st
import pandas as pd

from src.features import derive_persona, build_insights
from src.loader import detect_kwh_col
from tabs.behavior_components import (
    _insight_card,
    _render_persona_banner,
    _render_time_of_day,
    _render_weekday_weekend,
    _render_regularity,
    _render_peak_behaviour,
)


# ── Public entry point ────────────────────────────────────────────────────────

def render_behavior_profile(df_clean, simple: bool = False):
    st.header("Your Household Profile" if simple else "Usage Habits")

    features = _compute_features(df_clean)
    if features is None:
        return

    persona = derive_persona(features)

    if simple:
        _render_simple_summary(features, persona)
    else:
        _render_persona_banner(features, persona)
        st.divider()
        _render_time_of_day(features)
        st.divider()
        _render_weekday_weekend(features)
        st.divider()
        _render_regularity(features)
        st.divider()
        _render_peak_behaviour(features)


# ── Feature computation ───────────────────────────────────────────────────────

@st.cache_data
def _compute_features(df_clean: pd.DataFrame) -> pd.Series | None:
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
        kwh_col = detect_kwh_col(df)
        if kwh_col is None:
            st.error("No kWh column found in cleaned data.")
            return None
        df = df.rename(columns={kwh_col: "kWh"})

    try:
        return build_user_features(df)
    except Exception as e:
        st.error(f"Feature computation failed: {e}")
        return None


# ── Simple-mode summary ───────────────────────────────────────────────────────

def _render_simple_summary(f: pd.Series, persona: dict):
    """Compact single-page view for Simple mode."""
    st.markdown(
        f"""
        <div style="
            background: {persona['color']}22;
            border-left: 5px solid {persona['color']};
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 2.4rem; line-height: 1;">{persona['emoji']}</div>
            <div style="font-size: 1.4rem; font-weight: 700; margin-top: 6px;">{persona['label']}</div>
            <div style="font-size: 1rem; color: #555; margin-top: 4px;">{persona['tagline']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    insights = build_insights(f)
    cols = st.columns(2)
    for i, ins in enumerate(insights[:4]):
        with cols[i % 2]:
            _insight_card(ins)

    with st.expander("See all metrics in detail"):
        _render_time_of_day(f)
        st.divider()
        _render_weekday_weekend(f)
        st.divider()
        _render_regularity(f)
        st.divider()
        _render_peak_behaviour(f)
