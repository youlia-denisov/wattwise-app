"""
Behavioural fingerprint tab.

Calls build_user_features() from src/features.py on the full cleaned dataset
and renders the results as an interactive dashboard.

Two render modes:
  simple=False (default / Analyst) — full detail with charts and expanders.
  simple=True  (Simple mode)       — one-page summary card with persona + top insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_behavior_profile(df_clean, simple: bool = False):
    st.header("Your Household Profile" if simple else "Usage Habits")

    features = _compute_features(df_clean)
    if features is None:
        return

    persona = _derive_persona(features)

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Persona logic
# ─────────────────────────────────────────────────────────────────────────────

def _derive_persona(f: pd.Series) -> dict:
    """
    Map feature values to a single human-readable household persona.
    Returns a dict with: emoji, label, tagline, color.
    """
    ratio_day     = f.get("daytime_activity_share", 0)
    ratio_night   = f.get("ratio_night",            0)
    ratio_evening = f.get("ratio_evening",          0)
    weekend_ratio = f.get("weekend_ratio", 1)
    routine_score = f.get("routine_score", 0.5)

    if ratio_day > 0.45:
        return dict(emoji="🏠", label="Home Dweller",
                    tagline="Most of your electricity is used during the day — you're home a lot!",
                    color="#FFB74D")
    if ratio_night > 0.25:
        return dict(emoji="🌙", label="Night Owl",
                    tagline="Your household comes alive at night — late evenings are your peak time.",
                    color="#9575CD")
    if weekend_ratio > 1.35:
        return dict(emoji="🎉", label="Weekend Warrior",
                    tagline="You use noticeably more electricity on weekends — home is where the weekend is.",
                    color="#4DB6AC")
    if routine_score > 0.70:
        return dict(emoji="🕐", label="Clockwork Household",
                    tagline="Your daily routine is very consistent — predictable and efficient.",
                    color="#64B5F6")
    if ratio_evening > 0.40:
        return dict(emoji="🌆", label="After-Work Household",
                    tagline="Evenings are your busiest time — classic after-work energy spike.",
                    color="#F06292")
    return dict(emoji="⚖️", label="Balanced Household",
                tagline="Your usage is spread fairly evenly — no single pattern dominates.",
                color="#81C784")


# ─────────────────────────────────────────────────────────────────────────────
# Simple-mode: one-card summary
# ─────────────────────────────────────────────────────────────────────────────

def _render_simple_summary(f: pd.Series, persona: dict):
    """Compact single-page view for Simple mode."""

    # Persona banner
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

    # Key insights (2 columns × 2 rows)
    insights = _build_insights(f)
    cols = st.columns(2)
    for i, ins in enumerate(insights[:4]):
        with cols[i % 2]:
            _insight_card(ins)

    # Collapsible detail
    with st.expander("See all metrics in detail"):
        _render_time_of_day(f)
        st.divider()
        _render_weekday_weekend(f)
        st.divider()
        _render_regularity(f)
        st.divider()
        _render_peak_behaviour(f)


def _build_insights(f: pd.Series) -> list[dict]:
    """
    Derive 4-6 plain-English insight cards from the feature values.
    Each dict: icon, title, value_label, description, status (good/warn/info).
    """
    insights = []

    # --- Peak time ---
    hop = f.get("hour_of_peak", None)
    if hop is not None:
        hour_label = f"{int(hop):02d}:00"
        if hop < 10:
            peak_desc = "You peak in the morning — early riser or morning appliances."
        elif hop < 15:
            peak_desc = "Midday is your busiest time — typical of home workers."
        elif hop < 20:
            peak_desc = "Your peak is in the late afternoon or evening — common after-work pattern."
        else:
            peak_desc = "You peak late at night — night-owl household."
        insights.append(dict(icon="⏰", title="Peak hour", value=hour_label,
                             desc=peak_desc, status="info"))

    # --- Routine level ---
    rs = f.get("routine_score", None)
    if rs is not None:
        if rs > 0.70:
            r_label, r_desc, r_status = "Very routine", "Your schedule is highly predictable — same pattern day after day.", "good"
        elif rs > 0.45:
            r_label, r_desc, r_status = "Moderately routine", "Some variation in your daily pattern, but broadly consistent.", "info"
        else:
            r_label, r_desc, r_status = "Unpredictable", "Your usage varies a lot day-to-day — irregular schedule.", "warn"
        insights.append(dict(icon="📅", title="Routine level", value=r_label,
                             desc=r_desc, status=r_status))

    # --- Weekday vs weekend ---
    wr = f.get("weekend_ratio", None)
    if wr is not None:
        if wr > 1.2:
            wr_label, wr_desc, wr_status = f"{wr:.1f}× more on weekends", "You use noticeably more electricity on weekends — you're home more then.", "info"
        elif wr < 0.85:
            wr_label, wr_desc, wr_status = f"{wr:.1f}× less on weekends", "Weekdays dominate your usage — could be home-office setup or weekday-heavy appliances.", "info"
        else:
            wr_label, wr_desc, wr_status = "Similar on both days", "Your weekday and weekend usage are about the same.", "good"
        insights.append(dict(icon="📆", title="Weekday vs weekend", value=wr_label,
                             desc=wr_desc, status=wr_status))

    # --- Minimal Consumption Baseline ---
    nb = f.get("min_consumption_baseline_kwh", None)
    if nb is not None and not pd.isna(nb):
        if nb > 0.20:
            nb_label  = f"{nb:.2f} kWh/h"
            nb_desc   = (
                "This is your Minimal Consumption Baseline — electricity your home uses "
                "even when everyone is asleep (fridges, routers, standby devices). "
                "Your level is above average: worth checking for always-on appliances "
                "that could be switched off or replaced."
            )
            nb_status = "warn"
        elif nb > 0.10:
            nb_label  = f"{nb:.2f} kWh/h"
            nb_desc   = (
                "This is your Minimal Consumption Baseline — the unavoidable overnight draw "
                "from fridges, routers, and standby electronics. "
                "Your level is typical for a modern home."
            )
            nb_status = "info"
        else:
            nb_label  = f"{nb:.2f} kWh/h"
            nb_desc   = (
                "This is your Minimal Consumption Baseline — electricity consumed while "
                "the household sleeps. Your standby load is very low, suggesting "
                "well-managed or energy-efficient appliances."
            )
            nb_status = "good"
        insights.append(dict(icon="🔌", title="Minimal Consumption Baseline", value=nb_label,
                             desc=nb_desc, status=nb_status))

    # --- Sleep-in score ---
    ms = f.get("morning_shift_hours", None)
    if ms is not None and not pd.isna(ms):
        if ms > 1.5:
            ms_label, ms_desc = f"+{ms:.1f} h later", "You start your day noticeably later on weekends — classic sleep-in."
        elif ms < -0.5:
            ms_label, ms_desc = f"{ms:.1f} h earlier", "You actually start earlier on weekends — early bird!"
        else:
            ms_label, ms_desc = "No shift", "Morning routine is similar on weekdays and weekends."
        insights.append(dict(icon="😴", title="Weekend sleep-in", value=ms_label,
                             desc=ms_desc, status="info"))

    return insights


def _insight_card(ins: dict):
    status_colors = {"good": "#21c354", "warn": "#e6a817", "info": "#1c83e1"}
    color = status_colors.get(ins["status"], "#888")
    st.markdown(
        f"""
        <div style="
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.08);
            padding: 14px 16px;
            margin-bottom: 12px;
            border-left: 4px solid {color};
        ">
            <div style="font-size: 1.3rem;">{ins['icon']} <strong>{ins['title']}</strong></div>
            <div style="font-size: 1.05rem; font-weight: 600; margin: 4px 0;">{ins['value']}</div>
            <div style="font-size: 0.88rem; color: #666;">{ins['desc']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analyst-mode: persona banner at the top
# ─────────────────────────────────────────────────────────────────────────────

def _render_persona_banner(f: pd.Series, persona: dict):
    st.markdown(
        f"""
        <div style="
            background: {persona['color']}22;
            border-left: 5px solid {persona['color']};
            border-radius: 10px;
            padding: 16px 22px;
            margin-bottom: 10px;
        ">
            <span style="font-size: 2rem;">{persona['emoji']}</span>
            <span style="font-size: 1.3rem; font-weight: 700; margin-left: 10px;">{persona['label']}</span>
            <div style="color: #555; margin-top: 6px;">{persona['tagline']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This tab distils your consumption history into interpretable numbers. "
        "Together they form your **behavioural fingerprint** — a compact portrait of your household's habits."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section renderers (Analyst mode detail)
# ─────────────────────────────────────────────────────────────────────────────

def _render_time_of_day(f: pd.Series):
    st.subheader("🕐 When do you use electricity?")
    st.caption(
        "Your daily electricity is split across three windows — day (07–16), "
        "evening (17–22), and night (23–06). These fractions always add up to 100%, "
        "so they reflect *when* you use power, not *how much*."
    )

    ratios = {
        "Day\n07–16":     f.get("daytime_activity_share", 0),
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
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                          yaxis_title="Fraction of daily total")
        st.plotly_chart(fig, width="stretch")

    with col2:
        periods = ["Day", "Evening", "Night"]
        values  = [f.get("daytime_activity_share", 0), f.get("ratio_evening", 0), f.get("ratio_night", 0)]
        values_closed  = values + [values[0]]
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

    day     = f.get("daytime_activity_share", 0)
    evening = f.get("ratio_evening", 0)
    night   = f.get("ratio_night", 0)

    if day > 0.45:
        st.info("**Daytime-heavy** — most power is used during the day. Typical of households where someone is home all day.")
    elif evening > 0.40:
        st.info("**Evening-heavy** — the after-work hours drive your consumption. Classic 9-to-5 worker pattern.")
    elif night > 0.22:
        st.info("**Night-heavy** — significant overnight usage. Check for always-on appliances or night-owl habits.")
    else:
        st.info("**Balanced** — usage is spread fairly evenly across the day.")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **Day (07–16):** work hours — WFH, appliances running while people are awake\n"
            "- **Evening (17–22):** after-work peak — cooking, TV, EV charging, laundry\n"
            "- **Night (23–06):** baseline — standby devices, overnight appliances (dishwasher timer, etc.)\n\n"
            "Because the three values always add to 100%, they capture *shape*, not scale. "
            "A low-consumption household and a high-consumption one can have identical ratios."
        )


def _render_weekday_weekend(f: pd.Series):
    st.subheader("📅 Weekdays vs. weekends")

    col1, col2 = st.columns(2)

    wr = f.get("weekend_ratio", None)
    ms = f.get("morning_shift_hours", None)

    with col1:
        if wr is not None:
            # Plain-English label
            if wr > 1.2:
                label = f"🟡 {wr:.2f} — more on weekends"
                caption = "You use more electricity on weekends, likely because you're home more."
            elif wr < 0.85:
                label = f"🔵 {wr:.2f} — more on weekdays"
                caption = "Weekdays dominate — could be a home office or weekday-heavy appliances."
            else:
                label = f"🟢 {wr:.2f} — roughly equal"
                caption = "Your electricity use is similar on weekdays and weekends."

            st.metric(
                "Weekend ratio",
                label,
                help="Mean weekend hourly usage ÷ mean weekday hourly usage. 1.0 = equal use on both.",
            )
            st.caption(caption)
            _metric_with_bar(
                label="", value=wr, fmt=".2f",
                help_text="",
                reference=1.0, lo=0.5, hi=1.8,
                show_metric=False,
            )

    with col2:
        if ms is not None and not pd.isna(ms):
            if ms > 1.5:
                ms_label = f"🛌 +{ms:.1f} h — sleeping in"
                ms_caption = "You start your day noticeably later on weekends."
            elif ms < -0.5:
                ms_label = f"⏰ {ms:.1f} h — earlier start"
                ms_caption = "Weekend mornings start earlier than weekday ones."
            else:
                ms_label = f"✅ {ms:+.1f} h — same schedule"
                ms_caption = "Morning routine is similar on both weekdays and weekends."

            st.metric(
                "Weekend morning shift",
                ms_label,
                help="How many hours later the morning electricity peak falls on weekends vs weekdays.",
            )
            st.caption(ms_caption)


def _render_regularity(f: pd.Series):
    st.subheader("📊 How predictable is your household?")
    st.caption(
        "Routine score (0–1): how consistent your hour-by-hour usage is across different days. "
        "Close to 1 = clockwork schedule. Close to 0 = usage varies a lot."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        cv_d = f.get("cv_daily", None)
        if cv_d is not None:
            if cv_d < 0.3:
                cv_label, cv_color = "🟢 Very consistent", "#21c354"
            elif cv_d < 0.6:
                cv_label, cv_color = "🟡 Moderately variable", "#e6a817"
            else:
                cv_label, cv_color = "🔴 Highly variable", "#e05252"
            st.metric("Day-to-day consistency", cv_label,
                      help=f"Coefficient of Variation of daily totals = {cv_d:.2f}. Lower = more consistent.")
            st.caption(f"Technical value: CV = {cv_d:.2f}  (Low < 0.3 · Medium 0.3–0.6 · High > 0.6)")

    with col2:
        cv_h = f.get("cv_same_hour", None)
        if cv_h is not None:
            if cv_h < 0.3:
                cvh_label = "🟢 Very regular"
            elif cv_h < 0.6:
                cvh_label = "🟡 Moderately regular"
            else:
                cvh_label = "🔴 Irregular"
            st.metric("Hour-by-hour regularity", cvh_label,
                      help=f"Average CV across each hour of the day = {cv_h:.2f}. Are your mornings always similar?")
            st.caption(f"Technical value: CV = {cv_h:.2f}")

    with col3:
        rs = f.get("routine_score", None)
        if rs is not None:
            if rs > 0.70:
                rs_label = "🕐 Very routine"
                rs_caption = "Highly predictable — same schedule almost every day."
            elif rs > 0.45:
                rs_label = "〰️ Moderately routine"
                rs_caption = "Generally consistent, with some day-to-day variation."
            else:
                rs_label = "🎲 Unpredictable"
                rs_caption = "Usage varies a lot from day to day."

            st.metric("Routine score", f"{rs:.2f}  —  {rs_label}",
                      help="1 − CV(same hour). Closer to 1 = very routine household.")
            st.progress(float(rs), text=rs_caption)


def _render_peak_behaviour(f: pd.Series):
    st.subheader("⚡ Peak usage behaviour")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hop = f.get("hour_of_peak", None)
        if hop is not None:
            hour_label = f"{int(hop):02d}:00"
            if hop < 10:
                peak_type = "🌅 Morning peak"
            elif hop < 15:
                peak_type = "☀️ Midday peak"
            elif hop < 20:
                peak_type = "🌆 Evening peak"
            else:
                peak_type = "🌙 Night peak"
            st.metric("Hour of peak", hour_label,
                      help="The hour with the highest average electricity use.")
            st.caption(peak_type)

    with col2:
        ptm = f.get("peak_to_mean", None)
        if ptm is not None:
            if ptm > 3:
                ptm_label = "⚡ Very spiky"
                ptm_caption = "Usage is concentrated in short bursts."
            elif ptm > 2:
                ptm_label = "📈 Moderately spiky"
                ptm_caption = "Clear peaks but not extreme."
            else:
                ptm_label = "📉 Flat load"
                ptm_caption = "Usage is spread evenly through the day."
            st.metric("Peak intensity", f"{ptm:.1f}× average",
                      help="Peak hourly usage ÷ mean hourly usage.")
            st.caption(f"{ptm_label} — {ptm_caption}")

    with col3:
        ecs = f.get("evening_chores_score", None)
        if ecs is not None:
            if ecs > 1.3:
                ecs_label = "🧹 Active weekends"
                ecs_caption = "Fri/Sat evenings are busier — cooking, cleaning, entertaining."
            elif ecs < 0.8:
                ecs_label = "📅 Quieter weekends"
                ecs_caption = "Weekday evenings are actually busier."
            else:
                ecs_label = "⚖️ Similar both ways"
                ecs_caption = "Evening activity is similar on weekdays and weekends."
            st.metric("Weekend evenings", f"{ecs:.2f}",
                      help="Fri/Sat evening usage ÷ weekday evening usage.")
            st.caption(f"{ecs_label} — {ecs_caption}")

    with col4:
        nb = f.get("night_baseline_kwh", None)
        if nb is not None:
            if nb > 0.20:
                nb_label = "🔌 High standby"
                nb_caption = "Check always-on appliances — they add up over time."
            else:
                nb_label = "✅ Low standby"
                nb_caption = "Your overnight draw is minimal."
            st.metric("Overnight standby", f"{nb:.3f} kWh/h",
                      help="Mean hourly consumption 00:00–05:00. Captures always-on devices.")
            st.caption(f"{nb_label} — {nb_caption}")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_with_bar(label, value, fmt, help_text, reference, lo, hi,
                     show_metric: bool = True):
    """Show an optional metric label and a progress bar relative to a reference."""
    if show_metric and label:
        st.metric(label, f"{value:{fmt}}", help=help_text)
    pct     = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    ref_pct = max(0.0, min(1.0, (reference - lo) / (hi - lo)))
    st.progress(pct, text=f"reference (1.0) is at {ref_pct:.0%} of the scale")
