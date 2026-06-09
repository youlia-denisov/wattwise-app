import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sure src/ is importable when this module is loaded
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# These two helpers live at module level so @st.cache_data can manage them across reruns.
# Streamlit hashes DataFrame arguments by content, so the cache is invalidated automatically
# if the underlying data changes (e.g. after re-running the pipeline).

@st.cache_data
def _cached_build_pattern(monthly_kwh, pct_wd_day, pct_wd_evening, pct_wd_night, pct_weekend):
    """Builds the synthetic hourly DataFrame from slider values. Cached because it's called
    on every slider interaction and the result is deterministic."""
    from src.discount_calculator import build_custom_pattern_df
    return build_custom_pattern_df(monthly_kwh, pct_wd_day, pct_wd_evening, pct_wd_night, pct_weekend)


@st.cache_data
def _cached_compare_plans(calc_df, offers_df, tariff, has_sm, observation_days):
    """Runs the full plan comparison and annual extrapolation. The cache key includes
    all inputs, so changing the smart-meter toggle or sliders correctly triggers a recalc."""
    from src.discount_calculator import compare_all_plans, extrapolate_annual
    results = compare_all_plans(calc_df, offers_df, tariff=tariff, has_smart_meter=has_sm)
    return extrapolate_annual(results, observation_days=observation_days)


def _filter_offers_by_customer_type(offers_df, customer_types):
    """
    Keep only plans whose customer_type is in the selected set.
    If customer_types is None or empty, return all plans unchanged.
    Plans marked "All" are always retained regardless of the selection.
    """
    if not customer_types or "customer_type" not in offers_df.columns:
        return offers_df
    mask = offers_df["customer_type"].isin(customer_types)
    filtered = offers_df[mask].copy()
    n_hidden = len(offers_df) - len(filtered)
    if n_hidden > 0:
        st.info(
            f"Showing {len(filtered)} of {len(offers_df)} plans based on your customer type "
            f"selection in the sidebar. {n_hidden} plan(s) hidden — update "
            f"**'Which offers apply to you?'** in the sidebar to see them."
        )
    return filtered


def render_calculator(df_clean, offers_df, tariff, has_smart_meter=None, customer_types=None):
    """
    Interactive savings calculator with two modes:
      - "My data": uses the actual loaded consumption CSV
      - "Custom pattern": builds a synthetic profile from user sliders

    customer_types: list of customer_type values to include (from sidebar).
      Plans whose customer_type is in the list are shown; "All" plans are always included.
    """
    st.header("Savings Calculator")
    st.markdown(
        "Compare every available plan against your usage — showing real NIS saved, "
        "not just a score. Switch between your actual meter data and a hypothetical pattern."
    )

    # Filter offers by the customer types selected in the sidebar
    offers_df = _filter_offers_by_customer_type(offers_df, customer_types)

    mode = st.radio("Usage source", ["My meter data", "Custom pattern"], horizontal=True)
    calc_df, observation_days = _get_calc_df(df_clean, mode)
    if calc_df is None:
        return

    has_sm = _smart_meter_selector(has_smart_meter)

    if st.button("Calculate savings", type="primary"):
        with st.spinner("Calculating..."):
            results = _cached_compare_plans(calc_df, offers_df, tariff, has_sm, observation_days)
        if results.empty:
            st.warning("No plans to compare.")
            return
        _render_results(results, observation_days)


def _get_calc_df(df_clean, mode):
    """Return (calc_df, observation_days) for the chosen mode, or (None, None) on error."""
    if mode == "My meter data":
        if df_clean is None or df_clean.empty:
            st.warning("No consumption data loaded. Run the pipeline first.")
            return None, None
        calc_df = df_clean.rename(columns={
            c: "kWh" for c in df_clean.columns
            if any(w in c.lower() for w in ["kwh", "consumption"])
        })
        observation_days = (
            pd.to_datetime(calc_df["date"]).dt.date.nunique()
            if "date" in calc_df.columns else 30
        )
        st.caption(f"Using {len(calc_df):,} readings over {observation_days} days.")
        return calc_df, observation_days

    # Custom pattern mode
    st.markdown("**Set your typical monthly usage pattern**")
    col1, col2 = st.columns(2)
    with col1:
        monthly_kwh    = st.number_input("Monthly consumption (kWh)", min_value=10.0,
                                         max_value=2000.0, value=300.0, step=10.0)
        pct_wd_day     = st.slider("Weekday daytime (07-17)", 0, 100, 30,
                                   help="Sun-Thu 07:00-17:00")
        pct_wd_evening = st.slider("Weekday evening (17-23)", 0, 100, 35,
                                   help="Sun-Thu 17:00-23:00")
    with col2:
        pct_wd_night   = st.slider("Weekday night (23-07)", 0, 100, 15,
                                   help="Sun-Thu 23:00-07:00")
        pct_weekend    = st.slider("Weekend (Fri-Sat)", 0, 100, 20,
                                   help="All hours Fri-Sat")

    total_pct = pct_wd_day + pct_wd_evening + pct_wd_night + pct_weekend
    if total_pct == 0:
        st.error("At least one slider must be above 0.")
        return None, None
    if total_pct != 100:
        st.warning(
            f"Your sliders add up to **{total_pct}%**. "
            "They don't need to sum to exactly 100 — the calculator will scale them automatically — "
            "but make sure the proportions reflect how you actually use electricity."
        )
    else:
        st.success("Sliders sum to 100%.")

    calc_df = _cached_build_pattern(monthly_kwh, pct_wd_day, pct_wd_evening,
                                    pct_wd_night, pct_weekend)
    return calc_df, 30  # one synthetic month


def _smart_meter_selector(has_smart_meter):
    """Dropdown that inherits the sidebar setting but can be overridden per session."""
    sm_default = {True: "Yes", False: "No", None: "Unknown"}.get(has_smart_meter, "Unknown")
    choice = st.selectbox(
        "Do you have a smart meter?",
        ["Unknown", "Yes", "No"],
        index=["Unknown", "Yes", "No"].index(sm_default),
        help="Smart meter required for time-of-use plans. Inherits from sidebar setting.",
    )
    return {"Yes": True, "No": False, "Unknown": None}[choice]


def _render_results(results, observation_days):
    """Summary metrics, bar chart, colour legend, and detailed table."""
    best = results.iloc[0]
    eligible = results[results["eligibility"] != "not_eligible_requires_smart_meter"]
    best_eligible = eligible.iloc[0] if not eligible.empty else best

    m1, m2, m3 = st.columns(3)
    m1.metric("Best plan (eligible)",
              f"{best_eligible['supplier_name']} - {best_eligible['plan_name']}")
    m2.metric("Annual saving", f"NIS {best_eligible['annual_nis_saved']:,.0f}")
    m3.metric("Effective discount", f"{best_eligible['effective_discount_pct']:.1f}%")

    st.divider()
    _render_savings_chart(results)

    st.caption(
        "Green = Eligible  |  Yellow = Eligibility unknown (smart meter unspecified)  "
        "|  Red = Not eligible (requires smart meter you don't have)"
    )
    st.divider()
    _render_comparison_table(results)

    st.caption(
        f"Savings are extrapolated from {observation_days} days of data to a full year. "
        "Actual savings depend on your annual consumption pattern and any plan fees not "
        "reflected in the discount percentage."
    )


def _render_savings_chart(results):
    color_map = {
        "eligible":                              "#2ecc71",
        "eligible_or_unknown":                   "#f39c12",
        "unknown_smart_meter_required":          "#95a5a6",
        "not_eligible_requires_smart_meter":     "#e74c3c",
    }
    results = results.copy()
    results["color"] = results["eligibility"].map(lambda e: color_map.get(e, "#aaaaaa"))
    results["label"] = results["supplier_name"] + " - " + results["plan_name"]

    fig = go.Figure(go.Bar(
        x=results["annual_nis_saved"],
        y=results["label"],
        orientation="h",
        marker_color=results["color"],
        text=results["annual_nis_saved"].apply(lambda v: f"NIS {v:,.0f}"),
        textposition="outside",
        customdata=results[["effective_discount_pct", "matching_usage_share_pct",
                             "eligibility", "discount_pct"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Annual saving: NIS %{x:,.0f}<br>"
            "Advertised discount: %{customdata[3]}%<br>"
            "Effective discount on your usage: %{customdata[0]:.1f}%<br>"
            "Usage in discount window: %{customdata[1]:.1f}%<br>"
            "Eligibility: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Projected Annual Savings per Plan",
        xaxis_title="Estimated annual saving (NIS)",
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        height=max(400, len(results) * 32),
        margin={"l": 260, "r": 80},
    )
    st.plotly_chart(fig, width="stretch")


def _render_comparison_table(results):
    st.subheader("Detailed comparison")
    display_cols = [
        "supplier_name", "plan_name", "discount_pct",
        "weekdays_applicable", "hours_applicable",
        "matching_usage_share_pct", "effective_discount_pct",
        "annual_nis_saved", "eligibility",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    st.dataframe(
        results[display_cols].rename(columns={
            "supplier_name":             "Supplier",
            "plan_name":                 "Plan",
            "discount_pct":              "Advertised %",
            "weekdays_applicable":       "Days",
            "hours_applicable":          "Hours",
            "matching_usage_share_pct":  "Usage in window %",
            "effective_discount_pct":    "Effective discount %",
            "annual_nis_saved":          "Annual saving (NIS)",
            "eligibility":               "Eligibility",
        }).round(2),
        width="stretch",
    )
