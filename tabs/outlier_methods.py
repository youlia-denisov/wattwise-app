"""
outlier_methods.py
------------------
Streamlit tab: outlier detection for electricity consumption data.

Two methods are shown to users:
  1. IQR (Interquartile Range) — robust to skewed data; the default recommendation
  2. 3-Sigma (Z-Score) — works well when data is close to a bell curve

DBSCAN and Isolation Forest are still computed by the pipeline (outlier_pipeline.py)
but are not displayed here — they require domain expertise to interpret and are not
suitable for a general multi-user audience.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.outlier_pipeline import run_outlier_pipeline


# ── helpers ────────────────────────────────────────────────────────────────────

def _histogram_with_fences(kwh: pd.Series, lower: float, upper: float,
                            method_name: str, fence_color: str) -> go.Figure:
    """Histogram with vertical lines marking the outlier fences."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=kwh, nbinsx=60, name="All readings",
        marker_color="#4a90d9", opacity=0.7,
    ))
    if lower > kwh.min():
        fig.add_vline(x=lower, line_color="orange", line_dash="dash",
                      annotation_text="Lower fence", annotation_font_color="orange")
    if upper < kwh.max() * 1.5:
        fig.add_vline(x=upper, line_color=fence_color, line_dash="dash",
                      annotation_text="Upper fence", annotation_font_color=fence_color)
    fig.update_layout(
        xaxis_title="kWh per reading",
        yaxis_title="Number of readings",
        showlegend=False,
        margin=dict(t=20, b=30),
    )
    return fig


def _timeline_fig(df: pd.DataFrame, flagged_idx, color: str, method_name: str):
    """All readings over time; flagged ones in a different colour."""
    if "datetime" not in df.columns:
        return None
    df_plot = df[["datetime", "kWh"]].copy()
    df_plot["datetime"] = pd.to_datetime(df_plot["datetime"])
    df_plot["flagged"] = df_plot.index.isin(flagged_idx)
    fig = px.scatter(
        df_plot, x="datetime", y="kWh",
        color="flagged",
        color_discrete_map={False: "#aac4e8", True: color},
        labels={"flagged": "Flagged", "kWh": "kWh", "datetime": ""},
        opacity=0.6,
    )
    fig.update_layout(margin=dict(t=10, b=30), showlegend=True)
    return fig


# ── main render function ───────────────────────────────────────────────────────

def render_outlier_methods(df_clean: pd.DataFrame, consumption_col: str,
                           figure_dir: Path = None):

    st.header("Outlier Detection")
    st.markdown(
        "An **outlier** is a reading that looks unusual compared to the rest of your data — "
        "a sudden spike, a suspiciously low value, or a period of unexpected behaviour. "
        "Finding outliers helps you spot meter errors, appliance faults, or data quality issues."
    )

    # ── run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Analysing your data…"):
        try:
            res = run_outlier_pipeline(df_clean)
        except Exception as e:
            st.error(f"Could not run outlier detection: {e}")
            return

    kwh = pd.to_numeric(df_clean[consumption_col], errors="coerce").dropna()
    skew = res.stats["skewness"]
    n = res.stats["n_samples"]

    # ── recommendation banner ─────────────────────────────────────────────────
    st.subheader("Recommended method for your data")

    rec_method = res.recommended
    rec_color = "#1a7f4b" if rec_method == "IQR" else "#1a4f7f"

    st.markdown(
        f"""
        <div style="background:{rec_color}18; border-left:4px solid {rec_color};
                    padding:14px 18px; border-radius:6px; margin-bottom:8px;">
            <span style="font-size:1.15em; font-weight:700; color:{rec_color};">
                ✓ {rec_method}
            </span>
            <br/><span style="font-size:0.97em;">{res.reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What does skewness mean?"):
        st.markdown(
            f"**Skewness = {skew:.2f}** — this measures how lopsided your consumption data is.  \n"
            "A value near zero means most readings cluster around the average, like a bell curve.  \n"
            "A positive value means there are occasional high spikes pulling the average upward — "
            "common in electricity data.  \n\n"
            "| Skewness range | Meaning |\n"
            "|---|---|\n"
            "| −0.5 to 0.5 | Roughly symmetric — 3-Sigma works well |\n"
            "| 0.5 to 1.5 | Moderately skewed — IQR is safer |\n"
            "| Above 1.5 | Strongly skewed — IQR is clearly better |"
        )

    st.divider()

    # ── comparison bar ────────────────────────────────────────────────────────
    # Show only the two simple methods in the summary chart
    simple_methods = ["IQR", "3-Sigma"]
    summary_simple = res.summary[res.summary["Method"].isin(simple_methods)].copy()

    fig_bar = px.bar(
        summary_simple, x="Method", y="% of total",
        color="Method",
        color_discrete_map={"IQR": "#1a7f4b", "3-Sigma": "#1a4f7f"},
        text="Flagged",
        labels={"% of total": "% of readings flagged"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=10))

    col_a, col_b = st.columns(2)
    for method, col, color in [("IQR", col_a, "#1a7f4b"), ("3-Sigma", col_b, "#1a4f7f")]:
        row = summary_simple[summary_simple["Method"] == method]
        if not row.empty:
            flagged = int(row["Flagged"].values[0])
            pct = float(row["% of total"].values[0])
            rec_tag = " ✓ recommended" if method == rec_method else ""
            col.metric(f"{method}{rec_tag}", f"{flagged} readings ({pct:.1f}%)")

    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── METHOD: IQR ───────────────────────────────────────────────────────────
    st.subheader("Method 1 — IQR (Interquartile Range)")

    st.markdown(
        "**The idea in plain English:** Sort all your readings from lowest to highest. "
        "IQR looks at the middle half of that sorted list — from the 25th percentile (Q1) "
        "to the 75th percentile (Q3). Any reading that falls far outside that range is flagged.  \n\n"
        "This method is **not affected by extreme values** — even if you have a few giant spikes, "
        "they don't shift the fences. That makes it the most reliable choice for electricity data, "
        "which almost always has a right-skewed distribution."
    )

    if skew > 1.5:
        st.success(f"Your data has skewness {skew:.2f} — IQR is the best fit here.")
    elif skew > 0.5:
        st.info(f"Your data has skewness {skew:.2f} — IQR is the safer choice over 3-Sigma.")

    p_iqr = res.stats["iqr_params"]
    n_iqr = len(res.results_by_method["IQR"])
    pct_iqr = round(100 * n_iqr / n, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lower fence", f"{p_iqr['lower']:.2f} kWh")
    c2.metric("Upper fence", f"{p_iqr['upper']:.2f} kWh")
    c3.metric("Flagged readings", f"{n_iqr} ({pct_iqr}%)")

    # Box plot — the natural IQR visual
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=kwh, name="kWh", marker_color="#1a7f4b",
        boxpoints="outliers", jitter=0.3,
        hovertemplate="kWh: %{y:.3f}<extra></extra>",
    ))
    fig_box.update_layout(
        yaxis_title="kWh per reading",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "The box covers the middle 50% of readings. The whiskers extend to the IQR fences. "
        "Dots beyond the whiskers are flagged outliers."
    )

    tl = _timeline_fig(df_clean, res.results_by_method["IQR"].index, "#1a7f4b", "IQR")
    if tl:
        st.plotly_chart(tl, use_container_width=True)
        st.caption("Highlighted dots are the readings IQR flagged. Clusters in time may indicate a seasonal event, a faulty appliance, or a data recording issue.")

    st.divider()

    # ── METHOD: 3-Sigma ───────────────────────────────────────────────────────
    st.subheader("Method 2 — 3-Sigma (Standard Deviation Threshold)")

    st.markdown(
        "**The idea in plain English:** Calculate the average (mean) and spread (standard deviation) "
        "of all readings. Flag anything more than 3 standard deviations away from the average.  \n\n"
        "In a perfectly symmetric dataset, fewer than 0.3% of readings would be flagged. "
        "If you see a much higher percentage, your data has a long tail — and IQR will be more reliable."
    )

    if abs(skew) < 0.5:
        st.success(f"Your data has skewness {skew:.2f} — 3-Sigma is reliable here.")
    elif abs(skew) < 1.5:
        st.warning(
            f"Your data has skewness {skew:.2f}. 3-Sigma can still be useful, "
            "but the fences are pulled slightly upward by consumption spikes. IQR is more robust."
        )
    else:
        st.error(
            f"Your data has skewness {skew:.2f}. 3-Sigma is not well-suited here — "
            "the high spikes inflate the mean and standard deviation, making the upper fence too loose."
        )

    p_sig = res.stats["3sigma_params"]
    n_sig = len(res.results_by_method["3-Sigma"])
    pct_sig = round(100 * n_sig / n, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lower fence", f"{p_sig['lower']:.2f} kWh")
    c2.metric("Upper fence (mean + 3σ)", f"{p_sig['upper']:.2f} kWh")
    c3.metric("Flagged readings", f"{n_sig} ({pct_sig}%)")

    if pct_sig > 5:
        st.warning(
            f"3-Sigma flagged {pct_sig}% of your readings — far above the expected 0.3%. "
            "This confirms your data is too skewed for this method to be reliable."
        )

    fig_hist = _histogram_with_fences(kwh, p_sig["lower"], p_sig["upper"], "3-Sigma", "#1a4f7f")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "The dashed lines are the 3-sigma fences. Readings to the right of the upper fence are flagged. "
        "If the fence sits well inside the bulk of readings, the data is too skewed for this method."
    )

    tl = _timeline_fig(df_clean, res.results_by_method["3-Sigma"].index, "#1a4f7f", "3-Sigma")
    if tl:
        st.plotly_chart(tl, use_container_width=True)
        st.caption("Highlighted dots are the readings 3-Sigma flagged.")

    st.divider()

    # ── what to do next ───────────────────────────────────────────────────────
    st.subheader("What should you do with flagged readings?")
    st.markdown(
        "Flagged readings are **candidates for investigation**, not automatic errors.  \n\n"
        "- **High spikes** on weekday evenings → likely real heavy usage (cooking, AC, EV charging)\n"
        "- **High spikes** at 3 am → possible meter glitch or appliance left running accidentally\n"
        "- **Clusters of outliers** over days → check if a major appliance was running continuously\n"
        "- **Very low readings** → possible meter communication loss, not actual low consumption\n\n"
        f"The **{rec_method}** method flagged **{n_iqr if rec_method == 'IQR' else n_sig} readings** "
        f"({pct_iqr if rec_method == 'IQR' else pct_sig}% of your data). "
        "Start there."
    )
