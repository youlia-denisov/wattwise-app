"""
Electricity Consumption Analysis Dashboard — main entry point.

This file is intentionally thin: it sets up paths, loads data, and wires
the sidebar and tabs together. All logic lives in dedicated modules:

  app_sidebar.py  — sidebar UI and file upload
  app_loaders.py  — @st.cache_data functions and column helpers
  app_offers.py   — weekly-refresh logic for discount offers
  tabs/           — one file per dashboard tab
  src/            — data processing (loader, preprocessing, etc.)
  pipeline.py     — full analysis pipeline
"""
import sys
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path

# Plotly's default template is a bit plain, so we customize it to better match Streamlit's style.
import plotly.io as pio
import plotly.graph_objects as go

pio.templates["app_theme"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=13),
        paper_bgcolor="rgba(0,0,0,0)",   # transparent — inherits Streamlit background
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="#e9ecef", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="#e9ecef", gridwidth=1),
    )
)
pio.templates.default = "plotly+app_theme"


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(page_title="My Electricity Dashboard", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.title("⚡ WattWise ")

col_title, col_mode = st.columns([3, 1])
with col_title:
    st.markdown("### Household Electricity Consumption and Tariff Savings Analyzer")
with col_mode:
    view_mode = st.radio(
        "View mode",
        ["Simple", "Analyst"],
        index=0,
        horizontal=True,
        help="Simple: overview, discounts, and savings calculator. Analyst: all tabs.",
    )
# Place to custom CSS to make the sidebar font a bit bigger and more readable.
st.markdown("""
<style>
/* ── Global font ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Eggshell background ─────────────────────────────────── */
.stApp {
    background-color: #FAF7F1;
}

/* Keep sidebar slightly lighter */
[data-testid="stSidebar"] {
    background-color: #EDE7DA;
    font-size: 1.2rem;
}

/* ── App-like card frames around Plotly charts ───────────── */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    padding: 16px 12px 8px 12px;
    margin-bottom: 8px;
}

/* ── Metric cards get a subtle frame too ─────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.07);
    padding: 12px 16px;
}

/* ── Unified callout cards (st.info / warning / success / error) ── */
[data-testid="stAlert"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.08) !important;
    border: none !important;
    padding: 14px 18px !important;
}

/* Re-add left accent border per severity */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 4px solid #1c83e1 !important;  /* info  — blue   */
}
div[data-testid="stAlert"].st-emotion-cache-1y7vd64,
div[data-testid="stAlert"][kind="success"] {
    border-left: 4px solid #21c354 !important;  /* success — green */
}

/* Fallback: target by icon colour class that Streamlit injects */
/* info */
[data-testid="stAlert"]:has(svg[fill="#1c83e1"]),
[data-testid="stAlert"]:has(svg[color="#1c83e1"]) {
    border-left: 4px solid #1c83e1 !important;
}
/* success */
[data-testid="stAlert"]:has(svg[fill="#21c354"]),
[data-testid="stAlert"]:has(svg[color="#21c354"]) {
    border-left: 4px solid #21c354 !important;
}
/* warning */
[data-testid="stAlert"]:has(svg[fill="rgb(255, 227, 18)"]),
[data-testid="stAlert"]:has(svg[fill="#ffe312"]) {
    border-left: 4px solid #e6a817 !important;
}
/* error */
[data-testid="stAlert"]:has(svg[fill="rgb(255, 108, 108)"]),
[data-testid="stAlert"]:has(svg[fill="#ff6c6c"]) {
    border-left: 4px solid #e05252 !important;
}
</style>
""", unsafe_allow_html=True)

# ── PROJECT ROOT & PYTHON PATH ────────────────────────────────────────────────
# ROOT is the repo folder (where this file lives).
# Adding ROOT and ROOT/src to sys.path lets us import config.py and src/ modules.
ROOT = Path(__file__).resolve().parent

# ── SHARED CONFIG ─────────────────────────────────────────────────────────────
# config.py holds constants shared by all users (tariff, weekday order, etc.).
# Fallback values are used if config.py is missing on another machine.
from config import WEEKDAY_ORDER, TARIFF, DISCOUNT_OFFERS_FILE

# ── SRC IMPORTS ───────────────────────────────────────────────────────────────
from src.discount_analysis import add_offer_eligibility

# ── PIPELINE ──────────────────────────────────────────────────────────────────
try:
    from pipeline import run_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    run_pipeline = None
    PIPELINE_AVAILABLE = False

# ── PER-USER SESSION DIRECTORY ────────────────────────────────────────────────
# Each browser session gets its own temp folder so users never share data.
# The folder is created once and stored in session_state so it survives reruns.
if "user_dir" not in st.session_state:
    st.session_state["user_dir"] = tempfile.mkdtemp()

USER_DIR = Path(st.session_state["user_dir"])
PROCESSED_DIR = USER_DIR / "processed"
# By using the processed_dir as part of the cache key (via the function argument),
# we ensure that the cache is invalidated when the processed directory changes.

PROCESSED_DIR.mkdir(exist_ok=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
from app_sidebar import render_sidebar

sidebar = render_sidebar(
    user_dir=USER_DIR,
    pipeline_available=PIPELINE_AVAILABLE,
    run_pipeline_fn=run_pipeline,
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
from app_loaders import load_data, load_weather_data, load_clustering_data, load_report
from app_loaders import detect_columns, safe_mean, safe_max, _pipeline_cache_key

df_clean, hourly, daily, daily_totals, scenarios, load_error = load_data(
    str(PROCESSED_DIR), _pipeline_cache_key(PROCESSED_DIR)
)
if load_error is not None and _pipeline_cache_key(PROCESSED_DIR) != 0.0:
    # Only show the warning if the pipeline has run before (cache key != 0.0 means
    # the file existed at some point). On first open, stay silent.
    st.warning(
        "Some required processed files are missing or could not be loaded. "
        "Please run the pipeline first."
    )

# ── SIDEBAR DATASET SUMMARY ───────────────────────────────────────────────────
if df_clean is not None and "date" in df_clean.columns:
    with st.sidebar:
        st.divider()
        st.subheader("Dataset")
        dates = pd.to_datetime(df_clean["date"], errors="coerce").dropna()
        st.caption(
            f"Dates: **{dates.min().date()} to {dates.max().date()}**  \n"
            f"{len(df_clean):,} readings over {dates.dt.date.nunique()} days"
        )

# ── WELCOME SCREEN ────────────────────────────────────────────────────────────
# st.stop() halts execution here — nothing below runs until data is loaded.
if df_clean is None:
    st.markdown("---")
    st.subheader("Welcome! Let's get started.")
    st.markdown(
        "1. Upload your electricity CSV using the sidebar.\n"
        "2. Fill in **Do you have a smart meter?**\n"
        "3. Click **Run Full Pipeline** to process it.\n"
        "4. Depending on your file size, this may take a few minutes. Don't worry, there's a progress bar!\n\n"
        "Once done, all tabs will become available."
    )
    st.stop()

# ── COLUMN DETECTION ─────────────────────────────────────────────────────────
if daily_totals is None:
    st.error("Daily totals could not be generated.")
    st.stop()
    
consumption_col, date_col, daily_value_col = detect_columns(df_clean, daily_totals)

# ── OFFERS ────────────────────────────────────────────────────────────────────
from app_offers import get_offers_df

# ── TABS ──────────────────────────────────────────────────────────────────────
from tabs import (
    render_overview, render_hourly, render_trends,
    render_clustering, render_discounts, render_calculator,
    render_weather, render_report, render_about,
    render_behavior_profile, render_outlier_methods,
)

if view_mode == "Simple":
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", "My Profile", "Weather", "Calculate my savings", "Best deals", "Report", "About",
    ])

    with tab1:
        render_overview(daily_totals, date_col, daily_value_col, df_clean,
                        consumption_col, weekday_order=WEEKDAY_ORDER, simple=True)
    with tab2:
        render_behavior_profile(df_clean, simple=True)
    with tab3:
        render_weather(df_clean, simple=True)
    with tab4:        render_calculator(
            df_clean,
            get_offers_df(),
            sidebar["tariff"],
            has_smart_meter=sidebar["has_smart_meter"],
            customer_types=sidebar["customer_types"],
        )
    with tab5:
        render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, sidebar["tariff"])
    with tab6:
        render_report(lambda: load_report(str(PROCESSED_DIR), _pipeline_cache_key(PROCESSED_DIR)), USER_DIR, simple=True)
    with tab7:
        render_about(df_clean, daily_totals, hourly)

else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Overview", "Hourly Patterns", "Usage Habits", "Trends & Outliers",
        "Outlier Methods", "Clustering", "Weather", "Available Discounts",
        "Savings Calculator", "Report", "About",
    ])

    with tab1:
        render_overview(daily_totals, date_col, daily_value_col, df_clean, consumption_col)
    with tab2:
        render_hourly(df_clean, hourly, consumption_col, WEEKDAY_ORDER)
    with tab3:
        render_behavior_profile(df_clean)
    with tab4:
        render_trends(df_clean, consumption_col)
    with tab5:
        render_outlier_methods(df_clean, consumption_col)
    with tab6:
        render_clustering(lambda: load_clustering_data(str(PROCESSED_DIR)), WEEKDAY_ORDER)
    with tab7:
        render_weather(df_clean)
    with tab8:
        render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, sidebar["tariff"])
    with tab9:
        render_calculator(
            df_clean,
            get_offers_df(),
            sidebar["tariff"],
            has_smart_meter=sidebar["has_smart_meter"],
            customer_types=sidebar["customer_types"],
        )
    with tab10:
        render_report(lambda: load_report(str(PROCESSED_DIR), _pipeline_cache_key(PROCESSED_DIR)), USER_DIR)
    with tab11:
        render_about(df_clean, daily_totals, hourly)
