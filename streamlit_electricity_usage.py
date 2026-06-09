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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Electricity Dashboard", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Consumption Analysis Dashboard")
st.markdown("### Smart analysis of your household electricity usage")
# Place to custom CSS to make the sidebar font a bit bigger and more readable.
st.markdown("""
<style>
[data-testid="stSidebar"] {
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)
# ── PROJECT ROOT & PYTHON PATH ────────────────────────────────────────────────
# ROOT is the repo folder (where this file lives).
# Adding ROOT and ROOT/src to sys.path lets us import config.py and src/ modules.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── SHARED CONFIG ─────────────────────────────────────────────────────────────
# config.py holds constants shared by all users (tariff, weekday order, etc.).
# Fallback values are used if config.py is missing on another machine.
try:
    from config import WEEKDAY_ORDER, TARIFF, DISCOUNT_OFFERS_FILE
except ImportError:
    WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    TARIFF = 0.666
    DISCOUNT_OFFERS_FILE = ROOT / "data" / "external" / "electricity_discount_offers.csv"

# ── SRC IMPORTS ───────────────────────────────────────────────────────────────
from src.discount_analysis import _hours_from_restriction, extract_weekdays, add_offer_eligibility

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
from app_loaders import detect_columns, safe_mean, safe_max

df_clean, hourly, daily, daily_totals, scenarios = load_data(str(PROCESSED_DIR))

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
consumption_col, date_col, daily_value_col = detect_columns(df_clean, daily_totals)

# ── OFFERS ────────────────────────────────────────────────────────────────────
from app_offers import get_offers_df

# ── TABS ──────────────────────────────────────────────────────────────────────
from tabs import (
    render_overview, render_hourly, render_trends,
    render_clustering, render_discounts, render_calculator,
    render_weather, render_report, render_about,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Hourly Patterns", "Trends & Outliers", "Clustering",
    "Discounts", "Calculator", "Weather", "Report", "About",
])

with tab1:
    render_overview(daily_totals, date_col, daily_value_col, df_clean, consumption_col,
                    safe_mean, safe_max)
with tab2:
    render_hourly(df_clean, hourly, consumption_col, WEEKDAY_ORDER)
with tab3:
    render_trends(df_clean, consumption_col)
with tab4:
    # Lambda defers the file read until the user actually opens this tab.
    render_clustering(lambda: load_clustering_data(str(PROCESSED_DIR)), WEEKDAY_ORDER)
with tab5:
    render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, sidebar["tariff"],
                     add_offer_eligibility, extract_weekdays, _hours_from_restriction)
with tab6:
    render_calculator(
        df_clean,
        get_offers_df(),
        sidebar["tariff"],
        has_smart_meter=sidebar["has_smart_meter"],
        customer_types=sidebar["customer_types"],
    )
with tab7:
    render_weather(df_clean)
with tab8:
    render_report(lambda: load_report(str(PROCESSED_DIR)), ROOT)
with tab9:
    render_about(df_clean, daily_totals, hourly)
