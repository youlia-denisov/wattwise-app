"""
Cached data-loading functions and small column-detection helpers
for the Electricity Dashboard.

All loader functions are decorated with @st.cache_data so Streamlit
only reads from disk when the data actually changes. The cache key is
always a STRING path — Path objects are harder for Streamlit to hash,
and each user's unique PROCESSED_DIR gives them their own cache entry
(so users never see each other's data).
"""
from pathlib import Path

import pandas as pd
import streamlit as st


# ── LOADERS ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(processed_dir: str):
    """Load all pipeline output CSVs from the user's processed folder.

    Returns (df_clean, hourly, daily, daily_totals, scenarios).
    Returns all-None tuple if any file is missing (triggers the welcome screen).
    """
    processed_dir = Path(processed_dir)
    try:
        df_clean     = pd.read_csv(processed_dir / "cleaned_consumption.csv")
        hourly       = pd.read_csv(processed_dir / "weekly_hourly_stats.csv")
        daily        = pd.read_csv(processed_dir / "daily_stats.csv")
        daily_totals = pd.read_csv(processed_dir / "daily_totals.csv")
        scenarios    = pd.read_csv(processed_dir.parent / "tables" / "discount_scenarios.csv")
        return df_clean, hourly, daily, daily_totals, scenarios
    except Exception:
        return None, None, None, None, None


@st.cache_data
def load_weather_data(processed_dir: str):
    """Load weather-merged consumption data, or return None if not yet generated."""
    path = Path(processed_dir) / "consumption_with_weather.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_clustering_data(processed_dir: str):
    """Load clustered consumption data and cluster summary, or None for each if missing."""
    c = Path(processed_dir) / "cleaned_consumption_clustered.csv"
    s = Path(processed_dir) / "cluster_rank_summary.csv"
    return (
        pd.read_csv(c) if c.exists() else None,
        pd.read_csv(s) if s.exists() else None,
    )


@st.cache_data
def load_report(processed_dir: str):
    """Return the markdown report text, or None if the pipeline hasn't run yet."""
    p = Path(processed_dir).parent / "summary_report.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ── COLUMN HELPERS ────────────────────────────────────────────────────────────
# The pipeline can save consumption under slightly different column names
# depending on version. These helpers find the right column defensively.

def detect_columns(df_clean: pd.DataFrame, daily_totals: pd.DataFrame):
    """
    Return (consumption_col, date_col, daily_value_col).

    Looks for recognisable keywords in column names so the app works even
    if the pipeline ever renames a column slightly.
    """
    consumption_col = next(
        (c for c in df_clean.columns if any(w in c.lower() for w in ["kwh", "kwatt", "consumption"])),
        df_clean.columns[1],
    )
    date_col = next(
        (c for c in daily_totals.columns if "date" in c.lower()),
        daily_totals.columns[0],
    )
    daily_value_col = next(
        (c for c in daily_totals.columns if any(w in c.lower() for w in ["kwh", "kwatt", "daily"])),
        daily_totals.columns[1],
    )
    return consumption_col, date_col, daily_value_col


# ── NUMERIC HELPERS ───────────────────────────────────────────────────────────
# Coerce to numeric before aggregating so stray strings don't raise errors.

def safe_mean(series):
    return pd.to_numeric(series, errors="coerce").mean()


def safe_max(series):
    return pd.to_numeric(series, errors="coerce").max()
