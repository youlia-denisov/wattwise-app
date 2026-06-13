"""
Electricity discount offers loader with weekly-refresh logic.

get_offers_df() is the single entry point used by the sidebar and the
Calculator tab. It returns a DataFrame of available plans, refreshing
from the scraper at most once a week.
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────────────────────
from config import DISCOUNT_OFFERS_FILE

ONE_WEEK_SECONDS = 7 * 24 * 3600


def get_offers_df() -> pd.DataFrame:
    """
    Return the electricity discount offers as a DataFrame.

    Logic:
      1. If the offers CSV exists and was last modified less than a week ago
         → read and return it immediately (no scraping needed).
      2. If the file is missing or older than a week → try to scrape fresh data
         and save the result back to disk for the next user.
      3. If the scraper module doesn't exist yet → fall back to whatever is on
         disk (even if stale). Better than crashing.
      4. If no file exists at all → return an empty DataFrame so the app
         doesn't crash, and show a warning in the sidebar.
    """
    file_is_fresh = (
        DISCOUNT_OFFERS_FILE.exists()
        and (time.time() - DISCOUNT_OFFERS_FILE.stat().st_mtime) < ONE_WEEK_SECONDS
        # stat().st_mtime = file's last-modified timestamp in seconds since 1970.
        # Subtracting from time.time() gives the file's age in seconds.
    )

    if file_is_fresh:
        return pd.read_csv(DISCOUNT_OFFERS_FILE)

    # File is stale or missing — try to scrape fresh data.
    try:
        from src.scraper import scrape_offers  # built separately; may not exist yet
        df = scrape_offers()
        DISCOUNT_OFFERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DISCOUNT_OFFERS_FILE, index=False)
        return df
    except ImportError:
        # Scraper not available — use the existing file even if it's old.
        if DISCOUNT_OFFERS_FILE.exists():
            return pd.read_csv(DISCOUNT_OFFERS_FILE)
        # No file at all — return empty so tabs degrade gracefully.
        st.warning("Offers file not found and scraper is not available. Discount tabs will be empty.")
        return pd.DataFrame()
