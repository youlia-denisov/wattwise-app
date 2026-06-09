"""
Sidebar rendering for the Electricity Dashboard.

render_sidebar() draws everything in the left panel and returns a dict
with the user's settings so the main file can pass them to the tabs.
"""
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loader import load_raw_csv
from src.preprocessing import clean_consumption_data
from app_offers import get_offers_df, DISCOUNT_OFFERS_FILE

# ── CONFIG FALLBACK ───────────────────────────────────────────────────────────
try:
    from config import TARIFF
except ImportError:
    TARIFF = 0.666


def render_sidebar(user_dir: Path, pipeline_available: bool, run_pipeline_fn) -> dict:
    """
    Draw the sidebar and return the user's current settings.

    Parameters
    ----------
    user_dir : Path
        This user's temp folder (created once per browser session in main).
    pipeline_available : bool
        Whether pipeline.py was successfully imported.
    run_pipeline_fn : callable or None
        The run_pipeline function, or None if not available.

    Returns
    -------
    dict with keys:
        tariff          – float, electricity rate in ₪/kWh
        has_smart_meter – True / False / None
        customer_types  – list of strings
    """
    with st.sidebar:
        st.header("Controls")

        # ── FILE UPLOADER ──────────────────────────────────────────────────────
        # IEC files are in Hebrew with ~11 metadata rows at the top, so we can't
        # use pd.read_csv directly. Instead:
        #   1. Save raw bytes to a temp file so load_raw_csv can scan it line-by-line.
        #   2. load_raw_csv finds the real header row and returns date/time/kWh columns.
        #   3. clean_consumption_data parses dates, adds hour/weekday/month columns.
        uploaded_file = st.file_uploader("Upload your electricity CSV", type="csv")
        if uploaded_file is not None:
            raw_path = user_dir / "raw_input.csv"
            raw_path.write_bytes(uploaded_file.read())
            try:
                raw_df = load_raw_csv(raw_path)
                clean_df = clean_consumption_data(raw_df)
                st.session_state["uploaded_df"] = clean_df
                st.success(f"Loaded {len(clean_df):,} readings.")
            except Exception as e:
                st.error(f"Could not parse the file: {e}")

        st.divider()
        st.subheader("Settings")

        # ── TARIFF ─────────────────────────────────────────────────────────────
        sidebar_tariff = st.number_input(
            "Electricity tariff (₪/kWh)",
            min_value=0.10,
            max_value=5.00,
            value=float(TARIFF),
            step=0.01,
            format="%.3f",
            help="Default is taken from config.py. Change here to model a different rate.",
        )

        # ── SMART METER ────────────────────────────────────────────────────────
        sidebar_smart_meter = st.radio(
            "Do you have a smart meter?",
            ["Unknown", "Yes", "No"],
            index=0,
            help="Smart meters are required for time-of-use plans. 'Unknown' shows all plans.",
            horizontal=True,
        )
        # Map the string choice to True / False / None for use in tab logic.
        sidebar_has_sm = {"Yes": True, "No": False, "Unknown": None}[sidebar_smart_meter]

        # ── PIPELINE BUTTON ────────────────────────────────────────────────────
        st.divider()
        if pipeline_available:
            if st.button("Run Full Pipeline"):
                if "uploaded_df" not in st.session_state:
                    st.warning("Please upload a CSV file first.")
                else:
                    progress_bar = st.progress(0, text="Starting pipeline…")
                    status_text = st.empty()

                    def on_progress(step: int, total: int, label: str):
                        pct = int(step / total * 100)
                        progress_bar.progress(pct, text=f"Step {step}/{total} — {label}")
                        status_text.caption(label)

                    try:
                        run_pipeline_fn(
                            input_file=user_dir / "raw_input.csv",
                            output_dir=user_dir,
                            run_weather=False,
                            has_smart_meter=sidebar_has_sm,
                            on_progress=on_progress,
                        )
                        progress_bar.progress(100, text="✅ Done!")
                        status_text.empty()
                        st.success("Pipeline completed!")
                        st.cache_data.clear()  # force reload of newly written files
                        st.rerun()
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Pipeline error: {e}")
        else:
            st.info("pipeline.py not found. Pre-processed files will be loaded directly.")

        # ── CUSTOMER TYPE ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Customer type")

        _offers_df = get_offers_df()
        _all_customer_types = ["All"]
        if not _offers_df.empty and "customer_type" in _offers_df.columns:
            _ct = _offers_df["customer_type"].dropna().unique().tolist()
            _all_customer_types = sorted(set(_ct))

        if DISCOUNT_OFFERS_FILE.exists():
            _last_updated = datetime.datetime.fromtimestamp(DISCOUNT_OFFERS_FILE.stat().st_mtime)
            st.caption(f"Offers data last updated: **{_last_updated.strftime('%d %b %Y')}**")

        sidebar_customer_types = st.multiselect(
            "Which offers apply to you?",
            options=_all_customer_types,
            default=["All"],
            help=(
                "Plans marked 'All' are always included. "
                "Select additional types if you qualify (e.g. you are a Cellcom subscriber)."
            ),
        )
        # Always keep "All" so universal plans are never hidden.
        if "All" not in sidebar_customer_types:
            sidebar_customer_types = ["All"] + sidebar_customer_types

    return {
        "tariff": sidebar_tariff,
        "has_smart_meter": sidebar_has_sm,
        "customer_types": sidebar_customer_types,
    }
