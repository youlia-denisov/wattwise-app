"""
Sidebar rendering for the Electricity Dashboard.

render_sidebar() draws everything in the left panel and returns a dict
with the user's settings so the main file can pass them to the tabs.
"""
import datetime
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from config import TARIFF
from src.loader import load_raw_csv
from src.preprocessing import clean_consumption_data
from app_offers import get_offers_df, DISCOUNT_OFFERS_FILE
from app_loaders import load_data, load_clustering_data, load_weather_data

# ── DEMO DATA PATHS ───────────────────────────────────────────────────────────
# Pre-generated synthetic data bundled with the app for recruiter demos.
_APP_ROOT   = Path(__file__).resolve().parent
DEMO_CSV    = _APP_ROOT / "data" / "demo" / "sample_electricity.csv"
DEMO_PROCESSED = _APP_ROOT / "data" / "demo" / "processed"
DEMO_TABLES    = _APP_ROOT / "data" / "demo" / "tables"
DEMO_REPORTS   = _APP_ROOT / "data" / "demo" / "reports"


def _activate_demo_mode(user_dir: Path) -> None:
    """
    Copy pre-generated demo processed files into the user's temp dir
    and load the demo raw CSV into session_state — so the full dashboard
    is visible immediately without any user action.
    """
    try:
        processed_dir = user_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        # Copy all pre-generated CSVs into the user's processed folder
        for src in DEMO_PROCESSED.glob("*.csv"):
            shutil.copy2(src, processed_dir / src.name)
        # Copy tables (discount scenarios)
        tables_dir = user_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for src in DEMO_TABLES.glob("*.csv"):
            shutil.copy2(src, tables_dir / src.name)
        # Copy report
        reports_dir = user_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src in DEMO_REPORTS.glob("*.md"):
            shutil.copy2(src, reports_dir / src.name)
        # Also copy the raw demo CSV so the pipeline button works if clicked
        shutil.copy2(DEMO_CSV, user_dir / "raw_input.csv")
        # Load the cleaned df into session state
        raw_df   = load_raw_csv(DEMO_CSV)
        clean_df = clean_consumption_data(raw_df)
        st.session_state["uploaded_df"]  = clean_df
        st.session_state["demo_mode"]    = True
        # Bust the loader cache so the copied files are picked up
        load_data.clear()
        load_clustering_data.clear()
    except Exception as e:
        st.warning(f"Demo mode could not load: {e}")


def _handle_file_upload(uploaded_file, user_dir: Path) -> None:
    """
    Save the uploaded CSV to disk, clean it, and store the result in session_state.

    Separated from render_sidebar so that file-processing logic is testable
    independently of the Streamlit UI layout.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader (guaranteed non-None).
    user_dir : Path
        This user's temp folder where raw_input.csv will be written.
    """
    raw_path = user_dir / "raw_input.csv"
    raw_path.write_bytes(uploaded_file.read())
    try:
        raw_df   = load_raw_csv(raw_path)
        clean_df = clean_consumption_data(raw_df)
        st.session_state["uploaded_df"] = clean_df
        st.success(f"✅ Loaded {len(clean_df):,} readings.")
    except Exception as e:
        st.error(f"Could not parse the file: {e}")


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
        st.header("⚡ Dashboard Setup")

        # ── DEMO MODE ─────────────────────────────────────────────────────────
        # Auto-activate demo mode on first load if the demo data is available
        # and the user hasn't uploaded their own file yet.
        if (
            "uploaded_df" not in st.session_state
            and DEMO_CSV.exists()
            and DEMO_PROCESSED.exists()
            and not st.session_state.get("demo_dismissed")
        ):
            _activate_demo_mode(user_dir)
            st.rerun()

        if st.session_state.get("demo_mode"):
            st.info(
                "📊 **Demo mode** — showing synthetic sample data.  \n"
                "Upload your own IEC CSV below to analyse your household.",
                icon="ℹ️",
            )
            if st.button("✕ Dismiss demo", use_container_width=True):
                st.session_state["demo_dismissed"] = True
                st.session_state.pop("demo_mode", None)
                st.session_state.pop("uploaded_df", None)
                load_data.clear()
                load_clustering_data.clear()
                st.rerun()

        # ── STEP 1: FILE UPLOAD ────────────────────────────────────────────────
        st.markdown("**Step 1 — Upload your file**")
        uploaded_file = st.file_uploader("Electricity CSV", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state.pop("demo_mode", None)   # switch out of demo mode
            _handle_file_upload(uploaded_file, user_dir)

        st.divider()

        # ── STEP 2: SETTINGS ───────────────────────────────────────────────────
        st.markdown("**Step 2 — Your settings**")

        sidebar_smart_meter = st.radio(
            "Do you have a smart meter?",
            ["Unknown", "Yes", "No"],
            index=0,
            horizontal=True,
            help="Smart meters are required for time-of-use plans. 'Unknown' shows all plans.",
        )
        sidebar_has_sm = {"Yes": True, "No": False, "Unknown": None}[sidebar_smart_meter]

        sidebar_tariff = st.number_input(
            "Electricity tariff (₪/kWh)",
            min_value=0.10,
            max_value=5.00,
            value=float(TARIFF),
            step=0.01,
            format="%.3f",
            help="Default is the current IEC rate. Change to model a different tariff.",
        )

        st.divider()

        # ── STEP 3: RUN PIPELINE ───────────────────────────────────────────────
        st.markdown("**Step 3 — Process your data**")
        if pipeline_available:
            file_ready = "uploaded_df" in st.session_state
            if st.button(
                "▶ Run Full Pipeline",
                disabled=not file_ready,
                help="Upload a CSV file first." if not file_ready else "Click to run analysis.",
                width="stretch",
            ):
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
                    load_data.clear()
                    load_clustering_data.clear()
                    load_weather_data.clear()
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Pipeline error: {e}")
        else:
            st.info("pipeline.py not found — pre-processed files will be loaded directly.")

        st.divider()

        # ── CUSTOMER TYPE ──────────────────────────────────────────────────────
        with st.expander("🏷️ Eligible plan types", expanded=False):
            _offers_df = get_offers_df()
            _all_customer_types = ["All"]
            if not _offers_df.empty and "customer_type" in _offers_df.columns:
                _ct = _offers_df["customer_type"].dropna().unique().tolist()
                _all_customer_types = sorted(set(_ct))

            if DISCOUNT_OFFERS_FILE.exists():
                _last_updated = datetime.datetime.fromtimestamp(
                    DISCOUNT_OFFERS_FILE.stat().st_mtime
                )
                st.caption(f"Offers last updated: **{_last_updated.strftime('%d %b %Y')}**")

            sidebar_customer_types = st.multiselect(
                "Which offers apply to you?",
                options=_all_customer_types,
                default=["All"],
                help=(
                    "Plans marked 'All' are always included. "
                    "Select additional types if you qualify (e.g. Cellcom subscriber)."
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
