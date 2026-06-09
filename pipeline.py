"""
Main pipeline script for electricity consumption analysis.
This orchestrates the entire workflow:
1. Checks and creates necessary output directories.
2. Checks the presence of smart electric meter data and adjusts analysis accordingly.
3. Load raw consumption data from CSV.
4. Clean and preprocess the data.
5. Run KMeans clustering
6. Compute stats (daily/hourly) + detect outliers.
7. Generate and save visualizations.
8. (Optional) Perform weather analysis and discount scenario estimation.
9. Save all outputs (cleaned data, stats, visuals) to organized folders.
10. Generates report, with most important highlights and recommendations
The main function is `run_pipeline()` which executes all steps in sequence, with conditional execution of weather module
This script is designed to be run as a standalone program, and it will create all necessary output
folders if they don't exist.
The visualizations are saved as interactive HTML files, the cleaned data
and stats are saved as CSV files for further analysis or reporting.

Additional option: visualization of the pipeline using streamlit (the code is also available in this project "app/streamlit_electricity_usage.py)
"""
import sys
import logging
from pathlib import Path
import config
from src.loader import load_raw_csv, load_discount_offers
from src.preprocessing import clean_consumption_data
from src.aggregation import compute_hourly_stats, compute_daily_stats, compute_daily_totals, compute_summary
from src.outliers import detect_outliers_3sigma, detect_outliers_iqr, calculate_outlier_summary
from src.visualization import save_all_visuals, save_clustering_visuals
from src.weather_analysis import add_weather, summarize_weather, save_weather_plots
from src.discount_analysis import estimate_discount_scenarios, choose_recommendation, generate_side_by_side_plots, get_user_smart_meter_status
from src.reporting import write_report
from src.clustering import run_clustering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_pipeline(
    run_weather: bool = True,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    has_smart_meter: bool | None = None,
    on_progress=None,
):
    """
    Run the full analysis pipeline.

    Parameters
    ----------
    run_weather : bool
        Whether to fetch and analyse weather data (requires internet).
    input_file : Path or None
        Path to the raw IEC consumption CSV. If None, uses config.CONSUMPTION_FILE.
        Pass this when calling from Streamlit so each user's uploaded file is used.
    output_dir : Path or None
        Root folder for all outputs (processed CSVs, HTML, reports, etc.).
        If None, uses the paths defined in config (HTML_DIR, TABLE_DIR, etc.).
        Pass a per-user temp folder from Streamlit to keep users' data separate.
    """
    TOTAL_STEPS = 7

    def _progress(step: int, label: str):
        log.info("STEP %d/%d — %s", step, TOTAL_STEPS, label)
        if on_progress:
            on_progress(step, TOTAL_STEPS, label)

    log.info("Starting pipeline execution...")

    # ── RESOLVE PATHS ─────────────────────────────────────────────────────────
    csv_path = Path(input_file) if input_file is not None else config.CONSUMPTION_FILE
    log.info("Input file: %s", csv_path)

    if output_dir is not None:
        output_dir = Path(output_dir)
        processed_dir = output_dir / "processed"
        html_dir      = output_dir / "html"
        table_dir     = output_dir / "tables"
        report_dir    = output_dir / "reports"
        figure_dir    = output_dir / "figures"
    else:
        processed_dir = config.PROCESSED_DIR
        html_dir      = config.HTML_DIR
        table_dir     = config.TABLE_DIR
        report_dir    = config.REPORT_DIR
        figure_dir    = config.FIGURE_DIR

    log.info("Output dirs → processed=%s  html=%s  tables=%s", processed_dir, html_dir, table_dir)
    for folder in [processed_dir, html_dir, table_dir, report_dir, figure_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # ── SMART METER ───────────────────────────────────────────────────────────
    smart_meter_status = has_smart_meter if has_smart_meter is not None else config.HAS_SMART_METER
    if smart_meter_status is None and output_dir is None:
        print("\n[Input Prompt Required]")
        smart_meter_status = get_user_smart_meter_status()
    log.info("Smart meter status: %s", smart_meter_status)

    # ── LOAD & CLEAN ──────────────────────────────────────────────────────────
    _progress(1, "Loading and cleaning data…")
    raw = load_raw_csv(csv_path)
    df = clean_consumption_data(raw)
    log.info("  Loaded %d rows, columns: %s", len(df), df.columns.tolist())
    df.to_csv(processed_dir / "cleaned_consumption.csv", index=False)
    log.info("  Saved cleaned_consumption.csv")

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    _progress(2, "Running KMeans clustering…")
    df_clustered = run_clustering(
        input_path=processed_dir / "cleaned_consumption.csv",
        output_path=processed_dir / "cleaned_consumption_clustered.csv",
        summary_path=processed_dir / "cluster_rank_summary.csv",
        figure_dir=figure_dir,
    )
    log.info("  Clustering done. Clusters found: %s", df_clustered["cluster"].unique().tolist())

    # ── STATS & OUTLIERS ──────────────────────────────────────────────────────
    _progress(3, "Computing stats and detecting outliers…")
    hourly = compute_hourly_stats(df)
    daily = compute_daily_stats(df)
    daily_totals = compute_daily_totals(df)
    outliers_3sigma = detect_outliers_3sigma(df)
    outliers_iqr = detect_outliers_iqr(df)
    outlier_summary = calculate_outlier_summary(df, outliers_3sigma, outliers_iqr)

    hourly.to_csv(processed_dir / "weekly_hourly_stats.csv", index=False)
    daily.to_csv(processed_dir / "daily_stats.csv", index=False)
    daily_totals.to_csv(processed_dir / "daily_totals.csv", index=False)
    outliers_3sigma.to_csv(processed_dir / "outliers_3sigma.csv", index=False)
    outliers_iqr.to_csv(processed_dir / "outliers_iqr.csv", index=False)
    outlier_summary.to_csv(processed_dir / "outlier_summary.csv", index=False)
    log.info("  Stats saved. Outliers (3σ): %d  (IQR): %d", len(outliers_3sigma), len(outliers_iqr))

    # ── VISUALISATIONS ────────────────────────────────────────────────────────
    _progress(4, "Generating visualizations…")
    save_all_visuals(df, hourly, daily, outliers_3sigma, html_dir)
    log.info("  Main visuals saved to %s", html_dir)
    log.info("  Generating clustering visualizations...")
    save_clustering_visuals(df_clustered=df_clustered, figure_dir=figure_dir)
    log.info("  Clustering visuals saved.")

    # ── WEATHER (optional) ────────────────────────────────────────────────────
    weather_summary = None
    if run_weather:
        _progress(5, "Fetching weather data…")
        try:
            df_weather = add_weather(df)
            df_weather.to_csv(processed_dir / "consumption_with_weather.csv", index=False)
            weather_summary = summarize_weather(df_weather)
            save_weather_plots(df_weather, html_dir)
            log.info("  Weather analysis complete.")
        except Exception as error:
            log.warning("  Weather analysis skipped: %s", error)
    else:
        _progress(5, "Weather analysis skipped.")

    # ── DISCOUNTS ─────────────────────────────────────────────────────────────
    _progress(6, "Analysing discount scenarios…")
    log.info("  Loading offers from: %s", config.DISCOUNT_OFFERS_FILE)
    offers = load_discount_offers(config.DISCOUNT_OFFERS_FILE)
    log.info("  Offers loaded: %d rows, columns: %s", len(offers), offers.columns.tolist())

    log.info("  Estimating discount scenarios (smart_meter=%s)...", smart_meter_status)
    scenarios = estimate_discount_scenarios(df, offers, has_smart_meter=smart_meter_status)
    log.info("  Scenarios computed: %d rows", len(scenarios))
    scenarios.to_csv(table_dir / "discount_scenarios.csv", index=False, encoding="utf-8-sig")

    recommendation = choose_recommendation(scenarios)
    log.info("  Recommendation: %s", recommendation)

    log.info("  Generating tariff comparison plots...")
    generated_plots = generate_side_by_side_plots(
        has_smart_meter=smart_meter_status,
        processed_dir=processed_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )
    log.info("  Generated %d side-by-side plots.", len(generated_plots))

    # ── REPORT ────────────────────────────────────────────────────────────────
    import traceback as _tb
    _progress(7, "Writing report…")
    summary = compute_summary(df, hourly, daily)
    log.info("  Summary keys/types: %s", {k: type(v).__name__ for k, v in summary.items()})
    try:
        report_path = write_report(
            summary=summary,
            outliers=outlier_summary,
            weather_summary=weather_summary,
            scenarios=scenarios,
            recommendation=recommendation,
            report_dir=report_dir,
            generated_plots=generated_plots,
        )
    except Exception:
        log.error("  write_report FAILED:\n%s", _tb.format_exc())
        raise

    log.info("Pipeline completed successfully.")
    log.info("  Report:            %s", report_path)
    log.info("  HTML visuals:      %s", html_dir)
    log.info("  Processed data:    %s", processed_dir)
    log.info("  Comparison plots:  %s", figure_dir)
    log.info("  Discount CSV:      %s", table_dir / "discount_scenarios.csv")


if __name__ == "__main__":
    """ Options to disable weather analysis:
    - Command line: `python pipeline.py --no-weather`
    - Or set `run_weather=False` when calling `run_pipeline()`"""
    run_weather_flag = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--no-weather", "--skip-weather"]:
        run_weather_flag = False

    run_pipeline(run_weather=run_weather_flag)
