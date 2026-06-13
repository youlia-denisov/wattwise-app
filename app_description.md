# WattWise — App Description

**WattWise** is a Streamlit dashboard that lets households upload their electricity consumption CSV
(exported from the Israel Electric Company), run a full analysis pipeline, and explore
interactive charts for usage patterns, outliers, clustering, weather correlation, and discount savings.

Each browser session is isolated: every user gets their own temporary folder so no data is ever shared.

---

## Architecture — Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  UI LAYER                                                        │
│  streamlit_electricity_usage.py  ·  app_sidebar.py  ·  tabs/    │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER                                             │
│  pipeline.py  ·  app_loaders.py  ·  app_offers.py               │
├─────────────────────────────────────────────────────────────────┤
│  PROCESSING LAYER  (src/)                                        │
│  loader · preprocessing · aggregation · clustering              │
│  outliers · outlier_pipeline · features · discount_analysis     │
│  discount_calculator · visualization · weather_analysis         │
│  reporting · scraper · text_parsers · scaling                   │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                      │
│  User temp dir (processed CSVs)  ·  config paths  ·  Open-Meteo │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modules

### Entry point

| File | Role |
|------|------|
| `streamlit_electricity_usage.py` | Page config, custom CSS/Plotly theme, per-user temp dir, tab routing |
| `config.py` | All project-wide constants: paths, tariff, time-window hours, clustering params, URLs |

### UI layer

| File | Role |
|------|------|
| `app_sidebar.py` | Sidebar widgets (upload, settings, pipeline trigger); returns a settings dict |
| `tabs/overview.py` | Daily totals chart, weekday heatmap, key metrics |
| `tabs/hourly.py` | Hourly patterns by weekday |
| `tabs/behavior_profile.py` | Persona + feature cards (time-of-day ratios, peak, regularity) |
| `tabs/behavior_components.py` | Reusable UI sub-components for the behavior tab |
| `tabs/trends.py` | Rolling average trend + IQR outlier timeline |
| `tabs/outlier_methods.py` | Side-by-side comparison of 4 outlier detection methods |
| `tabs/clustering.py` | KMeans cluster explorer with interactive k selector |
| `tabs/weather.py` | Temperature vs. consumption dual-axis charts |
| `tabs/discounts.py` | Discount eligibility + plan overlap heatmaps |
| `tabs/calculator.py` | Interactive savings calculator (real data or custom sliders) |
| `tabs/report.py` | Markdown report viewer + pre-generated HTML charts |
| `tabs/about.py` | Dataset metadata and app description |

### Orchestration layer

| File | Role |
|------|------|
| `pipeline.py` | Sequential 7-step pipeline: load → clean → cluster → stats → visuals → discounts → report |
| `app_loaders.py` | `@st.cache_data` wrappers for reading pipeline output CSVs; column-detection helpers |
| `app_offers.py` | Weekly-refresh logic for discount offers CSV; falls back to scraper or cached file |

### Processing layer (`src/`)

| File | Role |
|------|------|
| `loader.py` | Parse raw IEC CSV (auto-detect header row, kWh column) |
| `preprocessing.py` | Parse datetime, clean kWh, interpolate missing readings, add time columns |
| `aggregation.py` | Compute hourly stats, daily stats, daily totals, overall summary |
| `clustering.py` | Feature engineering → KMeans → cluster ranking → summary table |
| `features.py` | Build user feature vector (time-of-day ratios, peaks, regularity) + persona label |
| `outliers.py` | 3-sigma and IQR outlier detection |
| `outlier_pipeline.py` | 4-method pipeline (3σ, IQR, DBSCAN, Isolation Forest) + auto-method selector |
| `discount_analysis.py` | Match each offer's time window against actual consumption; compute eligible kWh |
| `discount_calculator.py` | Compute NIS savings per plan; annual extrapolation; synthetic pattern builder |
| `visualization.py` | Save Plotly charts as HTML; clustering visuals; side-by-side heatmaps |
| `weather_analysis.py` | Fetch from Open-Meteo API; merge with consumption; correlation plots |
| `reporting.py` | Write Markdown summary report (summary, outliers, weather, discounts) |
| `scraper.py` | Scrape kamaze.co.il supplier pages for discount offer data |
| `text_parsers.py` | Parse Hebrew time-restriction text into structured hour ranges |
| `israel_cities.py` | City name → lat/lon/timezone lookup |
| `scaling.py` | kWh normalization by weekday/hour (used for heatmap scaling) |

---

## Key Functions

### `app_sidebar.py`

```
render_sidebar(user_dir, pipeline_available, run_pipeline_fn) → dict
    Draws all sidebar widgets. Returns {"tariff", "has_smart_meter", "customer_types"}.

_handle_file_upload(uploaded_file, user_dir) → None
    Saves the CSV to disk, runs load_raw_csv + clean_consumption_data,
    stores result in st.session_state["uploaded_df"].
```

### `app_loaders.py`

```
load_data(processed_dir, _mtime) → (df_clean, hourly, daily, daily_totals, scenarios, error)
    Reads 5 pipeline output CSVs. Returns None-tuple + error string on failure.

load_weather_data(processed_dir, _mtime) → DataFrame | None
load_clustering_data(processed_dir, _mtime) → (clustered_df, summary_df) | (None, None)
load_report(processed_dir, _mtime) → str | None

_pipeline_cache_key(processed_dir) → float
    Returns mtime of cleaned_consumption.csv. Used to invalidate @st.cache_data
    automatically whenever the pipeline writes new files.

detect_columns(df_clean, daily_totals) → (consumption_col, date_col, daily_value_col)
    Finds correct column names defensively using keyword matching.
```

### `app_offers.py`

```
get_offers_df() → DataFrame
    Returns discount offers. Reads from disk if file is < 1 week old.
    Otherwise calls scraper.scrape_offers() and saves result to disk.
    Falls back gracefully if scraper is unavailable.
```

### `pipeline.py`

```
run_pipeline(run_weather, input_file, output_dir, has_smart_meter, on_progress) → None
    Full 7-step pipeline. Steps:
      1. load_raw_csv + clean_consumption_data → cleaned_consumption.csv
      2. run_clustering → cleaned_consumption_clustered.csv + cluster_rank_summary.csv
      3. compute_hourly_stats / compute_daily_stats / compute_daily_totals / detect_outliers → CSVs
      4. save_all_visuals + save_clustering_visuals → HTML + PNG figures
      5. add_weather (optional) → consumption_with_weather.csv + weather HTML charts
      6. estimate_discount_scenarios + generate_side_by_side_plots → discount_scenarios.csv + PNGs
      7. write_report → reports/summary_report.md
```

### `src/loader.py`

```
load_raw_csv(csv_path) → DataFrame
    Auto-detects the real header row in an IEC export; keeps only date/time/kWh columns.

detect_kwh_col(df) → str | None
    Finds the consumption column by scanning for "kwh"/"kwatt" keywords.

load_discount_offers(csv_path) → DataFrame
    Loads the offers CSV used by the discount analysis step.
```

### `src/preprocessing.py`

```
clean_consumption_data(df) → DataFrame
    Parses datetime, coerces kWh to float, interpolates short gaps,
    and adds: hour, weekday, month, date columns.
```

### `src/aggregation.py`

```
compute_hourly_stats(df) → DataFrame    # mean/std/min/max by weekday × hour
compute_daily_stats(df) → DataFrame     # mean/std/min/max by weekday
compute_daily_totals(df) → DataFrame    # total kWh per calendar date
compute_summary(df, hourly, daily) → dict   # scalar stats for the report
```

### `src/clustering.py`

```
run_clustering(input_path, output_path, summary_path, n_clusters, random_state, figure_dir)
    Loads cleaned data → encodes cyclical features → scales → fits KMeans →
    assigns human-readable rank labels → saves clustered CSV + rank summary.
```

### `src/features.py`

```
build_user_features(df) → Series
    Aggregates time-of-day ratios, weekday/weekend split, regularity, and peak stats
    into a single named Series describing the household's usage behaviour.

derive_persona(f) → str
    Maps feature values to one of several human-readable household personas.
```

### `src/outlier_pipeline.py`

```
run_outlier_pipeline(df) → OutlierResults
    Runs 3σ, IQR, DBSCAN, and Isolation Forest. Returns an object with results
    for all methods plus a recommended method chosen by _select_method().
```

### `src/discount_analysis.py`

```
estimate_discount_scenarios(df, offers, has_smart_meter) → DataFrame
    For each offer, counts how many kWh fall within its discount hours/days,
    then estimates the cost reduction at the configured tariff.

add_offer_eligibility(offers, has_smart_meter) → DataFrame
    Marks each offer as eligible/ineligible based on smart meter requirement.
```

### `src/discount_calculator.py`

```
calculate_plan_savings(df, offer, tariff) → dict
    Computes actual NIS savings for one plan against real consumption data.

compare_all_plans(df, offers, tariff, has_smart_meter) → DataFrame
    Runs calculate_plan_savings for every plan; returns ranked comparison table.

build_custom_pattern_df(monthly_kwh, pct_weekday_day, ...) → DataFrame
    Builds a synthetic hourly DataFrame from slider inputs for the calculator tab.
```

### `src/reporting.py`

```
write_report(summary, outliers, weather_summary, scenarios, recommendation,
             report_dir, df_clean, generated_plots) → Path
    Assembles a Markdown report from all pipeline outputs and saves it.
```

---

## Data Flow

### Startup

1. `streamlit_electricity_usage.py` creates a per-user temp dir in `st.session_state`.
2. `render_sidebar()` draws the upload widget and settings controls.
3. `load_data()` attempts to read processed CSVs from the user's temp dir.
   - If no files exist → welcome screen is shown and `st.stop()` halts the app.

### File Upload & Pipeline

```
User uploads CSV
  → _handle_file_upload()
      → src/loader.load_raw_csv()         # parse raw IEC format
      → src/preprocessing.clean_consumption_data()   # datetime, kWh, time columns
      → saved to session_state["uploaded_df"]

User clicks "Run Full Pipeline"
  → pipeline.run_pipeline(input_file=user_dir/raw_input.csv, output_dir=user_dir)
      Step 1: load + clean → processed/cleaned_consumption.csv
      Step 2: clustering   → processed/cleaned_consumption_clustered.csv
      Step 3: stats        → processed/weekly_hourly_stats.csv, daily_stats.csv, daily_totals.csv
                             processed/outliers_3sigma.csv, outliers_iqr.csv
      Step 4: visuals      → html/ (Plotly HTML files), figures/ (PNG heatmaps)
      Step 5: weather      → processed/consumption_with_weather.csv  [optional]
      Step 6: discounts    → tables/discount_scenarios.csv, figures/ (comparison PNGs)
      Step 7: report       → reports/summary_report.md
  → st.rerun() refreshes the app; load_data() now finds the new files
```

### Tab Rendering

```
streamlit_electricity_usage.py
  → load_data()        # reads all processed CSVs, cached by mtime
  → detect_columns()   # finds correct column names
  → get_offers_df()    # discount offers (weekly cache)
  → renders tabs based on view_mode ("Simple" or "Analyst")

Each tab function receives pre-loaded DataFrames and settings from the sidebar dict.
Tabs do not read files directly — they receive data as arguments.
```

### Discount Offer Refresh

```
get_offers_df()
  → check file age (< 1 week?) → return CSV directly
  → else → src/scraper.scrape_offers()
               → fetches kamaze.co.il pages
               → src/text_parsers.fill_time_restriction_from_context()
               → saves updated CSV
```

---

## External Dependencies

| Library | Used for |
|---------|----------|
| `streamlit` | UI framework, session state, caching |
| `pandas` | All data manipulation |
| `plotly` | Interactive charts |
| `scikit-learn` | KMeans clustering, DBSCAN, Isolation Forest, scaling |
| `requests` / `beautifulsoup4` | Web scraping (discount offers) |
| `open-meteo API` | Historical weather data |
| `numpy` | Numerical operations |
| `matplotlib` | Static PNG figure export |

---

## Multi-user Isolation

Each browser session creates a unique temp directory via `tempfile.mkdtemp()` stored in
`st.session_state["user_dir"]`. All pipeline outputs go inside that directory.
`@st.cache_data` uses the directory path as part of the cache key, so cache entries
are per-user. Users never access each other's uploaded files or results.
