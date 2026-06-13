"""
Clustering logic for electricity consumption analysis.

This module is responsible only for:
1. Loading cleaned consumption data.
2. Preparing clustering features.
3. Applying cyclical encoding.
4. Scaling features.
5. Running KMeans.
6. Saving clustered data.
7. Ranking clusters by average electricity usage.

Visualizations are handled separately in visualization.py.
Reporting is handled separately in reporting.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from config import (
    N_CLUSTERS, RANDOM_STATE,
    DAY_START_HOUR, DAY_END_HOUR,
    EVENING_START_HOUR, EVENING_END_HOUR,
    NIGHT_START_HOUR,
)

log = logging.getLogger(__name__)

__all__ = ["run_clustering"]

# CONFIGURATION

# Paths are None by default — callers must pass explicit per-user paths.
CLEANED_FILE = None
CLUSTERED_FILE = None
CLUSTER_SUMMARY_FILE = None
FIGURE_DIR = None

# Range of k values tested by the elbow method.
# k=1 is excluded — one cluster means "no clustering".
# k=10 is a practical upper bound for a household consumption dataset.
ELBOW_K_RANGE = range(2, 11)


# DATA LOADING

def load_cleaned_data(csv_path: Path = CLEANED_FILE) -> pd.DataFrame:
    """
    Load cleaned electricity consumption data.

    Expected columns:
    - kWh
    - hour
    - weekday
    - month

    This file should already be created by preprocessing/pipeline.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {csv_path}\n"
            "Run preprocessing before clustering."
        )

    df = pd.read_csv(csv_path)

    required_cols = ["kWh", "hour", "weekday", "month"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for clustering: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df

# WEEKDAY CONVERSION

def add_weekday_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric weekday column for cyclical encoding.

    We keep:
    - weekday      = readable weekday label for plots/report
    - weekday_num  = numeric weekday for clustering

    Example:
        Sunday    -> 0
        Monday    -> 1
        ...
        Saturday  -> 6
    """
    weekday_map = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }
    df = df.copy()
    df["weekday_num"] = df["weekday"].map(weekday_map)
    if df["weekday_num"].isna().any():
        unknown = df.loc[df["weekday_num"].isna(), "weekday"].unique().tolist()
        raise ValueError(f"Unknown weekday labels found: {unknown}")
    return df

# FEATURE ENGINEERING

def apply_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features (hour, weekday, month) using sine/cosine
    so that the distance between e.g. hour 23 and hour 0 is small.

    New columns added:
    - hour_sin, hour_cos        (period 24)
    - weekday_sin, weekday_cos  (period 7)
    - month_sin, month_cos      (period 12)

    Original numeric columns are kept for reference.
    """
    df = df.copy()

    # Ensure numeric dtypes — CSV reads can leave these as strings.
    # "month" may arrive as "YYYY-MM" if preprocessing saved the period string
    # instead of extracting the month number — parse it defensively.
    df["hour"] = pd.to_numeric(df["hour"])
    df["weekday_num"] = pd.to_numeric(df["weekday_num"])
    if df["month"].dtype == object:
        df["month"] = pd.to_datetime(df["month"]).dt.month
    else:
        df["month"] = pd.to_numeric(df["month"])

    if not pd.api.types.is_numeric_dtype(df["hour"]):
        raise ValueError(f"Expected numeric hour column, got {df['hour'].dtype}")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday_num"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday_num"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily-aggregate features and merge them onto every hourly row.

    Why daily aggregates?
    A single hourly kWh reading doesn't describe a day's behaviour.
    Two days with the same total consumption can have very different shapes
    (flat all day vs. sharp evening spike). By computing these aggregates
    and attaching them to every row, KMeans can "see" the full-day context
    for each measurement — not just the value at that one hour.

    New columns added:
    - daily_total_kwh  : sum of all kWh readings for that calendar day
                         → separates high-demand days from low-demand days
    - evening_peak     : mean kWh between 17:00–22:00 for that day
                         → captures cooking / TV / lighting peak
    - night_baseline   : mean kWh between 23:00–06:00 for that day
                         → always-on devices (fridge, router) vs zero-use nights
    - peak_to_baseline : log(1 + evening_peak / (night_baseline + 0.01))
                         → shape of the day: high = sharp evening spike,
                            low = flat profile; log keeps extreme values in check

    Why NOT season or rolling averages?
    With only 13 weeks of data, seasonal windows span nearly the whole dataset
    and rolling averages smooth away the short-term patterns we want to find.
    """
    df = df.copy()

    # Keep 'date' as a string (YYYY-MM-DD) so the merge key type matches
    # the 'date' column already present in the CSV from preprocessing.
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

    # ── daily total ────────────────────────────────────────────────────────
    daily_total = (
        df.groupby("date")["kWh"]
        .sum()
        .rename("daily_total_kwh")
    )

    # ── evening peak (17:00–22:00) ─────────────────────────────────────────
    evening_peak = (
        df[df["hour"].between(EVENING_START_HOUR, EVENING_END_HOUR)]
        .groupby("date")["kWh"]
        .mean()
        .rename("evening_peak")
    )

    # ── night baseline (23:00–06:00) ───────────────────────────────────────
    night_baseline = (
        df[df["hour"].isin(list(range(NIGHT_START_HOUR, 24)) + list(range(0, DAY_START_HOUR)))]
        .groupby("date")["kWh"]
        .mean()
        .rename("night_baseline")
    )

    # ── assemble and compute ratio ─────────────────────────────────────────
    daily = pd.DataFrame({
        "daily_total_kwh": daily_total,
        "evening_peak":    evening_peak,
        "night_baseline":  night_baseline,
    })
    # np.log1p(x) = log(1 + x): compresses large ratios gracefully.
    # Epsilon 0.01 prevents division by zero on near-zero night readings.
    daily["peak_to_baseline"] = np.log1p(
        daily["evening_peak"] / (daily["night_baseline"] + 0.01)
    )

    # Merge back onto every hourly row that belongs to each day.
    df = df.merge(daily, on="date", how="left")

    n_missing = df[["daily_total_kwh", "evening_peak",
                    "night_baseline", "peak_to_baseline"]].isna().any(axis=1).sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} rows could not be matched to a daily aggregate. "
            "Check that the 'datetime' column parses correctly."
        )

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return the feature columns used for clustering.

    Features:
    - kWh              (consumption magnitude)
    - hour_sin/cos     (time of day, cyclical)
    - weekday_sin/cos  (day of week, cyclical)
    - daily_total_kwh  (full-day consumption context)
    - evening_peak     (evening demand for that day)
    - night_baseline   (overnight floor for that day)
    - peak_to_baseline (shape of the day's consumption curve)

    Removed vs. original:
    - month_sin/cos : only 3 months of data → near-zero variance, adds noise
    - is_weekend    : was monopolising k=2 split; weekday_sin/cos already
                      capture within-week variation
    """
    feature_cols = [
        "kWh",
        "hour_sin",    "hour_cos",
        "weekday_sin", "weekday_cos",
        "daily_total_kwh",
        "evening_peak",
        "night_baseline",
        "peak_to_baseline",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[feature_cols]


# ELBOW METHOD

def compute_elbow_inertias(
    X_scaled: np.ndarray,
    k_range: range = ELBOW_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Fit KMeans for each k in k_range and return a dict of {k: inertia}.

    Inertia = within-cluster sum of squared distances.
    Lower is better, but the gain should diminish after the "elbow".
    """
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_scaled)
        inertias[k] = km.inertia_
    return inertias

# CLUSTERING

def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    """
    Fit and return a KMeans model on the scaled feature matrix.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X_scaled)
    return km


def assign_cluster_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a cluster_rank column where rank 0 = lowest average kWh,
    rank (n-1) = highest average kWh.

    This makes clusters interpretable across runs, because raw KMeans
    cluster labels are arbitrary and change between fits.
    """
    df = df.copy()
    cluster_order = (
        df.groupby("cluster")["kWh"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    rank_map = {cluster: rank for rank, cluster in enumerate(cluster_order)}
    df["cluster_rank"] = df["cluster"].map(rank_map)
    return df


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table with one row per cluster_rank.

    Columns:
    - cluster_rank
    - mean_kWh
    - median_kWh
    - std_kWh
    - count
    - label  (human-readable: Low / Medium-Low / Medium-High / High)
    """
    summary = (
        df.groupby("cluster_rank")["kWh"]
        .agg(mean_kWh="mean", median_kWh="median", std_kWh="std", count="count")
        .reset_index()
    )

    n = len(summary)
    labels = _rank_labels(n)
    summary["label"] = summary["cluster_rank"].map(dict(enumerate(labels)))
    return summary


def _rank_labels(n: int) -> list:
    """
    Return n human-readable labels ordered from lowest to highest.
    Handles 2-5 clusters gracefully.
    """
    all_labels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
    if n <= len(all_labels):
        indices = [round(i * (len(all_labels) - 1) / (n - 1)) for i in range(n)]
        return [all_labels[i] for i in indices]
    return [f"Rank {i}" for i in range(n)]


# MAIN ENTRY POINT

def run_clustering(
    input_path: Path = CLEANED_FILE,
    output_path: Path = CLUSTERED_FILE,
    summary_path: Path = CLUSTER_SUMMARY_FILE,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
    figure_dir: Path = FIGURE_DIR,
) -> pd.DataFrame:
    """
    Full clustering pipeline. Called by pipeline.py.

    Steps:
    1. Load cleaned data.
    2. Add weekday number.
    3. Add daily aggregate features.
    4. Apply cyclical encoding.
    5. Scale features with RobustScaler.
    6. (Optional) Compute and save elbow curve data if figure_dir is given.
    7. Fit KMeans.
    8. Assign cluster ranks.
    9. Save clustered CSV and summary CSV.

    Parameters
    ----------
    input_path   : Path to cleaned_consumption.csv
    output_path  : Where to save the clustered CSV
    summary_path : Where to save the cluster rank summary CSV
    n_clusters   : Number of KMeans clusters (from config)
    random_state : For reproducibility
    figure_dir   : If provided, elbow inertia data is saved here as CSV
                   (actual plot is drawn in visualization.py)

    Returns
    -------
    df : DataFrame with cluster and cluster_rank columns added
    """
    # 1. Load
    df = load_cleaned_data(input_path)

    # 2. Weekday number
    df = add_weekday_number(df)

    # 3. Daily aggregate features
    # Must run before cyclical encoding so 'hour' is still a plain integer
    # (between() comparisons need numeric hour, not sin/cos).
    df = add_daily_features(df)

    # 4. Cyclical encoding
    df = apply_cyclical_encoding(df)

    # 5. Scale
    X = build_feature_matrix(df)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Elbow method (data only — plot is drawn in visualization.py)
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        inertias = compute_elbow_inertias(X_scaled, random_state=random_state)
        elbow_df = pd.DataFrame(list(inertias.items()), columns=["k", "inertia"])
        elbow_df.to_csv(figure_dir / "elbow_inertias.csv", index=False)
        log.info("  Elbow inertia data saved to %s", figure_dir / "elbow_inertias.csv")

    # 7. Fit KMeans
    km = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    df["cluster"] = km.labels_

    # 8. Rank clusters by mean kWh
    df = assign_cluster_ranks(df)

    # 9. Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = build_cluster_summary(df)
    summary.to_csv(summary_path, index=False)

    log.info("Clustered data saved to %s", output_path)
    log.info("Cluster summary saved to %s", summary_path)

    return df
