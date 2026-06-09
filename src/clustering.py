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

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from config import PROCESSED_DIR, N_CLUSTERS, RANDOM_STATE, FIGURE_DIR

__all__ = ["run_clustering"]

# CONFIGURATION

CLEANED_FILE = PROCESSED_DIR / "cleaned_consumption.csv"
CLUSTERED_FILE = PROCESSED_DIR / "cleaned_consumption_clustered.csv"
CLUSTER_SUMMARY_FILE = PROCESSED_DIR / "cluster_rank_summary.csv"

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

    assert df["hour"].dtype in [np.int64, np.float64], f"Expected numeric hour, got {df['hour'].dtype}"

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday_num"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday_num"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return the feature columns used for clustering.

    Features:
    - kWh              (consumption magnitude)
    - hour_sin/cos     (time of day, cyclical)
    - weekday_sin/cos  (day of week, cyclical)
    - month_sin/cos    (season, cyclical)
    """
    feature_cols = [
        "kWh",
        "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
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
    3. Apply cyclical encoding.
    4. Scale features with RobustScaler.
    5. (Optional) Compute and save elbow curve data if figure_dir is given.
    6. Fit KMeans.
    7. Assign cluster ranks.
    8. Save clustered CSV and summary CSV.

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

    # 3. Cyclical encoding
    df = apply_cyclical_encoding(df)

    # 4. Scale
    X = build_feature_matrix(df)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Elbow method (data only — plot is drawn in visualization.py)
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        inertias = compute_elbow_inertias(X_scaled, random_state=random_state)
        elbow_df = pd.DataFrame(list(inertias.items()), columns=["k", "inertia"])
        elbow_df.to_csv(figure_dir / "elbow_inertias.csv", index=False)
        print(f"  Elbow inertia data saved to {figure_dir / 'elbow_inertias.csv'}")

    # 6. Fit KMeans
    km = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    df["cluster"] = km.labels_

    # 7. Rank clusters by mean kWh
    df = assign_cluster_ranks(df)

    # 8. Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = build_cluster_summary(df)
    summary.to_csv(summary_path, index=False)

    print(f"  Clustered data saved to {output_path}")
    print(f"  Cluster summary saved to {summary_path}")

    return df
