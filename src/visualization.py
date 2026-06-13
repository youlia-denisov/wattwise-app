"""
This module generates the charts and graphs to help understand the electricity consumption patterns.
The main function is `save_all_visuals` which takes the cleaned data, aggregated stats, and detected outliers to produce a comprehensive set of visualizations.
The visualizations include:
- Heatmap of average consumption by weekday and hour.
- Heatmap of consumption variability (std) by weekday and hour.
- Bar chart of average hourly consumption by weekday with error bars.
- Bar chart of average daily consumption by weekday with error bars.
- Line chart of daily consumption trend with a 7-day rolling average.
- Load-duration curve to show the distribution of consumption values.
- Clustering (K-means) visualizations, including box plot and heatmap, with ranked electricity consumption. 

All charts are saved as interactive HTML files (by Plotly) in the specified output directory, and also displayed immediately for quick analysis.
The clustering visualizations are saved as PNG files for easy sharing and reporting.
"""

from pathlib import Path
import logging
import re
import traceback
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from config import WEEKDAY_ORDER
from src.discount_analysis import _load_plot_inputs, build_price_matrix, BASE_RATE

log = logging.getLogger(__name__)

def save_all_visuals(df: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame, outliers: pd.DataFrame, html_dir: Path) -> None:
    """Original Plotly visualizations - saved as interactive HTML."""
    html_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap of average consumption by weekday and hour
    pivot = df.pivot_table(index="weekday", columns="hour", values="kWh", aggfunc="mean").reindex(WEEKDAY_ORDER)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn_r", 
                    title="Average Consumption Heatmap — Weekday vs Hour")
    fig.update_layout(width=1150, height=620, xaxis_title="Hour", yaxis_title="Weekday")
    fig.update_traces(xgap=1, ygap=1)
    fig.write_html(html_dir / "heatmap_weekday_hour.html")

    # Heatmap of consumption variability (std) by weekday and hour
    std_pivot = hourly.pivot_table(index="weekday", columns="hour", values="std_kWh", aggfunc="mean").reindex(WEEKDAY_ORDER)
    fig = px.imshow(std_pivot, aspect="auto", color_continuous_scale="Oranges",
                    title="Consumption Variability Heatmap — Std by Weekday and Hour")
    fig.write_html(html_dir / "heatmap_variability_weekday_hour.html")

    # Bar chart of average hourly consumption by weekday with error bars
    fig = px.bar(hourly, x="hour", y="avg_kWh", color="weekday", error_y="std_kWh", barmode="group",
                 title="Average Hourly Consumption by Weekday", template="plotly_white")
    fig.update_layout(width=1250, height=680, xaxis=dict(dtick=1))
    fig.write_html(html_dir / "hourly_consumption_by_weekday.html")

    # Bar chart of average daily consumption by weekday with error bars
    fig = px.bar(daily, x="weekday", y="avg_daily_kWh", error_y="std_daily_kWh",
                 title="Average Daily Consumption by Weekday", template="plotly_white")
    fig.write_html(html_dir / "daily_consumption_distribution.html")

    # Line chart of daily consumption trend with a 7-day rolling average
    daily_totals = (df.groupby("date", as_index=False).agg(daily_kWh=("kWh", "sum")))
    daily_totals["rolling_7d"] = daily_totals["daily_kWh"].rolling(7, min_periods=1).mean()
    fig = px.line(daily_totals, x="date", y=["daily_kWh", "rolling_7d"],
                  title="Daily Consumption Trend with 7-Day Rolling Average", template="plotly_white")
    fig.write_html(html_dir / "daily_consumption_trend.html")

    # Load-duration curve
    load_curve = df[["kWh"]].sort_values("kWh", ascending=False).reset_index(drop=True)
    load_curve["time_percentile"] = (load_curve.index + 1) / len(load_curve) * 100
    fig = px.line(load_curve, x="time_percentile", y="kWh",
                  title="Load-Duration Curve", template="plotly_white")
    fig.write_html(html_dir / "load_duration_curve.html")

    # Outlier frequency heatmap
    if not outliers.empty:
        outliers = outliers.copy()
        outliers["weekday"] = outliers["datetime"].dt.day_name()
        outliers["hour"] = outliers["datetime"].dt.hour
        outlier_heatmap = (outliers.pivot_table(
            index="weekday", columns="hour", values="kWh", 
            aggfunc="count", fill_value=0)
            .reindex(WEEKDAY_ORDER)
        )
        
        fig = px.imshow(outlier_heatmap, aspect="auto", color_continuous_scale="Reds", 
                        text_auto=False, title="Outlier Frequency Heatmap — Weekday vs Hour")
        fig.update_layout(width=1150, height=620, xaxis_title="Hour", yaxis_title="Weekday")
        fig.update_traces(xgap=1, ygap=1)
        fig.write_html(html_dir / "outlier_frequency_heatmap.html")

# CLUSTERING VISUALIZATIONS

CLUSTER_PALETTE = {
    0: "#2ca25f",   # low use - green
    1: "#fee08b",   # medium-low - yellow
    2: "#fdae61",   # medium-high - orange
    3: "#d73027",   # high use - red
}

def _dominant_value(values: pd.Series):
    """
    Return the most frequent value in a group.
    Used for the dominant-cluster heatmap.
    """

    return values.mode().iloc[0]

def _ensure_cluster_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety helper.

    If df already has cluster_rank, use it.
    If not, create cluster_rank from average kWh per raw cluster.
    """

    df = df.copy()

    if "cluster_rank" in df.columns:
        return df

    cluster_order = (
        df.groupby("cluster")["kWh"]
        .mean()
        .sort_values()
        .index
    )

    rank_map = {
        cluster: rank
        for rank, cluster in enumerate(cluster_order)
    }

    df["cluster_rank"] = df["cluster"].map(rank_map)

    return df

def plot_elbow_curve(
    inertia_values: dict,
    chosen_k: int,
    output_path: Path,
) -> None:
    """
    Plot inertia vs. k (the elbow curve) and save as PNG.

    How to read this plot:
    - X-axis: number of clusters (k)
    - Y-axis: inertia — lower means more compact clusters
    - The curve drops steeply, then flattens
    - The "elbow" is the bend where improvement slows down
    - The red dashed line marks the chosen k (N_CLUSTERS)

    If the line sits at the elbow, the choice is well justified.
    If the curve hasn't bent yet at that k, consider a larger k.
    If the bend was earlier, consider a smaller k.
    """

    k_values = list(inertia_values.keys())
    inertia_list = list(inertia_values.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        k_values,
        inertia_list,
        marker="o",
        color="#2171b5",
        linewidth=2,
        markersize=7,
    )

    # Annotate each point with its inertia value for easy reading.
    for k, inertia in zip(k_values, inertia_list):
        ax.annotate(
            f"{inertia:,.0f}",
            xy=(k, inertia),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444444",
        )

    # Red dashed line marks the k used in the final model.
    ax.axvline(
        x=chosen_k,
        color="#d73027",
        linestyle="--",
        linewidth=1.5,
        label=f"Chosen k = {chosen_k}",
    )

    ax.set_title("Elbow Method — KMeans Inertia vs. Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_xticks(k_values)
    ax.legend()

    sns.despine()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("Saved elbow curve to %s", output_path)


def save_clustering_visuals(
    df_clustered: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]: # type: ignore
    """
    Save clustering visualizations.

    Parameters:
    - df_clustered:
        DataFrame returned by run_clustering().
        Expected columns:
        - kWh
        - hour
        - weekday
        - cluster
        - cluster_rank

    - figure_dir:
        Folder where PNG files should be saved.

    Returns:
    - list of generated plot paths
    """

    if df_clustered is None or df_clustered.empty:
        log.warning("No clustered data received. Skipping clustering visuals.")
        return []

    required_cols = ["kWh", "hour", "weekday", "cluster"]
    missing = [col for col in required_cols if col not in df_clustered.columns]

    if missing:
        log.warning("Missing clustering visualization columns: %s", missing)
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_cluster_rank(df_clustered)

    generated = []

    # 1. Cluster summary dashboard

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.boxplot(
        data=df,
        x="cluster_rank",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("kWh Distribution by Ranked Cluster")
    axes[0, 0].set_xlabel("Cluster rank: low → high usage")
    axes[0, 0].set_ylabel("kWh")

    hourly_profile = (
        df.groupby(["hour", "cluster_rank"], as_index=False)["kWh"]
        .mean()
    )

    sns.lineplot(
        data=hourly_profile, # type: ignore
        x="hour",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Average Hourly Profile by Ranked Cluster")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Average kWh")
    axes[0, 1].set_xticks(range(0, 24, 2))

    weekday_profile = (
        df.groupby(["weekday", "cluster_rank"], as_index=False)
        .agg(kWh=("kWh", "mean"))
    )

    weekday_profile["weekday"] = pd.Categorical(
        weekday_profile["weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    weekday_profile = weekday_profile.sort_values(["weekday", "cluster_rank"])

    sns.lineplot(
        data=weekday_profile,
        x="weekday",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Average Weekday Profile by Ranked Cluster")
    axes[1, 0].set_xlabel("Weekday")
    axes[1, 0].set_ylabel("Average kWh")
    axes[1, 0].tick_params(axis="x", rotation=35)

    cluster_sizes = (
        df["cluster_rank"]
        .value_counts()
        .sort_index()
    )

    axes[1, 1].bar(
        cluster_sizes.index.astype(str),
        cluster_sizes.values,
        color=[CLUSTER_PALETTE.get(i, "#aaa") for i in cluster_sizes.index],
    )
    axes[1, 1].set_title("Readings per Cluster Rank")
    axes[1, 1].set_xlabel("Cluster rank")
    axes[1, 1].set_ylabel("Number of readings")

    plt.suptitle("Clustering Summary Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()

    dashboard_path = figure_dir / "clustering_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(dashboard_path)
    log.info("Saved clustering dashboard to %s", dashboard_path)

    return generated


# ── DISCOUNT PLAN COMPARISON HEATMAPS ────────────────────────────────────────
# These were previously in src/discount_analysis.py. They live here because
# they produce image output — the same job as every other function in this file.

def _save_one_plot(
    consumption_matrix: pd.DataFrame,
    plan_matrix_df: pd.DataFrame,
    supplier: str,
    plan: str,
    discount: float,
    figure_dir: Path,
) -> str:
    """
    Render and save a side-by-side heatmap PNG for one discount plan.
    Returns the filename of the saved PNG (not the full path).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5.5))

    sns.heatmap(
        consumption_matrix, annot=False, fmt=".2f", cmap="Oranges", ax=ax1,
        cbar_kws={"label": "Usage (kWh)"},
    )
    ax1.set_title("Your Family's Real Consumption Profile", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    sns.heatmap(
        plan_matrix_df, annot=False, fmt=".3f", cmap="RdYlGn_r",
        vmin=BASE_RATE * 0.65, vmax=BASE_RATE, ax=ax2,
        cbar_kws={"label": "Tariff Rate (NIS)"},
    )
    ax2.set_title(f"{supplier} — {plan} ({discount}% Min Discount)", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", f"{supplier}_{plan}").lower()
    filename   = f"comparison_{safe_name}.png"
    plt.savefig(figure_dir / filename, dpi=150)
    plt.close()

    return filename


def generate_side_by_side_plots(
    has_smart_meter=None,
    processed_dir: Path | None = None,
    table_dir: Path | None = None,
    figure_dir: Path | None = None,
) -> list:
    """
    Save a side-by-side heatmap for each eligible plan.

    Parameters
    ----------
    has_smart_meter : bool or None
    processed_dir : Path  — per-user temp path (required)
    table_dir : Path      — per-user temp path (required)
    figure_dir : Path     — per-user temp path (required)
    """
    if processed_dir is None or table_dir is None or figure_dir is None:
        raise ValueError(
            "processed_dir, table_dir, and figure_dir must all be provided explicitly. "
            "Pass the per-user temp directory paths from the pipeline."
        )
    _processed_dir = Path(processed_dir)
    _table_dir     = Path(table_dir)
    _figure_dir    = Path(figure_dir)
    _figure_dir.mkdir(parents=True, exist_ok=True)

    consumption_matrix, unique_plans = _load_plot_inputs(_processed_dir, _table_dir, has_smart_meter)

    output_images = []
    for idx, row in unique_plans.iterrows():
        supplier    = row["supplier_name"]
        plan        = row["plan_name"]
        discount    = float(row["discount_pct"])
        restriction = row["time_restriction"]

        log.info("  [plot %d] %s — %s | discount=%.1f%% | restriction=%r",
                 idx, supplier, plan, discount, restriction)
        try:
            plan_matrix_df = build_price_matrix(restriction, discount)
            filename = _save_one_plot(
                consumption_matrix, plan_matrix_df,
                supplier, plan, discount, _figure_dir,
            )
            log.info("    saved %s", filename)
            output_images.append({"supplier": supplier, "plan": plan, "filename": filename})
        except Exception:
            log.error("    FAILED on plan '%s — %s':\n%s", supplier, plan, traceback.format_exc())
            plt.close()
            continue

    log.info("Saved %d comparison plots to %s", len(output_images), _figure_dir)
    return output_images
