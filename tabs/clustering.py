import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

CLUSTER_PALETTE = {0: "#2ca25f", 1: "#fee08b", 2: "#fdae61", 3: "#d73027"}
CLUSTER_LABELS  = {0: "Low use", 1: "Medium-low", 2: "Medium-high", 3: "High use"}


def render_clustering(load_clustering_data, WEEKDAY_ORDER):
    st.header("Consumption Clustering")
    st.markdown(
        "This analysis automatically groups all your hourly readings into 4 usage profiles, "
        "without being told what to look for. Each reading gets assigned to the profile it resembles most. "
        "The profiles are then ranked by how much electricity they use: "
        "**0 = your quietest hours** through to **3 = your heaviest-use hours**."
    )

    df_clustered, cluster_summary = load_clustering_data()

    if df_clustered is None:
        st.warning(
            "Usage profiling has not been run yet. "
            "Upload your CSV and click Run Full Pipeline to unlock this tab."
        )
        return

    c_col = next(
        (col for col in df_clustered.columns
         if any(w in col.lower() for w in ["kwh", "kwatt", "consumption"])),
        df_clustered.columns[0],
    )
    rank_col = "cluster_rank" if "cluster_rank" in df_clustered.columns else "cluster"
    df_clustered["_label"] = df_clustered[rank_col].map(CLUSTER_LABELS)

    _render_cluster_charts(df_clustered, c_col, rank_col)
    _render_cluster_heatmap(df_clustered, rank_col, WEEKDAY_ORDER)
    st.divider()
    _render_silhouette(df_clustered, rank_col)
    st.divider()
    _render_elbow(df_clustered)
    st.divider()
    _render_feature_centroids(df_clustered, c_col, rank_col)


# ── existing charts ────────────────────────────────────────────────────────────

def _render_cluster_charts(df_clustered, c_col, rank_col):
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(df_clustered, x=rank_col, y=c_col, color=rank_col,
                         color_discrete_map=CLUSTER_PALETTE,
                         title="Consumption Distribution by Cluster",
                         labels={rank_col: "Cluster (0=low to 3=high)", c_col: "kWh"},
                         category_orders={rank_col: [0, 1, 2, 3]})
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")
    with col2:
        size_df = df_clustered[rank_col].value_counts().sort_index().reset_index()
        size_df.columns = ["cluster_rank", "count"]
        fig_size = px.bar(size_df, x="cluster_rank", y="count", color="cluster_rank",
                          color_discrete_map=CLUSTER_PALETTE, title="Records per Cluster",
                          text="count")
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, width="stretch")


def _render_cluster_heatmap(df_clustered, rank_col, WEEKDAY_ORDER):
    if not {"weekday", "hour"}.issubset(df_clustered.columns):
        return
    st.subheader("Dominant Cluster by Weekday & Hour")
    dominant = (
        df_clustered.groupby(["weekday", "hour"])[rank_col]
        .agg(lambda x: x.mode().iloc[0])
        .unstack()
    )
    dominant = dominant.reindex([d for d in WEEKDAY_ORDER if d in dominant.index])
    colorscale = [
        [0.0,  CLUSTER_PALETTE[0]], [0.33, CLUSTER_PALETTE[1]],
        [0.66, CLUSTER_PALETTE[2]], [1.0,  CLUSTER_PALETTE[3]],
    ]
    fig_heat = go.Figure(go.Heatmap(
        z=dominant.values, x=dominant.columns.tolist(), y=dominant.index.tolist(),
        colorscale=colorscale, zmin=0, zmax=3,
        colorbar=dict(title="Cluster rank", tickvals=[0, 1, 2, 3],
                      ticktext=["0 Low", "1 Med-low", "2 Med-high", "3 High"]),
    ))
    fig_heat.update_layout(title="Dominant Cluster - Weekday x Hour",
                           xaxis_title="Hour", yaxis_title="Weekday", height=380)
    st.plotly_chart(fig_heat, width="stretch")


# ── silhouette analysis ────────────────────────────────────────────────────────

@st.cache_data
def _silhouette_scores(df_clustered: pd.DataFrame, rank_col: str):
    """
    Compute overall and per-sample silhouette scores.
    Uses the cyclical feature columns already in df_clustered.
    Subsamples to 5000 rows to keep it fast.
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import RobustScaler

    feature_cols = [c for c in ["kWh", "hour_sin", "hour_cos",
                                 "weekday_sin", "weekday_cos",
                                 "month_sin", "month_cos"]
                    if c in df_clustered.columns]
    if len(feature_cols) < 2 or df_clustered[rank_col].nunique() < 2:
        return None, None, None

    rng = np.random.RandomState(42)
    n = min(5000, len(df_clustered))
    idx = rng.choice(len(df_clustered), n, replace=False)
    sub = df_clustered.iloc[idx].reset_index(drop=True)

    X = RobustScaler().fit_transform(sub[feature_cols])
    labels = sub[rank_col].values

    overall = float(silhouette_score(X, labels))
    samples = silhouette_samples(X, labels)
    return overall, samples, labels


def _render_silhouette(df_clustered, rank_col):
    st.subheader("Silhouette Analysis")
    st.markdown(
        "The silhouette score measures how well each reading fits its assigned cluster. "
        "It ranges from **-1 to +1**: scores near +1 mean the reading is clearly "
        "in the right cluster; near 0 means it sits on the boundary between two clusters; "
        "negative means it might belong to a different cluster. "
        "If your clusters look similar, you will see low or negative scores here."
    )

    with st.spinner("Computing silhouette scores..."):
        overall, samples, labels = _silhouette_scores(df_clustered, rank_col)

    if overall is None:
        st.info("Not enough feature columns to compute silhouette scores.")
        return

    # Overall metric with interpretation
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Overall silhouette score", f"{overall:.3f}")
    with col2:
        if overall > 0.5:
            st.success("**Strong separation** - the 4 clusters are well-defined and distinct.")
        elif overall > 0.25:
            st.warning(
                "**Moderate separation** - clusters have some overlap. "
                "The algorithm found structure, but the boundaries are not crisp. "
                "Consider trying 2 or 3 clusters (see Elbow Curve below)."
            )
        else:
            st.error(
                "**Weak separation** - clusters overlap significantly. "
                "The data may not have 4 natural groups. "
                "The elbow curve below may suggest a better k."
            )

    # Per-cluster silhouette bar
    per_cluster = (
        pd.DataFrame({"score": samples, "cluster": labels})
        .groupby("cluster")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_silhouette"})
    )
    per_cluster["label"] = per_cluster["cluster"].map(CLUSTER_LABELS)
    per_cluster["color"] = per_cluster["cluster"].map(CLUSTER_PALETTE)

    fig_bar = px.bar(
        per_cluster, x="cluster", y="avg_silhouette",
        color="cluster", color_discrete_map=CLUSTER_PALETTE,
        text=per_cluster["avg_silhouette"].map(lambda v: f"{v:.3f}"),
        labels={"cluster": "Cluster", "avg_silhouette": "Avg silhouette score"},
        title="Average silhouette score per cluster",
    )
    fig_bar.add_hline(y=overall, line_dash="dash", line_color="grey",
                      annotation_text=f"Overall avg ({overall:.3f})")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, yaxis_range=[-0.1, 1.0])
    st.plotly_chart(fig_bar, width="stretch")
    st.caption(
        "Clusters with a score well below the overall average are the weak ones "
        "where readings could plausibly belong to a neighbouring cluster."
    )

    # Silhouette plot (sorted individual scores per cluster)
    with st.expander("Show full silhouette plot (individual readings)"):
        df_sil = pd.DataFrame({"score": samples, "cluster": labels})
        df_sil = df_sil.sort_values(["cluster", "score"]).reset_index(drop=True)
        df_sil["y"] = range(len(df_sil))
        df_sil["color"] = df_sil["cluster"].map(CLUSTER_PALETTE)
        df_sil["label"] = df_sil["cluster"].map(CLUSTER_LABELS)

        fig_sil = px.bar(
            df_sil, x="score", y="y", color="cluster",
            color_discrete_map=CLUSTER_PALETTE,
            orientation="h",
            labels={"score": "Silhouette score", "y": "", "cluster": "Cluster"},
            title="Silhouette plot - each bar is one reading",
            height=max(400, min(800, len(df_sil) // 5)),
        )
        fig_sil.update_traces(marker_line_width=0)
        fig_sil.update_layout(showlegend=True, yaxis_visible=False,
                               yaxis_showticklabels=False)
        fig_sil.add_vline(x=overall, line_dash="dash", line_color="grey",
                          annotation_text="Overall avg")
        st.plotly_chart(fig_sil, width="stretch")
        st.caption(
            "Each horizontal bar is one reading. Wide bars to the right = confidently in cluster. "
            "Bars to the left of 0 = possible misassignment."
        )


# ── elbow curve ────────────────────────────────────────────────────────────────

@st.cache_data
def _elbow_data(df_clustered: pd.DataFrame):
    """Fit KMeans for k=2..8 and return inertia + silhouette per k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import RobustScaler

    feature_cols = [c for c in ["kWh", "hour_sin", "hour_cos",
                                 "weekday_sin", "weekday_cos",
                                 "month_sin", "month_cos"]
                    if c in df_clustered.columns]
    if len(feature_cols) < 2:
        return None

    rng = np.random.RandomState(42)
    n = min(5000, len(df_clustered))
    idx = rng.choice(len(df_clustered), n, replace=False)
    X = RobustScaler().fit_transform(df_clustered.iloc[idx][feature_cols])

    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if k > 1 else float("nan")
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def _render_elbow(df_clustered):
    st.subheader("Elbow Curve - How many clusters is best?")
    st.markdown(
        "This chart runs K-Means for k=2 to 8 and shows two metrics. "
        "**Inertia** (lower is better) measures how tight the clusters are; "
        "the elbow is where adding more clusters stops helping much. "
        "**Silhouette** (higher is better) measures how well-separated the clusters are. "
        "The best k is where the silhouette peaks, or where inertia bends sharply."
    )

    with st.spinner("Running K-Means for k=2 to 8..."):
        elbow_df = _elbow_data(df_clustered)

    if elbow_df is None:
        st.info("Not enough feature columns to run elbow analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_in = px.line(elbow_df, x="k", y="inertia", markers=True,
                         title="Inertia vs number of clusters",
                         labels={"k": "Number of clusters (k)", "inertia": "Inertia"})
        fig_in.add_vline(x=4, line_dash="dash", line_color="grey",
                         annotation_text="Current k=4")
        st.plotly_chart(fig_in, width="stretch")
    with col2:
        fig_sil = px.line(elbow_df, x="k", y="silhouette", markers=True,
                          title="Silhouette score vs number of clusters",
                          labels={"k": "Number of clusters (k)", "silhouette": "Silhouette score"})
        fig_sil.add_vline(x=4, line_dash="dash", line_color="grey",
                          annotation_text="Current k=4")
        st.plotly_chart(fig_sil, width="stretch")

    best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
    current_sil = elbow_df.loc[elbow_df["k"] == 4, "silhouette"].values[0]
    best_sil = elbow_df["silhouette"].max()
    if best_k != 4:
        st.info(
            f"The silhouette score peaks at **k={best_k}** ({best_sil:.3f}) "
            f"vs k=4 ({current_sil:.3f}). "
            f"If the clusters look too similar, re-running the pipeline with k={best_k} "
            "may give more distinct groups."
        )
    else:
        st.success(f"k=4 has the highest silhouette score ({best_sil:.3f}). The current number of clusters looks optimal.")


# ── feature centroids ──────────────────────────────────────────────────────────

def _render_feature_centroids(df_clustered, c_col, rank_col):
    st.subheader("What distinguishes each cluster?")
    st.markdown(
        "This table shows the average value of key features per cluster. "
        "If the kWh column is similar across clusters, then it is the **time of day or "
        "day of week** that separates them, not the quantity of electricity used."
    )

    display_cols = {c_col: "Avg kWh"}
    for col, label in [("hour", "Avg hour"), ("weekday_num", "Avg weekday (0=Sun)")]:
        if col in df_clustered.columns:
            display_cols[col] = label

    agg = df_clustered.groupby(rank_col)[list(display_cols.keys())].mean().round(3)
    agg.index = [f"Cluster {i} - {CLUSTER_LABELS.get(i, '')}" for i in agg.index]
    agg.columns = [display_cols[c] for c in display_cols]

    # Add within-cluster std for kWh to show spread
    std = df_clustered.groupby(rank_col)[c_col].std().round(3)
    agg["Std kWh (spread)"] = std.values

    st.dataframe(agg, width="stretch")
    st.caption(
        "High std kWh within a cluster = that cluster contains very mixed readings. "
        "If avg kWh values are close together across clusters, the clusters are mainly "
        "separated by time-of-day or day-of-week patterns rather than consumption level."
    )


# ── summary table ──────────────────────────────────────────────────────────────

def _render_cluster_summary(df_clustered, c_col, rank_col, cluster_summary):
    st.subheader("Cluster Summary Statistics")
    if cluster_summary is not None:
        cols = [col for col in cluster_summary.columns if col != "cluster"]
        st.dataframe(cluster_summary[cols].round(3), width="stretch")
    else:
        on_the_fly = (
            df_clustered.groupby(rank_col)[c_col]
            .agg(avg_kWh="mean", median_kWh="median", min_kWh="min",
                 max_kWh="max", std_kWh="std", count="count")
            .reset_index().round(3)
        )
        on_the_fly["label"] = on_the_fly[rank_col].map(CLUSTER_LABELS)
        st.dataframe(on_the_fly, width="stretch")
