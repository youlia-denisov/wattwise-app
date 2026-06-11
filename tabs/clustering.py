import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd

from src.clustering import assign_cluster_ranks

# ── dynamic palette / label helpers ───────────────────────────────────────────

def _make_palette(k: int) -> dict:
    """
    Build a green→red colour map for k cluster ranks.
    Rank 0 (lowest use) = green, rank k-1 (highest use) = red.
    """
    positions = [(k - 1 - i) / max(k - 1, 1) for i in range(k)]
    raw = pc.sample_colorscale("RdYlGn", positions)
    hex_colors = []
    for c in raw:
        if isinstance(c, str):
            hex_colors.append(c)
        else:
            # c is an (r, g, b) tuple with values in [0, 1]
            r, g, b = c
            hex_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return {i: hex_colors[i] for i in range(k)}


def _make_labels(k: int) -> dict:
    """Human-readable label for each rank level, scaled to k groups."""
    if k == 2:
        return {0: "Low use", 1: "High use"}
    if k == 3:
        return {0: "Low use", 1: "Medium use", 2: "High use"}
    if k == 4:
        return {0: "Low use", 1: "Medium-low", 2: "Medium-high", 3: "High use"}
    if k == 5:
        return {0: "Low use", 1: "Medium-low", 2: "Medium", 3: "Medium-high", 4: "High use"}
    # k == 6
    return {0: "Low use", 1: "Below average", 2: "Medium-low",
            3: "Medium-high", 4: "Above average", 5: "High use"}


# ── live re-clustering ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _recluster(df_clustered: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Re-run KMeans with k clusters on the feature columns that the pipeline
    already computed and saved in df_clustered.  No pipeline re-run needed —
    the heavy feature engineering (cyclical encoding, daily aggregates) is
    already done; we just scale and fit again.

    Result is cached: switching back to a previously tried k is instant.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler

    feature_cols = [c for c in ["kWh", "hour_sin", "hour_cos",
                                 "weekday_sin", "weekday_cos",
                                 "month_sin", "month_cos"]
                    if c in df_clustered.columns]

    df = df_clustered.copy()
    X = RobustScaler().fit_transform(df[feature_cols])
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X)
    df = assign_cluster_ranks(df)
    return df


# ── main entry point ───────────────────────────────────────────────────────────

def render_clustering(load_clustering_data, WEEKDAY_ORDER):
    st.header("Consumption Clustering")
    st.markdown(
        "This analysis automatically groups all your hourly readings into usage profiles, "
        "without being told what to look for. Each reading gets assigned to the profile it "
        "resembles most, then profiles are ranked by consumption level: "
        "**rank 0 = your quietest hours**, highest rank = your heaviest-use hours."
    )

    df_base, cluster_summary = load_clustering_data()

    if df_base is None:
        st.warning(
            "Usage profiling has not been run yet. "
            "Upload your CSV and click Run Full Pipeline to unlock this tab."
        )
        return

    # ── k selector ────────────────────────────────────────────────────────────
    pipeline_k = int(df_base["cluster_rank"].nunique())
    k = _render_k_selector(df_base, pipeline_k)

    # Re-cluster if k differs from what the pipeline saved, otherwise use
    # the pipeline result as-is to guarantee consistency with other tabs.
    if k == pipeline_k:
        df_clustered = df_base.copy()
    else:
        with st.spinner(f"Grouping your readings into {k} clusters…"):
            df_clustered = _recluster(df_base, k)

    palette = _make_palette(k)
    labels  = _make_labels(k)

    c_col = next(
        (col for col in df_clustered.columns
         if any(w in col.lower() for w in ["kwh", "kwatt", "consumption"])),
        df_clustered.columns[0],
    )
    rank_col = "cluster_rank" if "cluster_rank" in df_clustered.columns else "cluster"
    df_clustered["_label"] = df_clustered[rank_col].map(labels)

    _render_cluster_charts(df_clustered, c_col, rank_col, k, palette, labels)
    _render_cluster_heatmap(df_clustered, rank_col, WEEKDAY_ORDER, k, palette, labels)
    st.divider()
    _render_silhouette(df_clustered, rank_col, k, palette, labels)
    st.divider()
    _render_elbow(df_base, k)
    st.divider()
    _render_feature_centroids(df_clustered, c_col, rank_col, labels)


# ── k selector UI ──────────────────────────────────────────────────────────────

def _render_k_selector(df_base: pd.DataFrame, pipeline_k: int) -> int:
    """
    Render the k selector with plain-language explanation.
    Returns the chosen k.
    """
    with st.expander("⚙️ Choose number of groups (k) — click to expand", expanded=False):
        st.markdown(
            """
**What is k?**

K-Means works by sorting every hourly reading into *k* groups (clusters), where readings
that look similar end up in the same group. The groups are found automatically — you only
decide *how many* you want.

| k | What you get |
|---|---|
| 2 | A simple Low / High split. Useful for spotting whether you have distinct "off-peak" and "on-peak" periods. |
| 3 | Low / Medium / High. Adds a middle band — good when k=2 feels too coarse. |
| **4** | **Default.** Low / Med-low / Med-high / High. Balances detail and interpretability for most households. |
| 5 | Adds an extra level of detail between the middle bands. |
| 6 | Maximum detail — useful only if k=4/5 still looks too merged in the silhouette analysis. |

**How to choose?**
Use the **Elbow Curve** section below as your guide:
- Look for the k where the **silhouette score peaks** — that is where the groups are most distinct.
- If the silhouette chart suggests a different k, try it here and see if the clusters look more meaningful.
- If the current silhouette score is already above **0.5**, the default k is doing well and changing it may not help much.

Switching k here **does not re-run the full pipeline** — it re-groups your readings
instantly in the browser, using the features already computed. Other tabs are unaffected.
            """
        )

        # Quick recommendation from elbow data (already cached)
        elbow_df = _elbow_data(df_base)
        if elbow_df is not None:
            best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
            best_sil = float(elbow_df["silhouette"].max())
            pipeline_sil_row = elbow_df.loc[elbow_df["k"] == pipeline_k, "silhouette"]
            pipeline_sil_val = float(pipeline_sil_row.values[0]) if not pipeline_sil_row.empty else None

            if best_k != pipeline_k:
                msg = (
                    f"💡 The elbow analysis suggests **k={best_k}** gives the best-separated "
                    f"clusters (silhouette {best_sil:.3f}"
                )
                if pipeline_sil_val is not None:
                    msg += f" vs {pipeline_sil_val:.3f} for k={pipeline_k}"
                msg += "). You can try it with the slider below."
                st.info(msg)
            else:
                st.success(
                    f"✅ The elbow analysis agrees: **k={pipeline_k}** is the best choice "
                    f"(silhouette {best_sil:.3f})."
                )

        k = st.slider(
            "Number of clusters (k)",
            min_value=2, max_value=6, value=pipeline_k, step=1,
            help=(
                "Move the slider to re-group your readings. "
                "The charts update instantly — no pipeline re-run needed."
            ),
        )

        if k != pipeline_k:
            st.caption(
                f"ℹ️ You are viewing a custom grouping with k={k}. "
                f"The pipeline was run with k={pipeline_k}. "
                "Other tabs (Overview, Discounts, etc.) still use the pipeline result."
            )

    return k


# ── cluster charts ────────────────────────────────────────────────────────────

def _render_cluster_charts(df_clustered, c_col, rank_col, k, palette, labels):
    col1, col2 = st.columns(2)
    rank_order = list(range(k))
    axis_label = f"Cluster rank (0 = lowest, {k-1} = highest)"

    with col1:
        fig_box = px.box(
            df_clustered, x=rank_col, y=c_col, color=rank_col,
            color_discrete_map=palette,
            title="Consumption Distribution by Cluster",
            labels={rank_col: axis_label, c_col: "kWh"},
            category_orders={rank_col: rank_order},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

    with col2:
        size_df = (
            df_clustered[rank_col].value_counts()
            .reindex(rank_order, fill_value=0)
            .reset_index()
        )
        size_df.columns = ["cluster_rank", "count"]
        size_df["label"] = size_df["cluster_rank"].map(labels)
        fig_size = px.bar(
            size_df, x="cluster_rank", y="count", color="cluster_rank",
            color_discrete_map=palette,
            title="Records per Cluster",
            labels={"cluster_rank": axis_label},
            text="count",
        )
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, width="stretch")


def _render_cluster_heatmap(df_clustered, rank_col, WEEKDAY_ORDER, k, palette, labels):
    if not {"weekday", "hour"}.issubset(df_clustered.columns):
        return
    st.subheader("Dominant Cluster by Weekday & Hour")
    dominant = (
        df_clustered.groupby(["weekday", "hour"])[rank_col]
        .agg(lambda x: x.mode().iloc[0])
        .unstack()
    )
    dominant = dominant.reindex([d for d in WEEKDAY_ORDER if d in dominant.index])

    # Build a step colorscale from the dynamic palette
    colorscale = []
    for i in range(k):
        pos_lo = i / k
        pos_hi = (i + 1) / k
        colorscale.append([pos_lo, palette[i]])
        colorscale.append([pos_hi, palette[i]])

    tick_labels = [f"{i} — {labels.get(i, f'Level {i}')}" for i in range(k)]

    fig_heat = go.Figure(go.Heatmap(
        z=dominant.values, x=dominant.columns.tolist(), y=dominant.index.tolist(),
        colorscale=colorscale, zmin=0, zmax=k - 1,
        colorbar=dict(
            title="Cluster rank",
            tickvals=list(range(k)),
            ticktext=tick_labels,
        ),
    ))
    fig_heat.update_layout(
        title="Dominant Cluster — Weekday × Hour",
        xaxis_title="Hour", yaxis_title="Weekday",
        font=dict(size=14),
        height=400,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_heat, width="stretch")


# ── silhouette analysis ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
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
    labels_arr = sub[rank_col].values

    overall = float(silhouette_score(X, labels_arr))
    samples = silhouette_samples(X, labels_arr)
    return overall, samples, labels_arr


def _render_silhouette(df_clustered, rank_col, k, palette, labels):
    st.subheader("Silhouette Analysis")
    st.markdown(
        "The silhouette score measures how well each reading fits its assigned cluster. "
        "It ranges from **−1 to +1**: scores near +1 mean the reading is clearly in the "
        "right cluster; near 0 means it sits on the boundary between two clusters; "
        "negative means it might fit a neighbouring cluster better."
    )

    with st.spinner("Computing silhouette scores…"):
        overall, samples, labels_arr = _silhouette_scores(df_clustered, rank_col)

    if overall is None:
        st.info("Not enough feature columns to compute silhouette scores.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Overall silhouette score", f"{overall:.3f}")
    with col2:
        if overall > 0.5:
            st.success(
                f"**Strong separation** — the {k} clusters are well-defined and distinct."
            )
        elif overall > 0.25:
            st.warning(
                f"**Moderate separation** — clusters have some overlap. "
                f"The algorithm found structure in {k} groups, but the boundaries are not "
                "crisp. Try a lower k in the ⚙️ selector above, or check the Elbow Curve below."
            )
        else:
            st.error(
                f"**Weak separation** — clusters overlap significantly with k={k}. "
                "The data may not have this many natural groups. "
                "Try reducing k — the Elbow Curve below will suggest a better number."
            )

    per_cluster = (
        pd.DataFrame({"score": samples, "cluster": labels_arr})
        .groupby("cluster")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_silhouette"})
    )
    per_cluster["label"] = per_cluster["cluster"].map(labels)

    fig_bar = px.bar(
        per_cluster, x="cluster", y="avg_silhouette",
        color="cluster", color_discrete_map=palette,
        text=per_cluster["avg_silhouette"].map(lambda v: f"{v:.3f}"),
        labels={"cluster": "Cluster rank", "avg_silhouette": "Avg silhouette score"},
        title="Average silhouette score per cluster",
        category_orders={"cluster": list(range(k))},
    )
    fig_bar.add_hline(y=overall, line_dash="dash", line_color="grey",
                      annotation_text=f"Overall avg ({overall:.3f})")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, yaxis_range=[-0.1, 1.0])
    st.plotly_chart(fig_bar, width="stretch")
    st.caption(
        "Clusters with a score well below the overall average are the weak ones — "
        "readings in those clusters could plausibly belong to a neighbouring group. "
        "If several clusters are below 0.2, try reducing k in the ⚙️ selector above."
    )

    with st.expander("Show full silhouette plot (individual readings)"):
        df_sil = pd.DataFrame({"score": samples, "cluster": labels_arr})
        df_sil = df_sil.sort_values(["cluster", "score"]).reset_index(drop=True)
        df_sil["y"] = list(range(len(df_sil)))
        df_sil["label"] = df_sil["cluster"].map(labels)

        fig_sil = px.bar(
            df_sil, x="score", y="y", color="cluster",
            color_discrete_map=palette,
            orientation="h",
            labels={"score": "Silhouette score", "y": "", "cluster": "Cluster rank"},
            title="Silhouette plot — each bar is one reading",
            height=max(400, min(800, len(df_sil) // 5)),
            category_orders={"cluster": list(range(k))},
        )
        fig_sil.update_traces(marker_line_width=0)
        fig_sil.update_layout(showlegend=True, yaxis_visible=False,
                               yaxis_showticklabels=False)
        fig_sil.add_vline(x=overall, line_dash="dash", line_color="grey",
                          annotation_text="Overall avg")
        st.plotly_chart(fig_sil, width="stretch")
        st.caption(
            "Each horizontal bar is one reading. Wide bars to the right = confidently "
            "in the right cluster. Bars to the left of 0 = possible misassignment."
        )


# ── elbow curve ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
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
    for k_val in range(2, 9):
        km = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
        lbl = km.fit_predict(X)
        sil = float(silhouette_score(X, lbl))
        rows.append({"k": k_val, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def _render_elbow(df_base, selected_k: int):
    st.subheader("Elbow Curve — How many clusters is best?")
    st.markdown(
        "This chart runs K-Means for every k from 2 to 8 and shows two quality measures. "
        "Use them together to pick the right k:\n\n"
        "- **Inertia** (left chart, lower is better): measures how tight the clusters are. "
        "Look for the *elbow* — the point where the line bends and flattens. "
        "Adding more clusters beyond the elbow gives diminishing returns.\n"
        "- **Silhouette score** (right chart, higher is better): measures how well-separated "
        "the clusters are. The k with the **highest peak** is your best option.\n\n"
        "The dashed line marks your **currently selected k**."
    )

    with st.spinner("Running K-Means for k=2 to 8..."):
        elbow_df = _elbow_data(df_base)

    if elbow_df is None:
        st.info("Not enough feature columns to run elbow analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_in = px.line(
            elbow_df, x="k", y="inertia", markers=True,
            title="Inertia vs number of clusters",
            labels={"k": "Number of clusters (k)", "inertia": "Inertia"},
        )
        fig_in.add_vline(x=selected_k, line_dash="dash", line_color="#636efa",
                         annotation_text=f"Current k={selected_k}")
        st.plotly_chart(fig_in, width="stretch")

    with col2:
        fig_sil = px.line(
            elbow_df, x="k", y="silhouette", markers=True,
            title="Silhouette score vs number of clusters",
            labels={"k": "Number of clusters (k)", "silhouette": "Silhouette score"},
        )
        fig_sil.add_vline(x=selected_k, line_dash="dash", line_color="#636efa",
                          annotation_text=f"Current k={selected_k}")
        st.plotly_chart(fig_sil, width="stretch")

    best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
    best_sil = float(elbow_df["silhouette"].max())
    current_sil_row = elbow_df.loc[elbow_df["k"] == selected_k, "silhouette"]
    current_sil = float(current_sil_row.values[0]) if not current_sil_row.empty else None

    if best_k != selected_k:
        msg = f"The silhouette score peaks at **k={best_k}** ({best_sil:.3f})"
        if current_sil is not None:
            msg += f" compared to {current_sil:.3f} for your current k={selected_k}."
        msg += " Use the selector at the top to try it."
        st.info(msg)
    else:
        st.success(
            f"k={selected_k} has the highest silhouette score ({best_sil:.3f}) -- "
            "the current number of clusters looks optimal."
        )


# ── feature centroids ──────────────────────────────────────────────────────────

def _render_feature_centroids(df_clustered, c_col, rank_col, labels):
    st.subheader("What distinguishes each cluster?")
    st.markdown(
        "This table shows the average value of key features per cluster. "
        "If the **kWh column is similar across clusters**, it means the groups are separated "
        "mainly by *when* usage happens (hour of day, day of week), not by *how much*."
    )

    display_cols = {c_col: "Avg kWh"}
    for col, label in [("hour", "Avg hour"), ("weekday_num", "Avg weekday (0=Sun)")]:
        if col in df_clustered.columns:
            display_cols[col] = label

    agg = df_clustered.groupby(rank_col)[list(display_cols.keys())].mean().round(3)
    agg.index = [
        f"Cluster {i} -- {labels.get(i, f'Level {i}')}" for i in agg.index
    ]
    agg.columns = [display_cols[c] for c in display_cols]

    std = df_clustered.groupby(rank_col)[c_col].std().round(3)
    agg["Std kWh (spread)"] = std.values

    st.dataframe(agg, width="stretch")
    st.caption(
        "**High std kWh** within a cluster = that cluster contains mixed readings. "
        "If average kWh values are close together across clusters, the clusters are mainly "
        "separated by time-of-day or day-of-week patterns rather than consumption level."
    )


# ── summary table ──────────────────────────────────────────────────────────────

def _render_cluster_summary(df_clustered, c_col, rank_col, labels, cluster_summary):
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
        on_the_fly["label"] = on_the_fly[rank_col].map(labels)
        st.dataframe(on_the_fly, width="stretch")
