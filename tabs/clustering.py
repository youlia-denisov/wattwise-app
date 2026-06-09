import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

CLUSTER_PALETTE = {0: "#2ca25f", 1: "#fee08b", 2: "#fdae61", 3: "#d73027"}
CLUSTER_LABELS  = {0: "Low use", 1: "Medium-low", 2: "Medium-high", 3: "High use"}


def render_clustering(load_clustering_data, WEEKDAY_ORDER):
    st.header("Consumption Clustering")
    st.markdown(
        "This analysis automatically groups all your hourly readings into 4 usage profiles, "
        "without being told what to look for. Each reading gets assigned to the profile it resembles most. "
        "The profiles are then ranked by how much electricity they use: "
        "**0 = your quietest hours** through to **3 = your heaviest-use hours**. "
        "The heatmap below shows which profile dominates at each hour of each day of the week."
    )

    df_clustered, cluster_summary = load_clustering_data()

    if df_clustered is None:
        st.warning(
            "Usage profiling hasn't been run yet. "
            "To unlock this tab, open a terminal in the project folder and run: "
            "`python clustering_with_visuals.py` — then come back and refresh the page."
        )
        with st.expander("Show me the exact command"):
            st.code("python clustering_with_visuals.py", language="bash")
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
    _render_cluster_summary(df_clustered, c_col, rank_col, cluster_summary)


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
    fig_heat.update_layout(title="Dominant Cluster — Weekday x Hour",
                           xaxis_title="Hour", yaxis_title="Weekday", height=380)
    st.plotly_chart(fig_heat, width="stretch")


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
