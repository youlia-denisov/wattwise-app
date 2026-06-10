import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Injected once per session to make explanatory captions slightly larger
_EXPLANATION_CSS = """
<style>
section[data-testid="stMain"] .stMarkdown p,
section[data-testid="stMain"] .stCaptionContainer p {
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
}
</style>
"""


def render_hourly(df_clean, hourly, consumption_col, WEEKDAY_ORDER):
    st.markdown(_EXPLANATION_CSS, unsafe_allow_html=True)
    st.header("Hourly Consumption Patterns")

    if not {"hour", "weekday"}.issubset(df_clean.columns):
        st.warning("Columns `hour` and `weekday` not found in cleaned data.")
        return

    col1, col2 = st.columns(2)

    pivot_mean = df_clean.pivot_table(
        index="weekday", columns="hour", values=consumption_col, aggfunc="mean"
    ).reindex([d for d in WEEKDAY_ORDER if d in df_clean["weekday"].unique()])

    pivot_std = df_clean.pivot_table(
        index="weekday", columns="hour", values=consumption_col, aggfunc="std"
    ).reindex([d for d in WEEKDAY_ORDER if d in df_clean["weekday"].unique()])

    with col1:
        st.plotly_chart(
            px.imshow(pivot_mean, title="Mean Usage by Weekday & Hour",
                      color_continuous_scale="RdYlGn_r", labels={"color": "Avg kWh"}),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            px.imshow(pivot_std, title="Variability by Weekday & Hour",
                      color_continuous_scale="Blues", labels={"color": "Std kWh"}),
            width="stretch",
        )

    st.caption(
        "**Left chart — Average usage:** darker red = you typically use more electricity at that time. "
        "**Right chart — Variability (standard deviation):** darker blue = your usage at that time varies a lot "
        "from week to week. A pale cell means you're very consistent; a dark cell means some weeks are very "
        "different from others."
    )

    st.divider()
    st.subheader("Consumption Range by Hour of Day")
    st.caption(
        "Average consumption for each hour, with the shaded band showing the 25–75th percentile range "
        "(how much it varies day to day)."
    )

    # Compute per-hour stats across all days
    hourly_stats = (
        df_clean.groupby("hour")[consumption_col]
        .agg(
            mean="mean",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            std="std",
        )
        .reset_index()
    )

    fig_range = go.Figure()

    # Shaded band: 25–75th percentile
    fig_range.add_trace(go.Scatter(
        x=pd.concat([hourly_stats["hour"], hourly_stats["hour"].iloc[::-1]]),
        y=pd.concat([hourly_stats["q75"], hourly_stats["q25"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(99, 183, 205, 0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="25–75th percentile",
    ))

    # Mean line
    fig_range.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["mean"],
        mode="lines+markers",
        line=dict(color="#1DB87E", width=2.5),
        marker=dict(size=6, color="#1DB87E"),
        name="Daily average",
        hovertemplate="Hour %{x}: %{y:.2f} kWh<extra></extra>",
    ))

    fig_range.update_layout(
        template="plotly_white",
        xaxis=dict(title="Hour", dtick=1, range=[-0.5, 23.5]),
        yaxis=dict(title="kWh"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=20, b=40),
    )

    st.plotly_chart(fig_range, width="stretch")

    # ── Insight card ──────────────────────────────────────────────────────────
    peak_row = hourly_stats.loc[hourly_stats["mean"].idxmax()]
    volatile_row = hourly_stats.loc[hourly_stats["std"].idxmax()]

    ic1, ic2 = st.columns(2)
    with ic1:
        st.info(
            f"**⚡ Peak hour: {int(peak_row['hour']):02d}:00**  \n"
            f"Average {peak_row['mean']:.2f} kWh — your highest-demand hour."
        )
    with ic2:
        st.info(
            f"**📊 Most variable hour: {int(volatile_row['hour']):02d}:00**  \n"
            f"Std dev {volatile_row['std']:.2f} kWh — consumption changes the most at this hour."
        )
