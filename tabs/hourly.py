import streamlit as st
import plotly.express as px

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
