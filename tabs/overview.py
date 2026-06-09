import streamlit as st
import pandas as pd
import plotly.express as px


def render_overview(daily_totals, date_col, daily_value_col, df_clean,
                    consumption_col, safe_mean, safe_max):
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_kwh = pd.to_numeric(daily_totals[daily_value_col], errors="coerce").sum()
        st.metric("Total Consumption", f"{total_kwh:,.1f} kWh")
    with col2:
        st.metric("Days Analyzed", len(daily_totals))
    with col3:
        st.metric("Avg Daily", f"{safe_mean(daily_totals[daily_value_col]):.1f} kWh")
    with col4:
        st.metric("Peak Day", f"{safe_max(daily_totals[daily_value_col]):.1f} kWh")

    st.plotly_chart(
        px.line(daily_totals, x=date_col, y=daily_value_col, title="Daily Consumption Trend"),
        width="stretch",
    )
