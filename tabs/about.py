import pandas as pd
import streamlit as st


def render_about(df_clean, daily_totals, hourly):
    st.header("About This Dashboard")
    st.markdown(
        "This dashboard was built as a portfolio project by a biologist transitioning to data science. "
        "It demonstrates an end-to-end analysis pipeline: raw smart-meter data → cleaning → "
        "statistical analysis → interactive visualisation."
    )

    st.subheader("Project Stack")
    st.markdown(
        "- **Data processing:** Python, pandas, NumPy  \n"
        "- **Visualisation:** Plotly, Streamlit  \n"
        "- **Machine learning:** scikit-learn (K-Means clustering)  \n"
        "- **Weather data:** Open-Meteo API  \n"
        "- **Reporting:** Markdown + `markdown` library  \n"
    )

    st.subheader("Loaded Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        if df_clean is not None:
            st.metric("Raw readings", f"{len(df_clean):,}")
    with col2:
        if daily_totals is not None:
            st.metric("Days of data", len(daily_totals))
    with col3:
        if hourly is not None:
            st.metric("Hourly stat rows", f"{len(hourly):,}")

    if df_clean is not None and "date" in df_clean.columns:
        dates = pd.to_datetime(df_clean["date"], errors="coerce").dropna()
        st.caption(
            f"Dataset spans **{dates.min().date()}** → **{dates.max().date()}** "
            f"({dates.dt.date.nunique()} unique days)."
        )

    st.divider()
    st.markdown(
        "Source code and methodology notes are available in the project README. "
        "Feel free to open an issue or pull request on GitHub."
    )
