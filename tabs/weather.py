import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from israel_cities import CITIES, get_city_coords
except ImportError:
    from src.israel_cities import CITIES, get_city_coords

try:
    from weather_analysis import fetch_weather_open_meteo
except ImportError:
    from src.weather_analysis import fetch_weather_open_meteo


@st.cache_data(show_spinner="Fetching weather data…")
def _fetch_weather(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather only — cached by city + date range."""
    coords = get_city_coords(city)
    return fetch_weather_open_meteo(
        start_date,
        end_date,
        latitude=coords["latitude"],
        longitude=coords["longitude"],
        timezone=coords["timezone"],
    )


def render_weather(df_clean: pd.DataFrame):
    st.header("Weather & Electricity")
    st.markdown(
        "Does your electricity use go up when it's hot? Does rain make a difference? "
        "This tab lines up your hourly consumption with weather data to find out."
    )

    if df_clean is None or "datetime" not in df_clean.columns:
        st.warning("No consumption data available. Please upload and process your file first.")
        return

    # City selector, uses list of 26 main Insraeli cities from the israel_cities module. Defaults to Tel Aviv if available.
    city_names = CITIES["city"].tolist()
    default_idx = city_names.index("Tel Aviv") if "Tel Aviv" in city_names else 0

    col_city, col_info = st.columns([2, 3])
    with col_city:
        selected_city = st.selectbox(
            "📍 Choose a city for weather data",
            options=city_names,
            index=default_idx,
            help="Weather is fetched from Open-Meteo for the selected city's coordinates.",
        )
    with col_info:
        coords = get_city_coords(selected_city)
        st.caption(
            f"**{selected_city}** — "
            f"lat {coords['latitude']:.4f}, lon {coords['longitude']:.4f}  \n"
            f"Timezone: {coords['timezone']}"
        )

    # ── Fetch weather, then merge with df_clean ───────────────────────────────
    datetimes = pd.to_datetime(df_clean["datetime"], errors="coerce").dropna()
    start_date = datetimes.min().strftime("%Y-%m-%d")
    end_date   = datetimes.max().strftime("%Y-%m-%d")

    try:
        weather = _fetch_weather(selected_city, start_date, end_date)
    except Exception as exc:
        st.error(f"Could not fetch weather data: {exc}")
        return

    df = df_clean.copy()
    df["weather_hour"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    df_weather = df.merge(weather, on="weather_hour", how="left")

    if df_weather["temperature_c"].isna().all():
        st.warning("Weather data was fetched but could not be merged with your consumption data.")
        return

    _render_correlation_metrics(df_weather)
    st.divider()
    _render_dual_axis_trend(df_weather)
    _render_rolling_averages(df_weather)
    _render_temp_band_chart(df_weather)


def _render_correlation_metrics(df_weather):
    corr_temp = df_weather["kWh"].corr(df_weather["temperature_c"])
    corr_hum  = df_weather["kWh"].corr(df_weather["humidity_pct"])
    corr_prec = df_weather["kWh"].corr(df_weather["precipitation_mm"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Corr: kWh vs Temperature",   f"{corr_temp:.3f}")
    with m2:
        st.metric("Corr: kWh vs Humidity",      f"{corr_hum:.3f}")
    with m3:
        st.metric("Corr: kWh vs Precipitation", f"{corr_prec:.3f}")

    st.caption(
        "**Correlation** measures how closely two things move together, on a scale from -1 to +1. "
        "+1 means they rise and fall perfectly in sync (hotter → always more electricity). "
        "-1 means the opposite. "
        "0 means no relationship. Values around ±0.3 are a moderate link; ±0.6 is strong."
    )


def _render_dual_axis_trend(df_weather):
    st.subheader("Daily Trend: Consumption & Temperature")
    daily_weather = (
        df_weather.groupby("date")
        .agg(daily_kWh=("kWh", "sum"), avg_temp=("temperature_c", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["daily_kWh"],
        name="Daily kWh", line=dict(color="#e55c30"), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["avg_temp"],
        name="Avg Temp (°C)", line=dict(color="#4a90d9", dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        title="Daily Consumption vs Average Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Daily kWh", side="left"),
        yaxis2=dict(title="Avg Temperature (°C)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")


def _render_rolling_averages(df_weather):
    st.subheader("7-Day Rolling Average: Consumption & Temperature")
    st.markdown(
        "Smooths out day-to-day noise to reveal longer-term trends. "
        "A rising rolling temperature line alongside a rising kWh line confirms a seasonal relationship."
    )
    daily = (
        df_weather.groupby("date")
        .agg(daily_kWh=("kWh", "sum"), avg_temp=("temperature_c", "mean"))
        .reset_index()
        .sort_values("date")
    )
    daily["roll_kWh"]  = daily["daily_kWh"].rolling(7, min_periods=1).mean()
    daily["roll_temp"] = daily["avg_temp"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["daily_kWh"],
        name="Daily kWh", line=dict(color="#f4a582", width=1), opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["roll_kWh"],
        name="7-day avg kWh", line=dict(color="#e55c30", width=2.5), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["avg_temp"],
        name="Daily temp (°C)", line=dict(color="#92c5de", width=1), opacity=0.4, yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["roll_temp"],
        name="7-day avg temp (°C)", line=dict(color="#4a90d9", width=2.5, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        title="7-Day Rolling Average — Consumption & Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Daily kWh", side="left"),
        yaxis2=dict(title="Avg Temperature (°C)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")


def _render_temp_band_chart(df_weather):
    st.subheader("Average Consumption by Temperature Band")
    df_weather = df_weather.copy()
    df_weather["temp_band"] = pd.cut(
        df_weather["temperature_c"],
        bins=[-10, 5, 10, 15, 20, 25, 30, 50],
        labels=["<5°C", "5-10°C", "10-15°C", "15-20°C", "20-25°C", "25-30°C", ">30°C"],
    )
    temp_band_agg = (
        df_weather.groupby("temp_band", observed=True)["kWh"]
        .agg(avg_kWh="mean", count="count")
        .reset_index()
    )
    st.plotly_chart(
        px.bar(temp_band_agg, x="temp_band", y="avg_kWh",
               text=temp_band_agg["count"].apply(lambda n: f"n={n}"),
               title="Avg Hourly Consumption by Temperature Band",
               labels={"temp_band": "Temperature Band", "avg_kWh": "Avg kWh"},
               color="avg_kWh", color_continuous_scale="RdYlBu_r"),
        width="stretch",
    )
