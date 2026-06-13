"""
This module provides functions to fetch and analyze weather data in relation to electricity consumption. It includes:
- `fetch_weather_open_meteo`: Fetches historical weather data from the Open-Meteo API for the specified date range.
- `add_weather`: Merges the fetched (temperature, humidity, precipitation) data with the main consumption DataFrame based on hourly timestamps.
- `summarize_weather`: Computes statistics and correlations between weather variables and electricity consumption.
- `save_weather_plots`: Generates and saves interactive plots to visualize the relationship between weather and consumption.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import requests
try:
    from config import OPEN_METEO_URL, HOST_LATITUDE, HOST_LONGITUDE, LOCAL_TIMEZONE
except ImportError:
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    HOST_LATITUDE = 31.5
    HOST_LONGITUDE = 34.8
    LOCAL_TIMEZONE = "Asia/Jerusalem"


def fetch_weather_open_meteo(
    start_date,
    end_date,
    latitude: float,
    longitude: float,
    timezone: str = LOCAL_TIMEZONE,
) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo.

    latitude/longitude/timezone override config defaults when provided.
    """
    params = {
        "latitude": latitude if latitude is not None else HOST_LATITUDE,
        "longitude": longitude if longitude is not None else HOST_LONGITUDE,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": timezone if timezone is not None else LOCAL_TIMEZONE,
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    weather = pd.DataFrame({
        "weather_hour": pd.to_datetime(hourly.get("time", [])),
        "temperature_c": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
        "humidity_pct": pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
        "precipitation_mm": pd.to_numeric(hourly.get("precipitation", []), errors="coerce"),
    })
    if weather.empty:
        raise ValueError("Open-Meteo returned no hourly data.")
    weather.loc[~weather["temperature_c"].between(-10, 55), "temperature_c"] = pd.NA
    weather.loc[~weather["humidity_pct"].between(0, 100), "humidity_pct"] = pd.NA
    return weather


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["weather_hour"] = pd.to_datetime(data["datetime"]).dt.floor("h")
    weather = fetch_weather_open_meteo(data["datetime"].min(), data["datetime"].max())
    return data.merge(weather, on="weather_hour", how="left")


def summarize_weather(df_weather: pd.DataFrame) -> dict:
    hot = df_weather["temperature_c"] >= 30
    normal = df_weather["temperature_c"] < 30
    return {
        "corr_kWh_temp": float(df_weather["kWh"].corr(df_weather["temperature_c"])),
        "corr_kWh_humidity": float(df_weather["kWh"].corr(df_weather["humidity_pct"])),
        "avg_temp": float(df_weather["temperature_c"].mean()),
        "avg_kWh_hot_hours": float(df_weather.loc[hot, "kWh"].mean()),
        "avg_kWh_normal_hours": float(df_weather.loc[normal, "kWh"].mean()),
    }


def save_weather_plots(df_weather: pd.DataFrame, html_dir: Path) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    fig = px.scatter(df_weather, x="temperature_c", y="kWh", color="weekday", hover_data=["datetime"],
                     title="Consumption vs Local Temperature", template="plotly_white")
    fig.write_html(html_dir / "weather_temperature_scatter.html")
    daily = df_weather.groupby("date", as_index=False).agg(total_kWh=("kWh", "sum"), avg_temp=("temperature_c", "mean"))
    fig = px.line(daily, x="date", y=["total_kWh", "avg_temp"], title="Daily Consumption vs Average Temperature", template="plotly_white")
    fig.write_html(html_dir / "daily_consumption_vs_temperature.html")
