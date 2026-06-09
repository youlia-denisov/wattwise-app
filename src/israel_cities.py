"""
Israeli cities with their geographic coordinates and timezone.
Used for weather data fetching in the Weather tab.
"""
import pandas as pd

ISRAEL_TIMEZONE = "Asia/Jerusalem"

CITIES = pd.DataFrame([
    {"city": "Tel Aviv",        "latitude": 32.0853, "longitude": 34.7818},
    {"city": "Jerusalem",       "latitude": 31.7683, "longitude": 35.2137},
    {"city": "Haifa",           "latitude": 32.7940, "longitude": 34.9896},
    {"city": "Beer Sheva",      "latitude": 31.2518, "longitude": 34.7915},
    {"city": "Eilat",           "latitude": 29.5577, "longitude": 34.9519},
    {"city": "Netanya",         "latitude": 32.3215, "longitude": 34.8532},
    {"city": "Ashdod",          "latitude": 31.8040, "longitude": 34.6553},
    {"city": "Rishon LeZion",   "latitude": 31.9730, "longitude": 34.7925},
    {"city": "Petah Tikva",     "latitude": 32.0841, "longitude": 34.8878},
    {"city": "Holon",           "latitude": 32.0107, "longitude": 34.7792},
    {"city": "Ramat Gan",       "latitude": 32.0684, "longitude": 34.8248},
    {"city": "Rehovot",         "latitude": 31.8969, "longitude": 34.8186},
    {"city": "Bat Yam",         "latitude": 32.0231, "longitude": 34.7503},
    {"city": "Herzliya",        "latitude": 32.1663, "longitude": 34.8441},
    {"city": "Kfar Saba",       "latitude": 32.1790, "longitude": 34.9079},
    {"city": "Ra'anana",        "latitude": 32.1868, "longitude": 34.8709},
    {"city": "Modiin",          "latitude": 31.8980, "longitude": 35.0108},
    {"city": "Nahariya",        "latitude": 33.0078, "longitude": 35.0972},
    {"city": "Tiberias",        "latitude": 32.7922, "longitude": 35.5312},
    {"city": "Ashkelon",        "latitude": 31.6688, "longitude": 34.5743},
    {"city": "Afula",           "latitude": 32.6078, "longitude": 35.2897},
    {"city": "Nazareth",        "latitude": 32.6996, "longitude": 35.3035},
    {"city": "Safed",           "latitude": 32.9646, "longitude": 35.4960},
    {"city": "Dimona",          "latitude": 31.0677, "longitude": 35.0326},
    {"city": "Arad",            "latitude": 31.2587, "longitude": 35.2127},
    {"city": "Mitzpe Ramon",    "latitude": 30.6100, "longitude": 34.8017},
])

# Add timezone column (all Israel cities share the same timezone)
CITIES["timezone"] = ISRAEL_TIMEZONE


def get_city_coords(city_name: str) -> dict:
    """Return {'latitude': ..., 'longitude': ..., 'timezone': ...} for a city name."""
    row = CITIES[CITIES["city"] == city_name]
    if row.empty:
        raise ValueError(f"City '{city_name}' not found in israel_cities.")
    return row.iloc[0][["latitude", "longitude", "timezone"]].to_dict()
