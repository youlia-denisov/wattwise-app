import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HTML_DIR = OUTPUT_DIR / "html"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "reports"

CONSUMPTION_FILE = RAW_DIR / "Electricity_consumption.csv"
DISCOUNT_OFFERS_FILE = EXTERNAL_DIR / "electricity_discount_offers.csv"

WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Clustering
N_CLUSTERS = 4
RANDOM_STATE = 42

# Time-of-day windows (24-hour clock, inclusive boundaries)
# Used in feature engineering, clustering, and discount calculations.
# Change here and every module that imports these will update automatically.
DAY_START_HOUR     = 7   # 07:00 — start of daytime window
DAY_END_HOUR       = 16  # 16:00 — end of daytime window
EVENING_START_HOUR = 17  # 17:00 — start of evening / after-work peak
EVENING_END_HOUR   = 22  # 22:00 — end of evening window
NIGHT_START_HOUR   = 23  # 23:00 — start of overnight baseline (wraps to 06:00)

# Location and API for weather data (if used)
# Set to your location's latitude and longitude for accurate weather data retrieval, which can be used in the API cross-analysis step.

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Location for weather data. Override via environment variables for deployment;
# defaults are set to Israel (original author's location) as a fallback.
HOST_LATITUDE  = float(os.environ.get("HOST_LATITUDE",  31.5))
HOST_LONGITUDE = float(os.environ.get("HOST_LONGITUDE", 34.8))
LOCAL_TIMEZONE = os.environ.get("LOCAL_TIMEZONE", "Asia/Jerusalem")

# Set to True/False if you know. Leave None to keep all offers and mark eligibility as unknown.
HAS_SMART_METER = None

# URL list of the electricity discount offers page to scrape. This is used in the API cross-analysis step.
URLS = [
    "https://www.kamaze.co.il/Companies/82227/Cellcom/electrical-power",
    "https://www.kamaze.co.il/Companies/82228/Hot/electrical-power",
    "https://www.kamaze.co.il/Companies/82260/Bezeq/electrical-power",
    "https://www.kamaze.co.il/Companies/82287/Partner/electrical-power",
    "https://www.kamaze.co.il/Companies/82471/supergas-electric/electrical-power",
    "https://www.kamaze.co.il/Companies/82501/amisragas--electric/electrical-power",
    "https://www.kamaze.co.il/Companies/82476/pazgas-electric/electrical-power",
    # Ramy-Levi removed — page returns no usable data (JS-rendered)
    # israelelectricity.com removed — 404 Not Found
]

# Local ElectricityTariff for cost calculations. 
# in NIS per kWh (2026), as a base rate for calculations. Adjust if needed based on your data or assumptions.

TARIFF = 0.666  