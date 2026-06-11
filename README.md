# WattWise — Multi-User Streamlit Dashboard

A multi-user Streamlit app for analyzing household electricity consumption in Israel. Each visitor uploads their own IEC consumption CSV and gets an isolated, personalized analysis session — no data leaks between users.

This is a full refactor of the [Electricity Consumption Analyser](../electricity-consumption-analyser) project, extended with new analytical tabs and a proper multi-user architecture.

Input: raw IEC electricity consumption file (.csv) uploaded via the sidebar.

---

## Quick Start

```bash
cd streamlit-multiuser-app

# Create and activate virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -e .

# Run the dashboard
streamlit run streamlit_electricity_usage.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. Upload your IEC consumption CSV via the sidebar file uploader.
2. Optionally trigger a discount offers refresh from the sidebar — the app scrapes current offers automatically via `scraping.py`.
3. Explore your data across the tabs.

No files need to be placed in `data/raw/` — everything is loaded per session from the sidebar upload.

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Overview | Total consumption, cost estimates, key stats |
| Hourly Patterns | Heatmaps and averages by hour / day of week |
| Trends & Outliers | Long-term trends, anomaly detection |
| Behavioural Fingerprint | WFH ratio, peak hour, night owl score, regularity index |
| Outlier Detection | Compare 3-sigma, IQR, and auto-selected methods |
| Usage Clustering | K-means profiles from quietest to heaviest use |
| Discounts | Rank discount plans by your actual usage patterns |
| Calculator | Estimate costs under different tariffs |
| Weather | Correlate consumption with temperature (Open-Meteo API) |
| Report | Auto-generated Markdown summary |
| About | Project documentation |

---

## Module Structure

```
streamlit-multiuser-app/
├── streamlit_electricity_usage.py   # Entry point — wires sidebar + tabs
├── app_sidebar.py                   # Sidebar UI and file upload
├── app_loaders.py                   # @st.cache_data loaders and column helpers
├── app_offers.py                    # Weekly-refresh logic for discount offers
├── config.py                        # Paths, constants, tariff settings
├── tabs/                            # One file per dashboard tab
│   ├── overview.py
│   ├── hourly.py
│   ├── trends.py
│   ├── behavior_profile.py
│   ├── clustering.py
│   ├── outlier_methods.py
│   ├── discounts.py
│   ├── calculator.py
│   ├── weather.py
│   ├── report.py
│   └── about.py
├── src/                             # Shared data processing modules
├── data/                            # Optional local fallback data
├── docs/                            # Prompts and reference documents
├── pipeline.py                      # Full analysis pipeline (optional, for batch use)
└── pyproject.toml
```

---

## Configuration

Edit `config.py` to adjust:

- `HAS_SMART_METER` — `True`, `False`, or `None` (affects discount plan eligibility)
- `TARIFF` — base electricity rate in NIS/kWh (default: `0.666`, 2026)

---

## Author

**Youlia Denisov** — Data Analyst / Data Scientist / Biologist  
GitHub: https://github.com/YouliaXX | LinkedIn: https://linkedin.com/in/youliadenisov-phd

## License

MIT License — © 2026 Youlia Denisov
