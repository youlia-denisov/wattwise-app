# Data Science Portfolio

**Youlia Denisov · Biologist Turned Data Explorer**

---

## About Me

Hi! I'm Youlia, a biologist with a PhD who's made the exciting jump into data science. After years spent exploring the beautiful complexity of living systems, I discovered I get the exact same spark of excitement from diving into real-world data and turning it into clear, actionable insights. What truly lights me up is using data to solve practical, everyday problems that actually make life better for people.

**Skills:**
Python (pandas, scikit-learn, NumPy) · SQL · Power BI · Plotly · Streamlit · EDA · Statistical Analysis · Machine Learning · Automation · Git

- **GitHub**: [YouliaXX](https://github.com/YouliaXX/DS-portfolio)
- **LinkedIn**: [youliadenisov-phd](https://linkedin.com/in/youliadenisov-phd)

---

## Projects

### ⚡ [Electricity Consumption Analyser](./electricity-consumption-analyser)

This is my first public personal project — and I'm pretty proud of it. I built a complete Python pipeline that takes raw electricity meter data from the Israel Electric Corporation (those messy CSVs with Hebrew headers and weird formatting) and turns it into clear, useful insights. It identifies consumption patterns, spots unusual spikes, correlates usage with weather, and tells you which discount plan would actually save you money based on your real usage.

What I'm especially happy with:
- Smart handling of IEC's quirky export format
- KMeans clustering with proper cyclical encoding for time features
- Robust outlier detection — both statistical and visual
- Optional weather integration via Open-Meteo API
- Interactive Plotly visualizations and a clean Streamlit dashboard
- Real savings analysis for a consumer navigating Israel's liberalized electricity market

**Tech stack:** Python · pandas · scikit-learn · Plotly · Streamlit · matplotlib · seaborn

---

### 🌐 [WattWise — Multi-User Streamlit App](./streamlit-multiuser-app)

A full refactor of the electricity analyser into a proper multi-user web app. The original project was single-user by design — this version gives each visitor their own isolated session, so multiple people can upload and analyze their data simultaneously without anything leaking between them.

Beyond the multi-user architecture, I added several new analytical tabs that weren't in the original:
- **Behavioural Fingerprint** — distils your entire consumption history into a handful of interpretable metrics (WFH ratio, peak hour, night owl score, regularity index)
- **Outlier Detection** — compares simple-to-understand methods: 3-sigma, IQR, with an auto-selected recommendation based on your data's distribution
- **Usage Clustering** — groups hourly readings into ranked profiles from quietest to heaviest use

**Tech stack:** Python · Streamlit · scikit-learn · Plotly · pandas

---

*More projects coming soon.*
