"""
Automated scraper for Israeli electricity discount offers.

Replicates the manual AI-prompt process (kamaze_scraping_ai_prompt.md):
  - Fetches each supplier page from config.URLS using requests
  - Parses Hebrew HTML with BeautifulSoup
  - Extracts plan cards with discount percentages
  - Returns a clean DataFrame with the same columns as electricity_discount_offers.csv

Called automatically by get_offers_df() in streamlit_electricity_usage.py
when the existing CSV is older than 7 days.

Requirements: pip install requests beautifulsoup4
"""

import re
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

# IMPORT URLS FROM CONFIG 
# URLS is the list of Israeli electricity supplier pages defined in config.py.
# We import here so the URL list is maintained in one place only.
try:
    from config import URLS
except ImportError:
    # Fallback in case this module is run standalone without config.py on the path.
    URLS = [
        "https://www.kamaze.co.il/Companies/82227/Cellcom/electrical-power",
        "https://www.kamaze.co.il/Companies/82228/Hot/electrical-power",
        "https://www.kamaze.co.il/Companies/82260/Bezeq/electrical-power",
        "https://www.kamaze.co.il/Companies/82287/Partner/electrical-power",
        "https://www.kamaze.co.il/Companies/82471/supergas-electric/electrical-power",
        "https://www.kamaze.co.il/Companies/82501/amisragas--electric/electrical-power",
        "https://www.kamaze.co.il/Companies/82476/pazgas-electric/electrical-power",
    ]

# ── HEADERS ───────────────────────────────────────────────────────────────────
# Hebrew Accept-Language tells the server to respond in Hebrew.
# A realistic User-Agent reduces the chance of being blocked as a bot.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
# Hebrew discount format: "%10 הנחה" or "10% הנחה" or "%10 - %15 הנחה"
# We capture one or two numbers so we can create one row per discount tier.
DISCOUNT_PATTERN = re.compile(
    r"%?(\d{1,2})(?:\s*[-–]\s*%?(\d{1,2}))?\s*הנחה"
)
# Cashback in Hebrew is "צבירה" (accumulation to a wallet).
CASHBACK_PATTERN = re.compile(r"צבירה")
# Smart meter in Hebrew is "מונה חכם".
SMART_METER_PATTERN = re.compile(r"מונה חכם")
# Time restriction — look for HH:MM-HH:MM patterns in the text.
TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})")

# RTL control characters to strip from Hebrew text.
RTL_CHARS = re.compile(r"[‎‏‪-‮⁦-⁩]")

# ── CUSTOMER-TYPE DETECTION PATTERNS ─────────────────────────────────────────
# Each entry: (customer_type_label, compiled_regex)
# The patterns match Hebrew/English phrases that indicate plan exclusivity.
CUSTOMER_TYPE_PATTERNS = [
    ("Hot subscriber",     re.compile(r"ללקוחות הוט|HOT mobile|HOT/NEXT|ללקוחות Hot", re.IGNORECASE)),
    ("Cellcom subscriber", re.compile(r"בלעדי בסלקום|ללקוחות סלקום", re.IGNORECASE)),
    ("Bezeq subscriber",   re.compile(r"ללקוחות בזק|בלעדי בזק", re.IGNORECASE)),
    ("Partner subscriber", re.compile(r"ללקוחות פרטנר|בלעדי פרטנר", re.IGNORECASE)),
    ("Amisragas customer", re.compile(r"ללקוחות אמישראגז|מנויים על החשמל והגז", re.IGNORECASE)),
    ("Pazgas customer",    re.compile(r"ללקוחות פזגז|בלעדי פזגז", re.IGNORECASE)),
]


def _detect_customer_type(plan_name: str, context: str) -> str:
    """
    Return the customer type required to access a plan.
    Returns 'All' if the plan is open to any customer.
    Checks plan name first, then context text.
    """
    combined = f"{plan_name} {context}"
    for label, pattern in CUSTOMER_TYPE_PATTERNS:
        if pattern.search(combined):
            return label
    return "All"


def _clean(text: str) -> str:
    """Remove RTL marks and normalize whitespace."""
    text = RTL_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_supplier_name(url: str) -> str:
    """
    Pull a readable supplier name from the URL path.
    e.g. '.../Companies/82227/Cellcom/electrical-power' → 'Cellcom'
    """
    parts = url.rstrip("/").split("/")
    # For kamaze URLs the supplier name is the 4th path segment (index -2).
    # For other URLs we just use the domain.
    if "kamaze.co.il" in url:
        try:
            return parts[-2].replace("-", " ").title()
        except IndexError:
            pass
    return parts[2].replace("www.", "").split(".")[0].title()


def _parse_time_restriction(text: str) -> str:
    """Extract a time range string like '17:00-23:00' from surrounding text."""
    match = TIME_PATTERN.search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return ""


def _scrape_page(url: str) -> list[dict]:
    """
    Fetch one supplier page and return a list of row dicts.
    Returns an empty list and prints a warning if the page fails or has no data.
    """
    rows = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"FAILED: {url} — {e}")
        return []

    # Some pages are JavaScript-rendered — requests gets an empty shell.
    # We detect this by checking if the body has very little visible text.
    soup = BeautifulSoup(response.content, "html.parser")
    body_text = _clean(soup.get_text())
    if len(body_text) < 200:
        print(f"FAILED:JS_RENDERED — {url} (body too short, page likely needs JS)")
        return []

    supplier_name = _extract_supplier_name(url)
    scraped_at = datetime.datetime.now().isoformat(timespec="seconds")

    # ── PLAN CARDS: preferred extraction target ───────────────────────────────
    # Each plan is an <h3> element (the plan title) near a discount badge.
    # We walk through all h3 tags and look for a discount % in the surrounding block.
    for h3 in soup.find_all("h3"):
        plan_name = _clean(h3.get_text())
        if not plan_name:
            continue

        # The "card" is the nearest ancestor that contains the discount badge.
        # We check the parent, grandparent, and great-grandparent in order.
        card = None
        for ancestor in [h3.parent, h3.parent.parent if h3.parent else None,
                         h3.parent.parent.parent if h3.parent and h3.parent.parent else None]:
            if ancestor and DISCOUNT_PATTERN.search(ancestor.get_text()):
                card = ancestor
                break

        if card is None:
            continue

        card_text = _clean(card.get_text())

        # Find the deal URL: the link on the h3 or the nearest <a>.
        deal_url = ""
        link_tag = h3.find("a") or h3.find_parent("a")
        if link_tag and link_tag.get("href"):
            # BeautifulSoup can return an AttributeValueList for multi-valued attributes.
            # str() converts it to a plain string so startswith() and + work correctly.
            href = str(link_tag["href"])
            deal_url = href if href.startswith("http") else url.split("/Companies")[0] + href

        # Smart meter: does the card mention "מונה חכם"?
        requires_smart_meter = bool(SMART_METER_PATTERN.search(card_text))

        # Cashback vs discount.
        discount_type = "cashback" if CASHBACK_PATTERN.search(card_text) else "discount"

        # Time restriction (e.g. "17:00-23:00 Sun-Thu").
        time_restriction = _parse_time_restriction(card_text)

        # Find all discount percentages in the card.
        # Each match may have one value (single tier) or two (tiered range).
        for match in DISCOUNT_PATTERN.finditer(card_text):
            pct1 = int(match.group(1))
            pct2 = int(match.group(2)) if match.group(2) else None

            # Ignore implausible values (per the prompt: keep only 1-30%).
            for pct in filter(lambda p: p is not None and 1 <= p <= 30, [pct1, pct2]):
                discount_note = f"up to {pct2}%" if pct2 and pct == pct1 else ""

                rows.append({
                    "source_site":         "kamaze.co.il" if "kamaze" in url else url.split("/")[2],
                    "company_page":        url,
                    "supplier_name":       supplier_name,
                    "plan_name":           plan_name,
                    "discount_pct":        pct,
                    "discount_type":       discount_type,
                    "discount_note":       discount_note,
                    "requires_smart_meter": requires_smart_meter,
                    "time_restriction":    time_restriction,
                    "customer_type":       _detect_customer_type(plan_name, card_text),
                    "context":             card_text[:300],
                    "deal_url":            deal_url,
                    "source_url":          url,
                    "scraped_at":          scraped_at,
                })

    return rows


def scrape_offers() -> pd.DataFrame:
    """
    Main entry point called by get_offers_df() in streamlit_electricity_usage.py.

    Scrapes all URLs in config.URLS, combines results into one DataFrame,
    deduplicates, sorts by discount_pct descending, and returns it.

    At the end prints a summary (rows, sources, highest discount).
    """
    all_rows = []
    failed_urls = []

    for url in URLS:
        rows = _scrape_page(url)
        if rows:
            all_rows.extend(rows)
        else:
            failed_urls.append(url)

    if not all_rows:
        print("WARNING: No discount data was collected from any URL.")
        return pd.DataFrame(columns=[
            "source_site", "company_page", "supplier_name", "plan_name",
            "discount_pct", "discount_type", "discount_note",
            "requires_smart_meter", "time_restriction", "customer_type",
            "context", "deal_url", "source_url", "scraped_at",
        ])

    df = pd.DataFrame(all_rows)

    # ── DEDUPLICATE ───────────────────────────────────────────────────────────
    # Same supplier + plan + percentage + note = duplicate row.
    df = df.drop_duplicates(
        subset=["supplier_name", "plan_name", "discount_pct", "discount_note"]
    ).sort_values("discount_pct", ascending=False).reset_index(drop=True)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n── Scrape complete ──────────────────────")
    print(f"  Rows collected : {len(df)}")
    print(f"  Sources        : {', '.join(df['source_site'].unique())}")
    print(f"  Highest discount: {df['discount_pct'].max()}%")
    if failed_urls:
        print(f"  Failed URLs    : {len(failed_urls)}")
        for u in failed_urls:
            print(f"    • {u}")
    print("─────────────────────────────────────────\n")

    return df


# ── STANDALONE RUN ────────────────────────────────────────────────────────────
# Run this file directly to test the scraper and save a fresh CSV:
#   python src/scraper.py
if __name__ == "__main__":
    from pathlib import Path
    try:
        from config import DISCOUNT_OFFERS_FILE
    except ImportError:
        DISCOUNT_OFFERS_FILE = Path("data/external/electricity_discount_offers.csv")

    df = scrape_offers()
    if not df.empty:
        DISCOUNT_OFFERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DISCOUNT_OFFERS_FILE, index=False, encoding="utf-8-sig")
        print(f"Saved → {DISCOUNT_OFFERS_FILE}")
