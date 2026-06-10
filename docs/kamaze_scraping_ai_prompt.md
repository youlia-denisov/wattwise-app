# AI prompt: scrape electricity discount offers to CSV

You are a web-scraping assistant.

Scrape electricity discount offers from the listed Hebrew web pages and create one downloadable CSV:

electricity_discount_offers.csv

Columns:
source_site, company_page, supplier_name, plan_name,
discount_pct, discount_type, discount_note,
requires_smart_meter, time_restriction, customer_type,
context, deal_url, source_url, scraped_at

Rules:
- Use requests with headers:
  - User-Agent
  - Accept-Language: he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7
- Decode Hebrew correctly.
- Remove RTL control marks.
- Normalize whitespace.
- Extract discounts in both formats:
  - 10%
  - %10
- Keep only percentages between 1 and 30.
- Do not invent discounts.
- If no discount is found, skip the row.

**Plan cards — preferred extraction target:**
- Each plan is presented as a card with an <h3> title and a percentage badge
  (Hebrew pattern: `%X הנחה` or `%X - %Y הנחה`).
- Prefer the percentage badge value over any figure mentioned in surrounding body text.
- Extract the card's own link (href on the <h3> anchor or nearest <a>) and store it
  in `deal_url`. This is the canonical identifier for the plan.

**One row per discount value:**
- If a single plan has multiple discount values that depend on a condition
  (consumption tier, year of contract, customer type, etc.), create one row per value.
- Record the condition in `discount_note`
  (e.g. "year 1", "monthly bill ≤149 ILS", "existing gas customers").

**discount_type field:**
- Set to `discount` when the benefit is described as הנחה (discount off the bill).
- Set to `cashback` when the benefit is described as צבירה (accumulation/cashback
  to a wallet or app), even if the percentage is the same.

- Detect smart meter requirement by searching for:
  - מונה חכם
- Translate the extracted context to English.
- Prefer plan cards. If cards are not useful, fall back to tables, headings,
  and nearby text blocks.
- If the fetched page body is empty or contains no discount data, log the URL as
  FAILED:JS_RENDERED and skip — do not count it as a successful source.
- Continue if one page fails.
- Print failed URL and error.
- Drop duplicate rows (same supplier_name + plan_name + discount_pct + discount_note).
- Sort by discount_pct descending.
- Save CSV using:
  encoding="utf-8-sig"

Target URLs:
["https://www.kamaze.co.il/Companies/82227/Cellcom/electrical-power",
    "https://www.kamaze.co.il/Companies/82228/Hot/electrical-power",
    "https://www.kamaze.co.il/Companies/82260/Bezeq/electrical-power",
    "https://www.kamaze.co.il/Companies/82287/Partner/electrical-power",
    "https://www.kamaze.co.il/Companies/82471/supergas-electric/electrical-power",
    "https://www.kamaze.co.il/Companies/82501/amisragas--electric/electrical-power",
    "https://www.kamaze.co.il/Companies/82476/pazgas-electric/electrical-power",
    "https://www.kamaze.co.il/Companies/82617/Ramy-Levi/electrical-power",
    "https://israelelectricity.com/%d7%94%d7%a9%d7%95%d7%90%d7%aa-%d7%94%d7%a0%d7%97%d7%95%d7%aa-%d7%97%d7%a9%d7%9e%d7%9c/",
]

At the end print:
- number of rows
- source sites collected
- highest discount found
- saved CSV path
