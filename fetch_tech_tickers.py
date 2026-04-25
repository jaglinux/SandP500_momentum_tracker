#!/usr/bin/env python3
"""
Fetch S&P 500 constituents from Wikipedia and save tech_tickers.txt.

Universe = all names in GICS **Information Technology** or **Communication Services**
(no hardcoded symbols). Alphabet (GOOGL/GOOG) and Meta (META) are Communication Services
in GICS—that is why they are not classified as IT.

Run occasionally to refresh when index composition changes.
"""

import io
import os

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "tech_tickers.txt")

TECH_SCREENS_SECTORS = (
    "Information Technology",
    "Communication Services",
)


def main():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    df["_sym"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    mask = df["GICS Sector"].isin(TECH_SCREENS_SECTORS)
    out = df.loc[mask].drop_duplicates(subset="_sym", keep="first")
    tickers = sorted(out["_sym"].tolist())
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for t in tickers:
            f.write(f"{t}\n")
    counts = out.groupby("GICS Sector", sort=False).size()
    print(f"Wrote {len(tickers)} tickers to {OUTPUT}")
    for sector, n in counts.items():
        print(f"  {sector}: {int(n)}")


if __name__ == "__main__":
    main()
