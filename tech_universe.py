#!/usr/bin/env python3
"""
Tickers for the tech-focused report: GICS Information Technology + Communication Services
(S&P 500), loaded from tech_tickers.txt (see fetch_tech_tickers.py).
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TECH_TICKERS_FILE = os.path.join(SCRIPT_DIR, "tech_tickers.txt")


def load_tech_tickers() -> frozenset:
    """Return ticker symbols (IT + Communication Services). Empty if file missing."""
    if not os.path.exists(TECH_TICKERS_FILE):
        return frozenset()
    with open(TECH_TICKERS_FILE, "r", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def filter_tech_dataframe(df: "pd.DataFrame", tech: frozenset) -> "pd.DataFrame":
    """Keep rows whose Ticker is in the tech-screen universe."""
    import pandas as pd

    if df is None or not tech:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    if df.empty:
        return df.copy()
    return df[df["Ticker"].astype(str).isin(tech)].copy()
