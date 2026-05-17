#!/usr/bin/env python3
"""
Tickers for the tech-focused report: GICS IT + Communication Services (tech_tickers.txt)
plus manual adds in extra_tech_tickers.txt (see fetch_tech_tickers.py).
"""

from typing import TYPE_CHECKING

from ticker_lists import (
    EXTRA_TECH_TICKERS_FILE,
    TECH_TICKERS_FILE,
    merge_tickers,
    read_ticker_file,
)

if TYPE_CHECKING:
    import pandas as pd


def load_tech_tickers() -> frozenset:
    """Return tech-screen symbols (S&P IT/Comm + extra_tech_tickers.txt)."""
    base = read_ticker_file(TECH_TICKERS_FILE)
    extras = read_ticker_file(EXTRA_TECH_TICKERS_FILE)
    return frozenset(merge_tickers(base, extras))


def filter_tech_dataframe(df: "pd.DataFrame", tech: frozenset) -> "pd.DataFrame":
    """Keep rows whose Ticker is in the tech-screen universe."""
    import pandas as pd

    if df is None or not tech:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    if df.empty:
        return df.copy()
    return df[df["Ticker"].astype(str).isin(tech)].copy()
