#!/usr/bin/env python3
"""Load and merge manual ticker additions (survive fetch_tickers / fetch_tech_tickers)."""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRA_TICKERS_FILE = os.path.join(SCRIPT_DIR, "extra_tickers.txt")
EXTRA_TECH_TICKERS_FILE = os.path.join(SCRIPT_DIR, "extra_tech_tickers.txt")
TECH_TICKERS_FILE = os.path.join(SCRIPT_DIR, "tech_tickers.txt")

# S&P lists both share classes; we track GOOGL only (Class A).
EXCLUDED_TICKERS = frozenset({"GOOG"})


def _normalize(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def read_ticker_file(path: str) -> list[str]:
    """One symbol per line; blank lines and # comments ignored."""
    if not os.path.exists(path):
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(_normalize(line))
    return out


def filter_tickers(tickers: list[str]) -> list[str]:
    """Drop symbols in EXCLUDED_TICKERS (e.g. GOOG when using GOOGL)."""
    return [t for t in tickers if _normalize(t) not in EXCLUDED_TICKERS]


def merge_tickers(*groups: list[str]) -> list[str]:
    """Union of symbol lists, sorted, deduplicated, with exclusions applied."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for sym in group:
            s = _normalize(sym)
            if s and s not in seen and s not in EXCLUDED_TICKERS:
                seen.add(s)
                merged.append(s)
    merged.sort()
    return merged
