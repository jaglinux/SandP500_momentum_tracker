#!/usr/bin/env python3
"""Append a symbol to manual ticker lists if not already present (used by GitHub Actions)."""

from __future__ import annotations

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ticker_lists import (  # noqa: E402
    EXCLUDED_TICKERS,
    EXTRA_TECH_TICKERS_FILE,
    EXTRA_TICKERS_FILE,
    TECH_TICKERS_FILE,
    read_ticker_file,
)

TICKER_TXT = os.path.join(SCRIPT_DIR, "ticker.txt")

# Reasonable NYSE/NASDAQ-style symbol after normalization.
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


def normalize(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def _append_line(path: str, symbol: str) -> None:
    needs_leading_nl = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_leading_nl = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_leading_nl:
            f.write("\n")
        f.write(symbol + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Add ticker to extra_tickers / extra_tech_tickers.")
    p.add_argument("ticker", help="Ticker symbol, e.g. NVDA or BRK-B")
    p.add_argument(
        "--tech",
        action="store_true",
        help="Also add to extra_tech_tickers.txt if not already in the tech universe.",
    )
    args = p.parse_args()

    sym = normalize(args.ticker)
    if not sym:
        print("::error::Ticker is empty", file=sys.stderr)
        return 1
    if sym in EXCLUDED_TICKERS:
        print(f"::error::Symbol {sym} is excluded (use GOOGL instead of GOOG).", file=sys.stderr)
        return 1
    if not _SYMBOL_RE.match(sym):
        print(f"::error::Invalid ticker after normalization: {sym}", file=sys.stderr)
        return 1

    tracked_anywhere = set()
    for path in (TICKER_TXT, EXTRA_TICKERS_FILE, TECH_TICKERS_FILE, EXTRA_TECH_TICKERS_FILE):
        tracked_anywhere.update(read_ticker_file(path))

    tech_universe = set(read_ticker_file(TECH_TICKERS_FILE)) | set(read_ticker_file(EXTRA_TECH_TICKERS_FILE))

    changed = False

    if sym in tracked_anywhere:
        print(f"skip-extra: {sym} already present in ticker sources (no change to extra_tickers.txt)")
    else:
        _append_line(EXTRA_TICKERS_FILE, sym)
        tracked_anywhere.add(sym)
        print(f"added-extra: appended {sym} to extra_tickers.txt")
        changed = True

    if args.tech:
        if sym in tech_universe:
            print(f"skip-tech: {sym} already in tech_tickers / extra_tech_tickers")
        else:
            _append_line(EXTRA_TECH_TICKERS_FILE, sym)
            tech_universe.add(sym)
            print(f"added-tech: appended {sym} to extra_tech_tickers.txt")
            changed = True

    if changed:
        print("CHANGED=1")
    else:
        print("CHANGED=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
