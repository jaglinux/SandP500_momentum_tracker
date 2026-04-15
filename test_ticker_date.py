#!/usr/bin/env python3
"""
One-off probe: yfinance daily bar for a single ticker and calendar date.
Usage:
  python test_ticker_date.py
  python test_ticker_date.py SNDK 2025-01-23
"""

import argparse
import traceback
from datetime import datetime, timedelta, timezone

import yfinance as yf


def main():
    parser = argparse.ArgumentParser(description="Test yfinance history for one ticker on one date.")
    parser.add_argument("ticker", nargs="?", default="SNDK", help="Symbol (default: SNDK)")
    parser.add_argument("date", nargs="?", default="2025-01-23", help="YYYY-MM-DD (default: 2025-01-23)")
    args = parser.parse_args()
    sym = args.ticker.strip().upper()
    d = datetime.strptime(args.date, "%Y-%m-%d")
    end = d + timedelta(days=1)

    print(f"Ticker: {sym}")
    print(f"Target date: {args.date} (weekday {d.strftime('%A')})")
    print(f"history(start={d:%Y-%m-%d}, end={end:%Y-%m-%d})  # exclusive end, same as snapshot.py")
    print("-" * 60)

    t = yf.Ticker(sym)

    first_trade_date = None
    try:
        info = t.info
        print("info keys (sample):", "shortName" in info, "symbol" in info)
        if info:
            print("  shortName:", info.get("shortName"))
            print("  quoteType:", info.get("quoteType"))
            print("  exchange:", info.get("exchange"))
            ft_ms = info.get("firstTradeDateMilliseconds")
            print("  firstTradeDateMilliseconds:", ft_ms)
            if ft_ms:
                first_trade_date = datetime.fromtimestamp(
                    float(ft_ms) / 1000.0, tz=timezone.utc
                ).date()
                print("  first trade date (UTC, from info):", first_trade_date)
                if d.date() < first_trade_date:
                    print()
                    print(
                        f"  NOTE: {args.date} is BEFORE Yahoo's first bar for this symbol ({first_trade_date})."
                    )
                    print(
                        '  Yahoo then returns empty history / "Data doesn\'t exist" — not rate limiting.'
                    )
    except Exception as e:
        print("t.info raised:", repr(e))
        traceback.print_exc()

    print("-" * 60)
    try:
        hist = t.history(start=d.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        print("history rows:", len(hist))
        if hist.empty:
            if first_trade_date and d.date() < first_trade_date:
                print("EMPTY — expected: requested date is before first trade (see NOTE above).")
            else:
                print(
                    "EMPTY — no daily bar (holiday/weekend, delisted window, or other Yahoo gap). "
                    "Compare to first trade date above if shown."
                )
        else:
            print(hist)
    except Exception as e:
        print("t.history raised:", repr(e))
        traceback.print_exc()

    print("-" * 60)
    print("Wider window (5 trading days before through day after) for context:")
    try:
        wide_start = d - timedelta(days=10)
        wide_end = end + timedelta(days=5)
        h2 = t.history(start=wide_start.strftime("%Y-%m-%d"), end=wide_end.strftime("%Y-%m-%d"))
        print(f"rows: {len(h2)}")
        if not h2.empty:
            print(h2.tail(8))
    except Exception as e:
        print("wide history raised:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
