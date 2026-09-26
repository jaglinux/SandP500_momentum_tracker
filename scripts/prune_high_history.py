#!/usr/bin/env python3
"""Trim high_history to an approximate trailing window (no LLM).

Keeps only hit dates within the last N calendar days relative to the most
recent date present in high_history.json (not wall-clock today). Tickers with
no remaining dates are dropped; count is recomputed from kept dates.
Rewrites output/high_history.json and output/high_history.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import high_history  # noqa: E402


def prune_history(history: dict, days: int) -> tuple[dict, datetime.date | None, datetime.date | None]:
    """Return (filtered_history, max_date, cutoff). Empty history -> ({}, None, None)."""
    all_dates: list[datetime.date] = []
    for rec in history.values():
        for d in rec.get("dates", []):
            all_dates.append(datetime.strptime(d, "%Y-%m-%d").date())

    if not all_dates:
        return {}, None, None

    max_date = max(all_dates)
    # Inclusive window of `days` calendar days ending at max_date.
    cutoff = max_date - timedelta(days=max(days, 1) - 1)

    filtered: dict = {}
    for ticker, rec in history.items():
        kept = [
            d
            for d in rec.get("dates", [])
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        ]
        if not kept:
            continue
        filtered[ticker] = {
            "count": len(kept),
            "dates": kept,
            "name": rec.get("name", ""),
        }
    return filtered, max_date, cutoff


def main() -> int:
    p = argparse.ArgumentParser(description="Prune high_history to last N days.")
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Inclusive calendar-day window ending at the latest hit date (default: 30).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only; do not write files.",
    )
    args = p.parse_args()

    if args.days < 1:
        print("::error::--days must be >= 1", file=sys.stderr)
        return 1

    history = high_history.load_history()
    before = len(history)
    filtered, max_date, cutoff = prune_history(history, args.days)

    if max_date is None:
        print("No dates in high_history.json; nothing to prune.")
        print("CHANGED=0")
        return 0

    after = len(filtered)
    print(f"window: {cutoff.isoformat()} .. {max_date.isoformat()} ({args.days} days)")
    print(f"tickers: {before} -> {after}")

    if filtered == history:
        print("Already within window; no file changes.")
        print("CHANGED=0")
        return 0

    if args.dry_run:
        print("dry-run: skipping write")
        print("CHANGED=0")
        return 0

    high_history.save_history(filtered)
    high_history.save_history_readable(filtered)
    print(f"wrote {high_history.HISTORY_FILE}")
    print(f"wrote {os.path.join(high_history.OUTPUT_DIR, 'high_history.txt')}")
    print("CHANGED=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
