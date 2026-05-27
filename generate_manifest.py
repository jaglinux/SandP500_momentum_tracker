#!/usr/bin/env python3
"""Build manifest.json for index.html (run after tracker or before deploy)."""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from ticker_lists import merge_tickers, read_ticker_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "manifest.json")
COMBINED_TICKERS_PATH = os.path.join(SCRIPT_DIR, "tickers_combined.txt")

TICKER_SOURCES = (
    ("ticker.txt", "S&P 500"),
    ("extra_tickers.txt", "Manual / watchlist"),
    ("tech_tickers.txt", "Tech screen (S&P IT + Comm Services)"),
    ("extra_tech_tickers.txt", "Manual tech screen"),
)

# Output section order (others follow alphabetically)
OUTPUT_FILE_ORDER = (
    "ai_tech_analysis.md",
    "ai_analysis.md",
)


def _file_entry(rel_path: str, label: Optional[str] = None) -> dict:
    path = os.path.join(SCRIPT_DIR, rel_path.replace("/", os.sep))
    st = os.stat(path)
    return {
        "path": rel_path.replace("\\", "/"),
        "label": label or os.path.basename(rel_path),
        "size": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def write_combined_tickers() -> list[str]:
    """Merge all ticker list files into tickers_combined.txt; return symbol list."""
    groups: list[list[str]] = []
    header_lines = [
        "# Combined ticker list (unique symbols, sorted)",
        "# Sources:",
    ]
    for filename, title in TICKER_SOURCES:
        path = os.path.join(SCRIPT_DIR, filename)
        syms = read_ticker_file(path)
        groups.append(syms)
        header_lines.append(f"#   - {filename} ({title}): {len(syms)}")

    combined = merge_tickers(*groups)
    header_lines.append(f"# Total unique: {len(combined)}")
    header_lines.append("")

    body = "\n".join(header_lines + combined) + "\n"
    with open(COMBINED_TICKERS_PATH, "w", encoding="utf-8") as f:
        f.write(body)
    return combined


def _output_sort_key(filename: str) -> tuple:
    try:
        rank = OUTPUT_FILE_ORDER.index(filename)
    except ValueError:
        rank = len(OUTPUT_FILE_ORDER)
    return (rank, filename.lower())


def build_manifest() -> dict:
    sections = []

    output_items = []
    if os.path.isdir(OUTPUT_DIR):
        names = [
            n
            for n in os.listdir(OUTPUT_DIR)
            if os.path.isfile(os.path.join(OUTPUT_DIR, n))
        ]
        for name in sorted(names, key=_output_sort_key):
            output_items.append(_file_entry(f"output/{name}"))
    if output_items:
        sections.append({"title": "Output", "files": output_items})

    combined = write_combined_tickers()
    if combined:
        sections.append(
            {
                "title": "Ticker lists",
                "files": [
                    _file_entry(
                        "tickers_combined.txt",
                        f"All tickers ({len(combined)} unique)",
                    )
                ],
            }
        )

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }


def main():
    manifest = build_manifest()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n = sum(len(s["files"]) for s in manifest["sections"])
    print(f"Wrote {MANIFEST_PATH} ({n} files)")


if __name__ == "__main__":
    main()
