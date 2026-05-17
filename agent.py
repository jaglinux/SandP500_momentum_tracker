#!/usr/bin/env python3
"""
AI Agent module for NASDAQ High Tracker.
Uses LangChain + GPT as a momentum analyst to recommend stocks based on high frequency.
"""

import os
import re
from datetime import datetime, timezone
from typing import FrozenSet, List, Optional, Tuple

from tabulate import tabulate

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import high_history
import tech_universe

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

GLOBAL_ANALYSIS_SECTIONS: Tuple[str, ...] = (
    "## 🚀 TOP MOMENTUM PICKS",
    "## 📈 BREAKOUT WATCH",
    "## 🤖 TECH MOMENTUM",
    "## ⚠️ RANGE-BOUND",
)

TECH_ANALYSIS_SECTIONS: Tuple[str, ...] = (
    "## 🚀 TOP MOMENTUM PICKS",
    "## 📈 EARLY / BREAKOUT WATCH",
    "## ⚠️ CAUTION",
)


def reorder_analysis_sections(content: str, section_prefixes: Tuple[str, ...]) -> str:
    """Ensure ## sections appear in the requested order; unknown sections keep relative order at end."""
    content = content.strip()
    if not content:
        return content

    parts = re.split(r"(?m)^(## .+)$", content)
    if len(parts) == 1:
        return content

    preamble = parts[0].strip()
    blocks: List[Tuple[str, str]] = []
    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1].rstrip() if i + 1 < len(parts) else ""
        blocks.append((header, body))
        i += 2

    ordered: List[str] = []
    used: set[int] = set()
    for prefix in section_prefixes:
        for idx, (header, body) in enumerate(blocks):
            if idx in used:
                continue
            if header.startswith(prefix) or prefix in header:
                used.add(idx)
                ordered.append(f"{header}\n\n{body}".strip() if body else header)
                break

    for idx, (header, body) in enumerate(blocks):
        if idx not in used:
            ordered.append(f"{header}\n\n{body}".strip() if body else header)

    sections_text = "\n\n".join(ordered)
    if preamble:
        return f"{preamble}\n\n{sections_text}".strip()
    return sections_text


def _history_hits(ticker: str, history: dict) -> int:
    d = history.get(ticker)
    if not isinstance(d, dict):
        return 0
    return int(d.get("count", 0) or 0)


def augment_with_history_notables(
    df_at_highs: pd.DataFrame,
    df_universe: pd.DataFrame,
    history: dict,
    min_hits: int = 5,
    max_extra: int = 45,
) -> pd.DataFrame:
    """
    Append names from the full universe that are in high_history but not in today's at-high
    set. Selection is by **% 1Y** (primary), **Market Cap (B)**, then **cumulative hits**—not
    raw hit rank alone—so extreme yearly performers are not pushed out by high-frequency
    low-beta names.
    """
    if df_universe is None or df_universe.empty:
        if df_at_highs is not None and not df_at_highs.empty:
            out = df_at_highs.copy()
            out["At high today"] = True
            return out
        return df_at_highs

    if df_at_highs is None:
        return df_at_highs

    today = set(df_at_highs["Ticker"].astype(str)) if not df_at_highs.empty else set()
    available = set(df_universe["Ticker"].astype(str))

    candidates: List[str] = []
    for t, d in history.items():
        if t in today or t not in available:
            continue
        cnt = d.get("count", 0) if isinstance(d, dict) else 0
        if cnt < min_hits:
            continue
        candidates.append(t)

    if not candidates:
        out = df_at_highs.copy()
        if not out.empty:
            out["At high today"] = True
        return out

    sub = df_universe[df_universe["Ticker"].isin(candidates)].drop_duplicates("Ticker", keep="first").copy()
    sub["_hits"] = sub["Ticker"].map(lambda x: _history_hits(x, history))
    y1 = pd.to_numeric(sub["% 1Y"], errors="coerce")
    mcap = pd.to_numeric(sub["Market Cap (B)"], errors="coerce")
    sub["__y1"] = y1
    sub["__mcap"] = mcap
    sub = sub.sort_values(
        by=["__y1", "__mcap", "_hits"],
        ascending=[False, False, False],
        na_position="last",
    )
    extra_tickers = sub["Ticker"].head(max_extra).tolist()

    if not extra_tickers:
        out = df_at_highs.copy()
        if not out.empty:
            out["At high today"] = True
        return out

    print(
        f"Adding {len(extra_tickers)} names not at today's high "
        f"(ranked by % 1Y, market cap, cumulative hits; cap {max_extra})..."
    )

    df_extra = df_universe[df_universe["Ticker"].isin(extra_tickers)].drop_duplicates(
        "Ticker", keep="first"
    )
    order = {t: i for i, t in enumerate(extra_tickers)}
    df_extra["_ord"] = df_extra["Ticker"].map(order)
    df_extra = df_extra.sort_values("_ord").drop(columns=["_ord"])

    orig_cols = [c for c in df_at_highs.columns]
    for col in orig_cols:
        if col not in df_extra.columns:
            df_extra[col] = pd.NA
    df_extra = df_extra[orig_cols]
    df_extra["At high today"] = False

    base = df_at_highs.copy()
    if base.empty:
        return df_extra
    base["At high today"] = True
    return pd.concat([base, df_extra], ignore_index=True)


def create_stocks_table(df: pd.DataFrame, history: dict = None) -> tuple:
    """
    Create a formatted table of stocks at highs with momentum data.
    Returns: (table_string, summary_dict)
    """
    if df.empty:
        return "No stocks at highs.", {"total": 0, "ath": 0, "52w_only": 0}
    
    # Load history if not provided
    if history is None:
        history = high_history.load_history()
    
    # Summary counts only for names actually at 52W/ATH today (excludes history-only rows)
    if "At high today" in df.columns:
        dft = df[df["At high today"] == True]
    else:
        dft = df

    at_ath = len(dft[dft["At ATH"] == True]) if "At ATH" in dft.columns else 0
    at_52w = len(dft[dft["At 52W High"] == True]) if "At 52W High" in dft.columns else 0
    at_52w_only = at_52w - at_ath  # 52W high but not ATH

    summary = {
        "total": len(dft),
        "ath": at_ath,
        "52w_only": at_52w_only
    }
    
    # Add momentum columns from history
    df = df.copy()
    df["Hits"] = df["Ticker"].apply(lambda t: history.get(t, {}).get("count", 1))
    df["Last Hit"] = df["Ticker"].apply(
        lambda t: history.get(t, {}).get("dates", [])[-1] if history.get(t, {}).get("dates") else "Today"
    )
    df["2nd Last"] = df["Ticker"].apply(
        lambda t: history.get(t, {}).get("dates", [])[-2] if len(history.get(t, {}).get("dates", [])) >= 2 else ""
    )
    
    # Select columns for display (include momentum, price changes, volume)
    display_cols = ["Ticker", "Name", "At high today", "Price", "Market Cap (B)", "Hits", "Last Hit", "2nd Last",
                    "% 1D", "% 1W", "% 1Y", "Vol", "Vol % 1D",
                    "% From 52W High", "% From ATH"]
    
    # Filter to only columns that exist
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()
    
    # Sort: today's highs first, then strongest yearly performance, then hits
    sort_keys = []
    sort_asc = []
    if "At high today" in df_display.columns:
        sort_keys.append("At high today")
        sort_asc.append(False)
    if "% 1Y" in df_display.columns:
        df_display["_sort_y1"] = pd.to_numeric(df_display["% 1Y"], errors="coerce").fillna(-1e18)
        sort_keys.append("_sort_y1")
        sort_asc.append(False)
    sort_keys.append("Hits")
    sort_asc.append(False)
    df_display = df_display.sort_values(sort_keys, ascending=sort_asc)
    if "_sort_y1" in df_display.columns:
        df_display = df_display.drop(columns=["_sort_y1"])
    
    # Reset index and add row number
    df_display = df_display.reset_index(drop=True)
    df_display.index = df_display.index + 1  # Start from 1
    df_display.index.name = "#"
    
    # Format the table with row numbers
    table = tabulate(
        df_display,
        headers="keys",
        tablefmt="github",
        showindex=True,
        floatfmt=".2f",
    )
    
    return table, summary


def generate_recommendations(table_md: str, history: dict = None) -> str:
    """
    Use GPT as a momentum analyst to generate recommendations.
    Uses hits history, price changes, volume, and distance from highs.
    """
    # Get top momentum stocks for context
    momentum_context = ""
    if history:
        sorted_history = sorted(history.items(), key=lambda x: x[1].get("count", 0), reverse=True)
        top_momentum = sorted_history[:25]
        if top_momentum:
            momentum_lines = [f"- {t}: {d['count']} hits (last: {d['dates'][-1] if d.get('dates') else 'N/A'})" 
                             for t, d in top_momentum]
            momentum_context = "\n\nTop 25 Momentum Leaders by cumulative high-hit frequency (may include names not in the table if they are off the exact high today):\n" + "\n".join(momentum_lines)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a STOCK MOMENTUM ANALYST. Your job is to find stocks with REAL price momentum, not range-bound stocks.

DATA COLUMNS:
- **At high today**: True = in today's at-52W-or-ATH list; False = still listed from history + momentum ranking (pulled back from the tick threshold)—consider sector context when relevant.
- **% 1Y**: Year-over-year price change - THIS IS THE PRIMARY MOMENTUM INDICATOR
- **% 1W**: Weekly price change - shows current trend direction
- **% 1D**: Daily price change
- **Hits**: Times stock hit highs - WARNING: High hits + low % 1Y = RANGE-BOUND (avoid!)
- **Vol % 1D**: Volume spike vs yesterday (>50% = institutional interest)

CRITICAL MOMENTUM RULES:
1. **% 1Y is PRIMARY** - True momentum = stocks UP 50%+ in a year. Stocks up 100%+ are exceptional.
2. **AVOID RANGE-BOUND STOCKS** - High "Hits" with low % 1Y (<30%) means the stock keeps hitting the SAME ceiling repeatedly. This is NOT momentum - it's resistance!
3. **% 1W must be positive** - Confirms current uptrend is active
4. **Volume confirms conviction** - Vol % 1D > 50% shows institutions are buying

RANGE-BOUND WARNING:
- If a stock has many Hits but % 1Y < 30%, it's hitting the same price ceiling repeatedly = AVOID
- Example: 18 hits but only 7% yearly gain = stock is stuck, not trending

TRUE MOMENTUM STOCKS have:
- % 1Y > 50% (ideally 70%+) - significant price appreciation
- % 1W > 2% - active current uptrend  
- Positive % 1D - today's momentum
- Volume spike confirming the move

## 🚀 TOP MOMENTUM PICKS
Select 3-5 stocks with the HIGHEST % 1Y gains (50%+) that also have positive % 1W.
For each: **TICKER** ($Price) - 1Y: +X%, 1W: +X%, Vol spike: +X% - Brief analysis.

## 📈 BREAKOUT WATCH  
New breakouts (Hits=1-2) with strong % 1Y and volume spike.
These are early-stage momentum plays.

## 🤖 TECH MOMENTUM
Tech/semiconductor/AI stocks with % 1Y > 70% and positive weekly trend.
Skip if none qualify.

## ⚠️ RANGE-BOUND / CAUTION
Stocks with HIGH Hits but LOW % 1Y - these are hitting resistance, not breaking out.
Also flag: negative % 1W despite a daily pop.

OUTPUT ORDER (required): print the ## sections below in this exact sequence — **## 🚀 TOP MOMENTUM PICKS first**, caution sections last. Do not reorder.

Be specific with numbers. Prioritize % 1Y over Hits count. Call out large **Market Cap (B)** names with extreme % 1Y when they appear in the table."""),
        ("human", """S&P 500 stocks at 52-week high or all-time high:

{table}
{momentum_context}

Find stocks with REAL momentum (high % 1Y), not range-bound stocks (high hits but low % 1Y):""")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt_template | llm
    
    response = chain.invoke({"table": table_md, "momentum_context": momentum_context})
    return response.content.strip()


def generate_tech_recommendations(
    table_md: str, history: dict, tech_tickers: FrozenSet[str]
) -> str:
    """Tech-screen momentum: GICS Information Technology + Communication Services names in the table."""
    momentum_context = ""
    if history and tech_tickers:
        filtered = {k: v for k, v in history.items() if k in tech_tickers}
        sorted_hist = sorted(filtered.items(), key=lambda x: x[1].get("count", 0), reverse=True)
        top_momentum = sorted_hist[:25]
        if top_momentum:
            momentum_lines = [
                f"- {t}: {d['count']} hits (last: {d['dates'][-1] if d.get('dates') else 'N/A'})"
                for t, d in top_momentum
            ]
            momentum_context = (
                "\n\nTop 25 tech-screen momentum leaders by cumulative high-hit frequency "
                "(may include names not in the table if off the exact high today):\n"
                + "\n".join(momentum_lines)
            )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a tech & digital platforms momentum analyst for S&P 500 names only.
The table is **GICS Information Technology** and **Communication Services** (e.g. software, semis, internet platforms, media/telecom in that sector). Note: Alphabet and Meta are Communication Services in GICS, not IT—that is standard industry classification.

DATA COLUMNS:
- **At high today**: True = in today's at-52W-or-ATH list; False = listed from history + momentum ranking.
- **% 1Y**, **% 1W**, **% 1D**, **Hits**, **Vol % 1D**, **Market Cap (B)** — interpret through **tech / comms** lens (growth, product cycles, ads/cloud, semis, connectivity).

Focus on:
- True price momentum (high % 1Y) vs range-bound high-hit names with weak % 1Y
- Leadership within this universe (mega-cap platforms vs semis vs telecom)
- Volume confirmation

## 🚀 TOP MOMENTUM PICKS
3-5 names with the strongest % 1Y (typically 50%+) and constructive % 1W; cite price, cap, volume.

## 📈 EARLY / BREAKOUT WATCH
Newer high-hit or smaller names in this universe with strong % 1Y and volume.

## ⚠️ CAUTION
High Hits + low % 1Y (range-bound). Also flag weak % 1W despite a one-day bounce.

OUTPUT ORDER (required): print the ## sections below in this exact sequence — **## 🚀 TOP MOMENTUM PICKS first**, ## ⚠️ CAUTION last. Do not reorder.

Be specific with numbers. Do not discuss sectors outside IT and Communication Services.""",
            ),
            (
                "human",
                """S&P 500 — Information Technology + Communication Services — stocks at 52-week high or all-time high:

{table}
{momentum_context}

Momentum within tech + communication services only (no banks, energy, healthcare, staples, etc.):""",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt_template | llm
    response = chain.invoke({"table": table_md, "momentum_context": momentum_context})
    return response.content.strip()


def save_analysis_md(
    table: str,
    summary: dict,
    recommendations: str,
    snapshot_date: str = None,
    filename: str = "ai_analysis.md",
    header_title: str = "# S&P 500 Momentum Tracker - AI Analysis\n\n",
    section_stocks_title: str = "### 📊 Stocks at Highs (Full Momentum Data)\n\n",
    section_order: Tuple[str, ...] = GLOBAL_ANALYSIS_SECTIONS,
):
    """Save table and recommendations to markdown; AI picks first, full table after."""
    now = datetime.now(timezone.utc)
    generated_timestamp = now.strftime("%Y-%m-%d %H:%M UTC")

    if snapshot_date is None:
        snapshot_date = now.strftime("%Y-%m-%d")

    filepath = os.path.join(OUTPUT_DIR, filename)
    recommendations = reorder_analysis_sections(recommendations, section_order)

    new_entry = f"""---
## Snapshot: {snapshot_date} | Generated: {generated_timestamp}

### 🚀 AI Momentum Analysis

{recommendations}

{section_stocks_title}**Total: {summary['total']} stocks** | 🏆 At ATH: {summary['ath']} | 🔥 At 52W High only: {summary['52w_only']}

> **Columns**: **At high today** = in today's at-52W/ATH set; **False** = still shown (history + ranked by % 1Y / market cap / hits). | Hits = cumulative high hits | % 1D/1W/1Y = price changes | Vol % 1D = volume vs prior day

{table}

"""

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_content = f.read()

    if existing_content.startswith(header_title):
        existing_content = existing_content[len(header_title) :]

    old_header = "# S&P 500 High Tracker - AI Analysis\n\n"
    if existing_content.startswith(old_header):
        existing_content = existing_content[len(old_header) :]

    content = header_title + new_entry + existing_content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def main(
    df_stocks: pd.DataFrame = None,
    snapshot_date: str = None,
    df_universe: Optional[pd.DataFrame] = None,
):
    """
    Main function to generate AI momentum analysis.
    
    Args:
        df_stocks: DataFrame with stocks at highs (required)
        snapshot_date: Date of the snapshot data (YYYY-MM-DD)
        df_universe: Full S&P snapshot (all rows). Used to append high-hit names that are
            not at today's exact high so they still appear in the analysis.
    """
    print("\n=== Generating AI Momentum Analysis ===")
    
    if df_stocks is None or df_stocks.empty:
        print("No data to analyze.")
        return
    
    # Load high history for momentum data
    history = high_history.load_history()
    print(f"Loaded momentum history for {len(history)} stocks")

    df_for_table = augment_with_history_notables(df_stocks, df_universe, history)
    
    # Create stocks table with momentum data
    print("Creating stocks table with momentum data...")
    table, summary = create_stocks_table(df_for_table, history)
    print(f"\n🏆 At ATH: {summary['ath']} | 🔥 At 52W High only: {summary['52w_only']} | Total: {summary['total']}")
    print(f"\n{table}\n")
    
    # Generate AI momentum recommendations
    print("Generating AI momentum recommendations...")
    recommendations = generate_recommendations(table, history)
    print(f"\n{recommendations}\n")
    
    filepath = save_analysis_md(table, summary, recommendations, snapshot_date)
    print(f"Analysis saved to {filepath}")

    tech = tech_universe.load_tech_tickers()
    if not tech:
        print(
            "Skipping tech AI report: tech_tickers.txt missing or empty — run: python fetch_tech_tickers.py"
        )
        return

    df_tech_highs = tech_universe.filter_tech_dataframe(df_stocks, tech)
    uni_tech = (
        tech_universe.filter_tech_dataframe(df_universe, tech)
        if df_universe is not None
        else None
    )
    if uni_tech is None or uni_tech.empty:
        print("Skipping tech AI report: no IT/Communication Services rows in universe snapshot.")
        return

    seed = df_tech_highs if not df_tech_highs.empty else df_stocks.iloc[0:0].copy()
    df_tech_full = augment_with_history_notables(seed, uni_tech, history)
    if df_tech_full.empty:
        print(
            "Skipping tech AI report: no tech-screen names at highs and none added from history augmentation."
        )
        return

    print("\n=== Generating Tech (IT + Communication Services) AI Analysis ===")
    table_t, summary_t = create_stocks_table(df_tech_full, history)
    print(
        f"\n🏆 [Tech screen] At ATH: {summary_t['ath']} | 🔥 At 52W only: {summary_t['52w_only']} | At-high count: {summary_t['total']}"
    )
    print(f"\n{table_t}\n")
    print("Generating tech-screen AI recommendations...")
    rec_t = generate_tech_recommendations(table_t, history, tech)
    print(f"\n{rec_t}\n")
    path_t = save_analysis_md(
        table_t,
        summary_t,
        rec_t,
        snapshot_date,
        filename="ai_tech_analysis.md",
        header_title="# S&P 500 Momentum Tracker - Tech (IT + Communication Services) AI Analysis\n\n",
        section_stocks_title="### 📊 Tech Screen — IT & Communication Services — Stocks at Highs\n\n",
        section_order=TECH_ANALYSIS_SECTIONS,
    )
    print(f"Tech analysis saved to {path_t}")
