#!/usr/bin/env python3
"""
AI Agent module for NASDAQ High Tracker.
Uses LangChain + GPT as a momentum analyst to recommend stocks based on high frequency.
"""

import os
from datetime import datetime, timezone
from tabulate import tabulate

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import high_history

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


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
    
    # Count stocks at ATH and 52W high
    at_ath = len(df[df["At ATH"] == True]) if "At ATH" in df.columns else 0
    at_52w = len(df[df["At 52W High"] == True]) if "At 52W High" in df.columns else 0
    at_52w_only = at_52w - at_ath  # 52W high but not ATH
    
    summary = {
        "total": len(df),
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
    display_cols = ["Ticker", "Name", "Price", "Market Cap (B)", "Hits", "Last Hit", "2nd Last",
                    "% 1D", "% 1W", "% 1Y", "Vol", "Vol % 1D",
                    "Sentiment", "% From 52W High", "% From ATH"]
    
    # Filter to only columns that exist
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()
    
    # Sort by Hits (momentum) descending, then by % From 52W High
    df_display = df_display.sort_values(["Hits", "% From 52W High"], ascending=[False, False])
    
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
        top_momentum = sorted_history[:10]
        if top_momentum:
            momentum_lines = [f"- {t}: {d['count']} hits (last: {d['dates'][-1] if d.get('dates') else 'N/A'})" 
                             for t, d in top_momentum]
            momentum_context = "\n\nTop 10 Momentum Leaders (all-time high hit frequency):\n" + "\n".join(momentum_lines)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a STOCK MOMENTUM ANALYST with expertise in technical analysis. Identify and recommend stocks with the strongest momentum using ALL available data.

DATA COLUMNS EXPLAINED:
- **Hits**: Number of times stock hit 52W High or ATH (higher = stronger momentum pattern)
- **Last Hit / 2nd Last**: Dates of recent high hits (recent = active momentum)
- **% 1D**: Price change in last 1 day (positive = today's momentum)
- **% 1W**: Price change in last 1 week (positive = short-term trend)
- **% 1Y**: Price change over 1 year (positive = long-term uptrend)
- **Vol**: Current trading volume
- **Vol % 1D**: Volume change vs yesterday (high = unusual activity, institutional interest)
- **% From 52W High**: Distance from 52-week high (0% = at high, negative = below)
- **% From ATH**: Distance from all-time high (0% = at ATH)
- **Sentiment**: News sentiment (Bullish/Neutral/Bearish)

MOMENTUM SCORING FRAMEWORK:
1. **Hit Frequency** (Primary): 3+ hits = proven winner, 2 = building, 1 = new breakout
2. **Price Trend**: Look for stocks with positive % 1D, % 1W, AND % 1Y (triple positive = strong trend)
3. **Volume Confirmation**: High Vol % 1D (>50%) confirms breakout, institutions are buying
4. **Proximity to Highs**: Closer to 0% means currently breaking out
5. **Sentiment**: Bullish sentiment confirms the move has legs

BEST PICKS have:
- Multiple hits (3+) showing consistent momentum
- Triple positive price changes (1D, 1W, 1Y all positive)
- Volume spike (Vol % 1D > 20%) confirming buyer interest
- At or very near 52W High / ATH
- Bullish sentiment

## 🚀 TOP MOMENTUM PICKS
Select 3-5 stocks with the BEST combination of ALL factors above.
For each pick: **TICKER** ($Price) - Hits: X, 1D: +X%, 1W: +X%, Vol spike: +X% - Why this is a strong momentum play.

## 📈 BREAKOUT WATCH
Stocks showing first-time breakouts (Hits=1) with strong volume and price action.
These could be the START of a new momentum trend. Look for high Vol % 1D.

## 🤖 TECH MOMENTUM
Tech/AI stocks (semiconductors, cloud, software) with strong momentum signals.
Skip if no qualifying tech stocks.

## ⚠️ CAUTION FLAGS
Any stocks at highs showing warning signs:
- Bearish sentiment despite price high (divergence)
- Low volume on breakout (weak conviction)
- Negative weekly trend (% 1W < 0) despite daily pop

Be specific with numbers from the data. Don't just list - ANALYZE and EXPLAIN."""),
        ("human", """Here are today's S&P 500 stocks at 52-week high or all-time high with full momentum data:

{table}
{momentum_context}

Analyze ALL the data (hits, price changes, volume, sentiment) and provide your recommendations:""")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt_template | llm
    
    response = chain.invoke({"table": table_md, "momentum_context": momentum_context})
    return response.content.strip()


def save_analysis_md(table: str, summary: dict, recommendations: str, snapshot_date: str = None):
    """Save table and recommendations to markdown file, prepending new entries at the top."""
    now = datetime.now(timezone.utc)
    generated_timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    
    # Use snapshot_date if provided, otherwise use today
    if snapshot_date is None:
        snapshot_date = now.strftime("%Y-%m-%d")
    
    filepath = os.path.join(OUTPUT_DIR, "ai_analysis.md")
    
    new_entry = f"""---
## Snapshot: {snapshot_date} | Generated: {generated_timestamp}

### 📊 Stocks at Highs (Full Momentum Data)

**Total: {summary['total']} stocks** | 🏆 At ATH: {summary['ath']} | 🔥 At 52W High only: {summary['52w_only']}

> **Columns**: Hits = high hit count | % 1D/1W/1Y = price changes | Vol % 1D = volume spike | Sentiment = news

{table}

### 🚀 AI Momentum Analysis

{recommendations}

"""
    
    # Read existing content if file exists
    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_content = f.read()
    
    # Prepend new entry (newest at top)
    header = "# S&P 500 Momentum Tracker - AI Analysis\n\n"
    if existing_content.startswith(header):
        existing_content = existing_content[len(header):]
    
    # Also handle old header format
    old_header = "# S&P 500 High Tracker - AI Analysis\n\n"
    if existing_content.startswith(old_header):
        existing_content = existing_content[len(old_header):]
    
    content = header + new_entry + existing_content
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def main(df_stocks: pd.DataFrame = None, snapshot_date: str = None):
    """
    Main function to generate AI momentum analysis.
    
    Args:
        df_stocks: DataFrame with stocks at highs (required)
        snapshot_date: Date of the snapshot data (YYYY-MM-DD)
    """
    print("\n=== Generating AI Momentum Analysis ===")
    
    if df_stocks is None or df_stocks.empty:
        print("No data to analyze.")
        return
    
    # Load high history for momentum data
    history = high_history.load_history()
    print(f"Loaded momentum history for {len(history)} stocks")
    
    # Create stocks table with momentum data
    print("Creating stocks table with momentum data...")
    table, summary = create_stocks_table(df_stocks, history)
    print(f"\n🏆 At ATH: {summary['ath']} | 🔥 At 52W High only: {summary['52w_only']} | Total: {summary['total']}")
    print(f"\n{table}\n")
    
    # Generate AI momentum recommendations
    print("Generating AI momentum recommendations...")
    recommendations = generate_recommendations(table, history)
    print(f"\n{recommendations}\n")
    
    # Save to file (table first, then recommendations)
    filepath = save_analysis_md(table, summary, recommendations, snapshot_date)
    print(f"Analysis saved to {filepath}")
