#!/usr/bin/env python3
"""
Snapshot module for NASDAQ High Tracker.
Fetches current quotes, 52-week high, and all-time high data for S&P 500 stocks.
Supports historical date fetching for backfilling data.
"""

import os
import yfinance as yf
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timezone, timedelta

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_FILE = os.path.join(SCRIPT_DIR, "ticker.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

from ticker_lists import EXTRA_TICKERS_FILE, merge_tickers, read_ticker_file


def load_tickers() -> list:
    """
    Load tickers from ticker.txt plus extra_tickers.txt (manual adds).
    Exits with error if the base file doesn't exist.
    """
    if not os.path.exists(TICKER_FILE):
        print(f"Error: ticker.txt not found at {TICKER_FILE}")
        print(f"Run 'python fetch_tickers.py' first to generate the S&P 500 ticker list.")
        exit(1)

    base = read_ticker_file(TICKER_FILE)
    extras = read_ticker_file(EXTRA_TICKERS_FILE)
    tickers = merge_tickers(base, extras)

    if not tickers:
        print("Error: ticker.txt is empty.")
        exit(1)

    if extras:
        print(
            f"Loaded {len(tickers)} tickers ({len(base)} S&P + {len(extras)} from extra_tickers.txt)"
        )
    else:
        print(f"Loaded {len(tickers)} tickers from ticker.txt")
    return tickers


TICKERS = load_tickers()


def fetch_quote(ticker: str, target_date: datetime = None) -> dict:
    """
    Fetch price, 52-week high, all-time high, and price/volume changes for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        target_date: If provided, fetch historical data for this date.
                     If None, fetch current/latest data.
    
    Returns a dict with all relevant high-tracking metrics.
    """
    t = yf.Ticker(ticker)
    price = None
    volume = None
    market_cap = None
    yr_high = None
    yr_low = None
    all_time_high = None
    company_name = None
    
    # Price/volume change variables
    price_1d_ago = None
    price_1w_ago = None
    price_1y_ago = None
    vol_1d_ago = None
    vol_1w_ago = None
    vol_1y_ago = None

    # Get company name from info (always current)
    try:
        info = t.info
        company_name = info.get("shortName") or info.get("longName")
        market_cap = info.get("marketCap")
    except Exception:
        pass

    if target_date is not None:
        # HISTORICAL MODE: Fetch data for specific date
        try:
            # Get price for target date
            start_date = target_date
            end_date = target_date + timedelta(days=1)
            hist_day = t.history(start=start_date.strftime("%Y-%m-%d"), 
                                  end=end_date.strftime("%Y-%m-%d"))
            if not hist_day.empty:
                price = hist_day["Close"].iloc[-1]
                volume = hist_day["Volume"].iloc[-1]
            
            # Get 52-week high/low as of target date (look back 252 trading days)
            year_ago = target_date - timedelta(days=365)
            hist_year = t.history(start=year_ago.strftime("%Y-%m-%d"), 
                                   end=end_date.strftime("%Y-%m-%d"))
            if not hist_year.empty:
                yr_high = hist_year["High"].max()
                yr_low = hist_year["Low"].min()
                
                # Get historical prices/volumes for comparison
                if len(hist_year) >= 2:
                    price_1d_ago = hist_year["Close"].iloc[-2]
                    vol_1d_ago = hist_year["Volume"].iloc[-2]
                if len(hist_year) >= 6:
                    price_1w_ago = hist_year["Close"].iloc[-6]
                    vol_1w_ago = hist_year["Volume"].iloc[-6]
                # For yearly: use first available data point (which is ~1 year ago)
                if len(hist_year) >= 200:  # At least ~200 trading days
                    price_1y_ago = hist_year["Close"].iloc[0]
                    vol_1y_ago = hist_year["Volume"].iloc[0]
            
            # Get all-time high as of target date
            hist_all = t.history(start="1900-01-01", 
                                  end=end_date.strftime("%Y-%m-%d"))
            if not hist_all.empty:
                all_time_high = hist_all["High"].max()
                
        except Exception:
            pass
    else:
        # CURRENT MODE: Fetch latest data
        # Try fast_info first (faster)
        try:
            fi = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "last_close", None)
            if market_cap is None:
                market_cap = getattr(fi, "market_cap", None)
            yr_high = getattr(fi, "year_high", None)
            yr_low = getattr(fi, "year_low", None)
        except Exception:
            pass

        # Fallback to info for missing values
        try:
            if price is None:
                price = info.get("regularMarketPrice")
            if yr_high is None:
                yr_high = info.get("fiftyTwoWeekHigh")
            if yr_low is None:
                yr_low = info.get("fiftyTwoWeekLow")
            volume = info.get("regularMarketVolume")
        except Exception:
            pass

        # Fetch historical data for price/volume changes and ATH
        try:
            hist = t.history(period="max")
            if not hist.empty:
                all_time_high = hist["High"].max()
                
                # Get comparison prices/volumes by index
                if len(hist) >= 2:
                    price_1d_ago = hist["Close"].iloc[-2]
                    vol_1d_ago = hist["Volume"].iloc[-2]
                if len(hist) >= 6:
                    price_1w_ago = hist["Close"].iloc[-6]
                    vol_1w_ago = hist["Volume"].iloc[-6]
                
                # For yearly: find data point closest to 1 year ago by date
                if len(hist) >= 200:
                    today = datetime.now(timezone.utc)
                    one_year_ago = today - timedelta(days=365)
                    # Filter to data around 1 year ago (within a week window)
                    hist_tz_aware = hist.copy()
                    if hist_tz_aware.index.tz is None:
                        hist_tz_aware.index = hist_tz_aware.index.tz_localize('UTC')
                    mask = hist_tz_aware.index <= one_year_ago
                    if mask.any():
                        price_1y_ago = hist_tz_aware.loc[mask, "Close"].iloc[-1]
                        vol_1y_ago = hist_tz_aware.loc[mask, "Volume"].iloc[-1]
        except Exception:
            pass

    # Calculate percentages from highs
    pct_from_52w_high = None
    pct_from_ath = None
    
    if price is not None and yr_high not in (None, 0):
        pct_from_52w_high = (price / yr_high - 1.0) * 100.0
    
    if price is not None and all_time_high not in (None, 0):
        pct_from_ath = (price / all_time_high - 1.0) * 100.0

    # Calculate price changes
    pct_chg_1d = None
    pct_chg_1w = None
    pct_chg_1y = None
    
    if price is not None and price_1d_ago not in (None, 0):
        pct_chg_1d = (price / price_1d_ago - 1.0) * 100.0
    if price is not None and price_1w_ago not in (None, 0):
        pct_chg_1w = (price / price_1w_ago - 1.0) * 100.0
    if price is not None and price_1y_ago not in (None, 0):
        pct_chg_1y = (price / price_1y_ago - 1.0) * 100.0
    
    # Calculate volume changes
    vol_chg_1d = None
    vol_chg_1w = None
    vol_chg_1y = None
    
    if volume is not None and vol_1d_ago not in (None, 0):
        vol_chg_1d = (volume / vol_1d_ago - 1.0) * 100.0
    if volume is not None and vol_1w_ago not in (None, 0):
        vol_chg_1w = (volume / vol_1w_ago - 1.0) * 100.0
    if volume is not None and vol_1y_ago not in (None, 0):
        vol_chg_1y = (volume / vol_1y_ago - 1.0) * 100.0

    # Determine if at or near highs (within 2%)
    at_52w_high = pct_from_52w_high is not None and pct_from_52w_high >= -2.0
    at_ath = pct_from_ath is not None and pct_from_ath >= -2.0

    return {
        "Ticker": ticker,
        "Name": company_name,
        "Price": price,
        "Market Cap (B)": None if market_cap is None else market_cap / 1e9,
        "52W High": yr_high,
        "52W Low": yr_low,
        "% From 52W High": pct_from_52w_high,
        "All-Time High": all_time_high,
        "% From ATH": pct_from_ath,
        "At 52W High": at_52w_high,
        "At ATH": at_ath,
        "% 1D": pct_chg_1d,
        "% 1W": pct_chg_1w,
        "% 1Y": pct_chg_1y,
        "Vol": volume,
        "Vol % 1D": vol_chg_1d,
        "Vol % 1W": vol_chg_1w,
        "Vol % 1Y": vol_chg_1y,
    }


def main(date_str: str = None) -> pd.DataFrame:
    """
    Fetch quotes for all tickers and return a DataFrame.
    
    Args:
        date_str: Optional date string (YYYY-MM-DD) for historical data.
                  If None, fetches current/latest data.
    
    Returns:
        DataFrame with stock data
    """
    # Parse date string if provided
    target_date = None
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            print(f"Fetching HISTORICAL quotes for {len(TICKERS)} stocks for date: {date_str}...")
        except ValueError:
            print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.")
            exit(1)
    else:
        print(f"Fetching quotes for {len(TICKERS)} S&P 500 stocks...")
    
    rows = []
    for symbol in TICKERS:
        print(f"  {symbol}...", end=" ", flush=True)
        quote = fetch_quote(symbol, target_date=target_date)
        rows.append(quote)
        print("✓")

    df = pd.DataFrame(rows)

    # Sort by: At 52W High (True first), then At ATH (True first), then % From 52W High
    df = df.sort_values(
        by=["At 52W High", "At ATH", "% From 52W High"],
        ascending=[False, False, False],
    )

    # Print table (selected columns for readability)
    display_cols = [
        "Ticker", "Name", "Price", "Market Cap (B)", 
        "% 1D", "% 1W", "% 1Y",
        "52W High", "% From 52W High", "All-Time High", "% From ATH", 
        "At 52W High", "At ATH",
        "Vol", "Vol % 1D", "Vol % 1W", "Vol % 1Y"
    ]
    
    # Format table output
    table_output = tabulate(
        df[display_cols],
        headers="keys",
        tablefmt="github",
        showindex=False,
        floatfmt=".2f",
    )
    
    print("\n" + table_output)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to single output file (overwrite each time)
    output_file = os.path.join(OUTPUT_DIR, "snapshot.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"S&P 500 High Tracker Snapshot\n")
        f.write(f"Data Date: {date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Total stocks: {len(df)}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_output)
        f.write("\n")
    
    print(f"\nSaved to {output_file}")

    return df


if __name__ == "__main__":
    main()
