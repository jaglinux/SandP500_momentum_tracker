# S&P 500 Momentum Tracker

A Python toolkit to track S&P 500 stocks hitting 52-week highs and all-time highs, analyze momentum patterns, fetch news sentiment, and generate AI-driven recommendations.

## Features

- **Momentum Tracking**: Track how many times each stock hits 52W High or ATH over time
- **Price & Volume Analysis**: Monitor 1-day, 1-week, and 1-year price/volume changes
- **Sentiment Analysis**: Fetch news headlines and compute sentiment using VADER
- **AI Recommendations**: Generate buy recommendations using GPT based on momentum, price trends, volume spikes, and sentiment

## Project Structure

| File | Description |
|------|-------------|
| `tracker.py` | Main orchestrator - runs the full pipeline |
| `snapshot.py` | Fetches current quotes, 52W high, ATH, price/volume changes |
| `sentiment.py` | Fetches headlines and computes VADER sentiment |
| `agent.py` | Uses LangChain + GPT to generate AI recommendations |
| `high_history.py` | Tracks momentum (hit count) for each stock over time |
| `fetch_tickers.py` | Fetches S&P 500 ticker list from Wikipedia |
| `ticker.txt` | List of S&P 500 tickers |

## Output Files

All outputs are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `snapshot.txt` | Latest stock data table |
| `high_history.json` | Momentum history (hit counts and dates) |
| `high_history.txt` | Human-readable momentum history |
| `ai_analysis.md` | AI recommendations log (newest at top) |

## Installation

```bash
pip install yfinance pandas tabulate langchain langchain-openai vaderSentiment
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Main Tracker (`tracker.py`)

The main entry point that orchestrates the full pipeline.

```bash
# Full run: fetch prices, update history, get sentiment, generate AI analysis
python tracker.py

# Use cached snapshot (skip yfinance API calls)
python tracker.py --use-cache

# Specify a date for historical tracking
python tracker.py --date 2025-02-07

# History only: update high_history.json, skip sentiment and AI
python tracker.py --history-only

# Analyze only: use cached snapshot + history, run sentiment and AI
python tracker.py --analyze-only
```

#### Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--use-cache` | `-c` | Use cached `snapshot.txt` instead of fetching from yfinance API |
| `--date DATE` | `-d` | Date for tracking (YYYY-MM-DD). Defaults to today |
| `--history-only` | `-H` | Only update `high_history.json`, skip sentiment and AI analysis |
| `--analyze-only` | `-a` | Use cached snapshot and history, fetch sentiment, generate AI analysis |

### Individual Modules

```bash
# Fetch S&P 500 tickers and save to ticker.txt
python fetch_tickers.py

# Fetch current quotes and save snapshot
python snapshot.py

# Fetch sentiment for specific tickers
python sentiment.py

# View high history summary
python high_history.py

# Run AI analysis on existing data
python agent.py
```

## Workflow Examples

### Daily Run
```bash
python tracker.py
```
This fetches latest prices, updates momentum history, gets sentiment, and generates AI recommendations.

### Backfill Historical Data
```bash
python tracker.py --date 2025-02-01 --history-only
python tracker.py --date 2025-02-02 --history-only
python tracker.py --date 2025-02-03 --history-only
```
Use `--history-only` to quickly backfill momentum data without running sentiment/AI.

### Re-run AI Analysis
```bash
python tracker.py --analyze-only
```
Uses existing `snapshot.txt` and `high_history.json` to regenerate AI recommendations without making yfinance API calls.

## AI Analysis Features

The AI analyzes stocks using:
- **Hit Frequency**: Number of times hitting 52W High/ATH (3+ = strong momentum)
- **Price Trends**: 1-day, 1-week, 1-year price changes
- **Volume Spikes**: Unusual volume indicates institutional interest
- **Sentiment**: News sentiment (Bullish/Neutral/Bearish)
- **Proximity to Highs**: How close to 52W High and ATH

Output sections:
- 🚀 **Top Momentum Picks**: Best combination of all factors
- 📈 **Breakout Watch**: First-time breakouts with strong volume
- 🤖 **Tech Momentum**: Tech/AI stocks with momentum
- ⚠️ **Caution Flags**: Stocks showing warning signs
