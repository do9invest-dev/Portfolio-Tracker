# Portfolio Dashboard

A Streamlit dashboard for tracking equal-weight stock portfolios with rebalancing, powered by Yahoo Finance.

## Features

- **CSV-driven portfolios** — upload a CSV with `Ticker` and `Date` columns; each unique date triggers an equal-weight rebalance
- **Multiple portfolios** — upload as many CSVs as you want; each is named after its file
- **Compare up to 3** — select up to three portfolios to overlay on the same charts
- **Persistent storage** — uploaded CSVs are saved to `portfolios/` and survive app restarts
- **Benchmark comparison** — overlay SPY (or any ticker) on your equity curve
- **Interactive charts** — Plotly equity curve and drawdown chart with rebalance markers
- **Key metrics** — Total Return, CAGR, Annualized Volatility, Max Drawdown, Sharpe Ratio

## CSV Format

```csv
Ticker,Date
AAPL,2024-01-01
MSFT,2024-01-01
NVDA,2024-01-01
AMZN,2024-03-01
META,2024-03-01
GOOGL,2024-03-01
```

Each unique date is a rebalance date. All tickers listed for that date become equal-weight holdings until the next rebalance date.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
portfolio-dashboard/
├── app.py                # Streamlit UI
├── portfolio_engine.py   # CSV parsing, price downloads, equity curve logic
├── metrics.py            # Performance metric calculations
├── requirements.txt
├── README.md
└── portfolios/           # Uploaded CSVs are stored here
```

## How It Works

1. **Upload** a CSV in the sidebar
2. **Select** a portfolio from the dropdown
3. **Adjust** the date range and optional benchmark
4. **View** equity curves, drawdowns, and metrics

Prices are fetched from Yahoo Finance with `yfinance` and cached for one hour.
