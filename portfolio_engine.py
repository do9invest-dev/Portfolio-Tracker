"""
Portfolio engine: parse CSVs, download price data, compute equal-weight equity curves.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from pathlib import Path

PORTFOLIOS_DIR = Path("portfolios")
PORTFOLIOS_DIR.mkdir(exist_ok=True)


def save_uploaded_csv(uploaded_file) -> Path:
    """Save an uploaded CSV to the portfolios directory."""
    dest = PORTFOLIOS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def list_portfolios() -> list[str]:
    """Return names of all saved portfolio CSVs (without extension)."""
    return sorted(
        p.stem for p in PORTFOLIOS_DIR.glob("*.csv")
    )


def load_portfolio_csv(name: str) -> pd.DataFrame:
    """Load a portfolio CSV by name and return a cleaned DataFrame."""
    path = PORTFOLIOS_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].str.strip().str.upper()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_rebalance_schedule(df: pd.DataFrame) -> list[tuple[pd.Timestamp, list[str]]]:
    """
    From a portfolio DataFrame, return a list of (date, [tickers]) tuples
    sorted by date.
    """
    schedule = []
    for date, group in df.groupby("Date"):
        tickers = group["Ticker"].tolist()
        schedule.append((pd.Timestamp(date), tickers))
    schedule.sort(key=lambda x: x[0])
    return schedule


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    tickers = list(set(tickers))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    prices = prices.ffill().bfill()
    return prices


def build_equity_curve(
    schedule: list[tuple[pd.Timestamp, list[str]]],
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    initial_value: float = 10000.0,
) -> pd.Series:
    """
    Build a daily equity curve from a rebalance schedule.

    At each rebalance date, the portfolio is split equally among the tickers
    listed for that date. Between rebalance dates, positions drift with the market.
    """
    if not schedule:
        return pd.Series(dtype=float)

    all_tickers = list({t for _, ticks in schedule for t in ticks})
    dl_start = schedule[0][0] - pd.Timedelta(days=5)
    dl_end = end_date or pd.Timestamp.today()

    prices = download_prices(all_tickers, dl_start.strftime("%Y-%m-%d"), dl_end.strftime("%Y-%m-%d"))
    if prices.empty:
        return pd.Series(dtype=float)

    # Build daily returns for each ticker
    returns = prices.pct_change().fillna(0.0)

    # Determine the full date range
    first_date = max(schedule[0][0], prices.index[0])
    if start_date and start_date > first_date:
        first_date = start_date
    last_date = prices.index[-1]
    if end_date and end_date < last_date:
        last_date = end_date

    trading_days = prices.index[(prices.index >= first_date) & (prices.index <= last_date)]
    if len(trading_days) == 0:
        return pd.Series(dtype=float)

    # Walk through trading days
    portfolio_value = initial_value
    equity = {}
    current_tickers = []
    current_weights = np.array([])
    schedule_idx = 0

    for day in trading_days:
        # Check if we need to rebalance
        rebalance = False
        while schedule_idx < len(schedule) and schedule[schedule_idx][0] <= day:
            current_tickers = schedule[schedule_idx][1]
            rebalance = True
            schedule_idx += 1

        if rebalance and len(current_tickers) > 0:
            n = len(current_tickers)
            current_weights = np.ones(n) / n

        if len(current_tickers) == 0:
            equity[day] = portfolio_value
            continue

        # Get today's returns for current holdings
        day_returns = []
        for t in current_tickers:
            if t in returns.columns and day in returns.index:
                day_returns.append(returns.loc[day, t])
            else:
                day_returns.append(0.0)
        day_returns = np.array(day_returns, dtype=float)

        # Portfolio return = weighted sum
        port_return = np.dot(current_weights, day_returns)
        portfolio_value *= (1 + port_return)

        # Update weights to reflect drift
        drifted = current_weights * (1 + day_returns)
        weight_sum = drifted.sum()
        if weight_sum > 0:
            current_weights = drifted / weight_sum

        equity[day] = portfolio_value

    return pd.Series(equity, name="Portfolio")


def build_benchmark_curve(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_value: float = 10000.0,
) -> pd.Series:
    """Build an equity curve for a single benchmark ticker."""
    prices = download_prices([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if prices.empty:
        return pd.Series(dtype=float)
    col = ticker if ticker in prices.columns else prices.columns[0]
    series = prices[col].dropna()
    return (series / series.iloc[0]) * initial_value
