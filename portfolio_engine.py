"""
Portfolio engine: parse CSVs, download price data, compute equal-weight equity curves.

Rebalance logic uses open prices:
  - On a rebalance day, old positions are closed at the OPEN (prev close → open return)
  - New positions are entered at the OPEN (open → close return for that day)
  - All other days use standard close-to-close returns
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
def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download Open and Close prices for a list of tickers.
    Returns (close_prices, open_prices) DataFrames.
    """
    tickers = list(set(tickers))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"]
        open_prices = data["Open"]
    else:
        # Single ticker case
        close_prices = data[["Close"]].copy()
        close_prices.columns = tickers
        open_prices = data[["Open"]].copy()
        open_prices.columns = tickers

    close_prices = close_prices.ffill().bfill()
    open_prices = open_prices.ffill().bfill()
    return close_prices, open_prices


def _get_return(series, day, value_type="close_to_close"):
    """Safely get a return value, defaulting to 0."""
    if day in series.index:
        val = series.loc[day]
        if pd.notna(val):
            return val
    return 0.0


def build_equity_curve(
    schedule: list[tuple[pd.Timestamp, list[str]]],
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    initial_value: float = 10000.0,
) -> pd.Series:
    """
    Build a daily equity curve from a rebalance schedule.

    On rebalance days:
      1. Old positions earn prev_close → open return (morning move)
      2. Portfolio is then split equally among new tickers at the open
      3. New positions earn open → close return (rest of day)

    On normal days:
      - Standard close-to-close returns with drifting weights
    """
    if not schedule:
        return pd.Series(dtype=float)

    all_tickers = list({t for _, ticks in schedule for t in ticks})
    dl_start = schedule[0][0] - pd.Timedelta(days=5)
    dl_end = end_date or pd.Timestamp.today()

    close_prices, open_prices = download_prices(
        all_tickers, dl_start.strftime("%Y-%m-%d"), dl_end.strftime("%Y-%m-%d")
    )
    if close_prices.empty:
        return pd.Series(dtype=float)

    # Precompute return series for each ticker
    # close-to-close: standard daily return
    ret_c2c = close_prices.pct_change().fillna(0.0)

    # prev close to open: (open - prev_close) / prev_close
    ret_c2o = (open_prices - close_prices.shift(1)) / close_prices.shift(1)
    ret_c2o = ret_c2o.fillna(0.0)

    # open to close: (close - open) / open
    ret_o2c = (close_prices - open_prices) / open_prices
    ret_o2c = ret_o2c.fillna(0.0)

    # Determine the full date range
    first_date = max(schedule[0][0], close_prices.index[0])
    if start_date and start_date > first_date:
        first_date = start_date
    last_date = close_prices.index[-1]
    if end_date and end_date < last_date:
        last_date = end_date

    trading_days = close_prices.index[
        (close_prices.index >= first_date) & (close_prices.index <= last_date)
    ]
    if len(trading_days) == 0:
        return pd.Series(dtype=float)

    # Build rebalance date set and lookup for quick access
    rebalance_dates = {}
    for date, tickers in schedule:
        rebalance_dates[date] = tickers

    # Find the nearest trading day on or after each rebalance date
    rebalance_trading_days = {}
    for reb_date, tickers in schedule:
        candidates = trading_days[trading_days >= reb_date]
        if len(candidates) > 0:
            rebalance_trading_days[candidates[0]] = tickers

    # Walk through trading days
    portfolio_value = initial_value
    equity = {}
    current_tickers = []
    current_weights = np.array([])
    schedule_idx = 0

    for day in trading_days:
        is_rebalance = day in rebalance_trading_days

        if is_rebalance:
            new_tickers = rebalance_trading_days[day]

            # ── Step 1: Close out old positions at the open ──────────────
            if len(current_tickers) > 0:
                old_c2o = []
                for t in current_tickers:
                    old_c2o.append(_get_return(ret_c2o[t], day) if t in ret_c2o.columns else 0.0)
                old_c2o = np.array(old_c2o, dtype=float)

                # Old portfolio earns the prev-close-to-open move
                morning_return = np.dot(current_weights, old_c2o)
                portfolio_value *= (1 + morning_return)

            # ── Step 2: Rebalance into new positions at the open ─────────
            current_tickers = new_tickers
            n = len(current_tickers)
            current_weights = np.ones(n) / n

            # ── Step 3: New positions earn the open-to-close move ────────
            new_o2c = []
            for t in current_tickers:
                new_o2c.append(_get_return(ret_o2c[t], day) if t in ret_o2c.columns else 0.0)
            new_o2c = np.array(new_o2c, dtype=float)

            afternoon_return = np.dot(current_weights, new_o2c)
            portfolio_value *= (1 + afternoon_return)

            # Update weights for drift from the afternoon move
            drifted = current_weights * (1 + new_o2c)
            weight_sum = drifted.sum()
            if weight_sum > 0:
                current_weights = drifted / weight_sum

        else:
            # ── Normal day: close-to-close returns ───────────────────────
            if len(current_tickers) == 0:
                equity[day] = portfolio_value
                continue

            day_returns = []
            for t in current_tickers:
                day_returns.append(
                    _get_return(ret_c2c[t], day) if t in ret_c2c.columns else 0.0
                )
            day_returns = np.array(day_returns, dtype=float)

            port_return = np.dot(current_weights, day_returns)
            portfolio_value *= (1 + port_return)

            # Drift weights
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
    close_prices, _ = download_prices(
        [ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    if close_prices.empty:
        return pd.Series(dtype=float)
    col = ticker if ticker in close_prices.columns else close_prices.columns[0]
    series = close_prices[col].dropna()
    return (series / series.iloc[0]) * initial_value
