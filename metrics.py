"""
Portfolio performance metrics.
"""
import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    """Total return as a decimal (e.g. 0.25 = 25%)."""
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return total ** (365.25 / days) - 1


def annualized_volatility(equity_curve: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    daily = equity_curve.pct_change().dropna()
    if len(daily) < 2:
        return 0.0
    return daily.std() * np.sqrt(252)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative decimal."""
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1
    return drawdowns.min()


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    daily = equity_curve.pct_change().dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    excess = daily.mean() - risk_free_rate / 252
    return (excess / daily.std()) * np.sqrt(252)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown time series."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def compute_all_metrics(equity_curve: pd.Series) -> dict:
    """Return a dict of all key metrics."""
    return {
        "Total Return": f"{total_return(equity_curve):.2%}",
        "CAGR": f"{cagr(equity_curve):.2%}",
        "Ann. Volatility": f"{annualized_volatility(equity_curve):.2%}",
        "Max Drawdown": f"{max_drawdown(equity_curve):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(equity_curve):.2f}",
    }
