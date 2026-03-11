"""
Portfolio Dashboard — Equal-Weight Rebalancing Tracker
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_engine import (
    save_uploaded_csv,
    list_portfolios,
    load_portfolio_csv,
    get_rebalance_schedule,
    build_equity_curve,
    build_benchmark_curve,
)
from metrics import compute_all_metrics, drawdown_series

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #e2e8f0 !important;
        font-size: 1.4rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #cbd5e0; }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helpers ─────────────────────────────────────────────────────
COLORS = {
    "portfolio": "#63b3ed",
    "benchmark": "#fc8181",
    "drawdown_fill": "rgba(99,179,237,0.15)",
    "bench_dd_fill": "rgba(252,129,129,0.12)",
    "grid": "rgba(255,255,255,0.06)",
    "bg": "#0e1117",
    "paper": "#0e1117",
    "text": "#cbd5e0",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"], size=13),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hovermode="x unified",
)


def styled_layout(**overrides):
    layout = {**LAYOUT_DEFAULTS}
    layout.update(overrides)
    return layout


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Portfolio Dashboard")
    st.markdown("---")

    # Upload
    st.markdown("#### Upload Portfolio")
    uploaded = st.file_uploader(
        "CSV with Ticker & Date columns",
        type=["csv"],
        help="Column A: Ticker, Column B: Date. Each unique date is a rebalance date.",
    )
    if uploaded:
        save_uploaded_csv(uploaded)
        st.success(f"Saved **{uploaded.name}**")

    st.markdown("---")

    # Portfolio selection
    portfolios = list_portfolios()
    if not portfolios:
        st.info("Upload a CSV to get started.")
        st.stop()

    st.markdown("#### Select Portfolio")
    selected = st.selectbox("Portfolio", portfolios, label_visibility="collapsed")

    st.markdown("---")

    # Date range
    st.markdown("#### Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2024-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp.today())

    st.markdown("---")

    # Benchmark
    st.markdown("#### Benchmark")
    show_bench = st.checkbox("Compare to benchmark", value=True)
    bench_ticker = st.text_input("Ticker", value="SPY", disabled=not show_bench)

    st.markdown("---")

    # Portfolio holdings info
    st.markdown("#### Holdings Schedule")
    df_csv = load_portfolio_csv(selected)
    schedule = get_rebalance_schedule(df_csv)

    for i, (date, tickers) in enumerate(schedule):
        label = "Current" if i == len(schedule) - 1 else f"Period {i+1}"
        with st.expander(f"{label} — {date.strftime('%Y-%m-%d')}", expanded=(i == len(schedule) - 1)):
            for t in sorted(tickers):
                st.markdown(f"` {t} ` — {1/len(tickers):.1%}")


# ── Main panel ───────────────────────────────────────────────────────────────
st.markdown(f"# {selected}")

# Build curves
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

equity = build_equity_curve(schedule, start_date=start_ts, end_date=end_ts)

if equity.empty:
    st.error("No data returned. Check your tickers and date range.")
    st.stop()

bench_equity = None
if show_bench and bench_ticker:
    bench_equity = build_benchmark_curve(bench_ticker, start_ts, end_ts)
    if not bench_equity.empty:
        # Align to same start value
        bench_equity = bench_equity.reindex(equity.index, method="ffill")
        bench_equity = (bench_equity / bench_equity.dropna().iloc[0]) * equity.iloc[0]

# ── Metrics row ──────────────────────────────────────────────────────────────
port_metrics = compute_all_metrics(equity)

if bench_equity is not None and not bench_equity.dropna().empty:
    bench_metrics = compute_all_metrics(bench_equity.dropna())
else:
    bench_metrics = None

cols = st.columns(5)
metric_keys = ["Total Return", "CAGR", "Ann. Volatility", "Max Drawdown", "Sharpe Ratio"]
for i, key in enumerate(metric_keys):
    with cols[i]:
        delta = None
        if bench_metrics:
            # Parse percentages/floats for delta
            pv = port_metrics[key].replace("%", "")
            bv = bench_metrics[key].replace("%", "")
            try:
                diff = float(pv) - float(bv)
                delta = f"{diff:+.2f}{'%' if '%' in port_metrics[key] else ''} vs {bench_ticker}"
            except ValueError:
                pass
        st.metric(key, port_metrics[key], delta=delta)

st.markdown("")

# ── Equity curve ─────────────────────────────────────────────────────────────
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=equity.index, y=equity.values,
    name="Portfolio",
    line=dict(color=COLORS["portfolio"], width=2.5),
    hovertemplate="$%{y:,.0f}<extra>Portfolio</extra>",
))

if bench_equity is not None and not bench_equity.dropna().empty:
    fig_eq.add_trace(go.Scatter(
        x=bench_equity.index, y=bench_equity.values,
        name=bench_ticker,
        line=dict(color=COLORS["benchmark"], width=2, dash="dot"),
        hovertemplate="$%{y:,.0f}<extra>" + bench_ticker + "</extra>",
    ))

# Mark rebalance dates
for date, tickers in schedule:
    if date >= equity.index[0] and date <= equity.index[-1]:
        fig_eq.add_vline(
            x=date, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash")
        )

fig_eq.update_layout(**styled_layout(
    title="Equity Curve",
    yaxis_title="Portfolio Value ($)",
    yaxis_tickprefix="$",
    yaxis_tickformat=",",
    height=440,
))
st.plotly_chart(fig_eq, use_container_width=True)

# ── Drawdown chart ───────────────────────────────────────────────────────────
dd = drawdown_series(equity)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd.index, y=dd.values,
    name="Portfolio",
    fill="tozeroy",
    fillcolor=COLORS["drawdown_fill"],
    line=dict(color=COLORS["portfolio"], width=1.5),
    hovertemplate="%{y:.2%}<extra>Portfolio DD</extra>",
))

if bench_equity is not None and not bench_equity.dropna().empty:
    bench_dd = drawdown_series(bench_equity.dropna())
    fig_dd.add_trace(go.Scatter(
        x=bench_dd.index, y=bench_dd.values,
        name=bench_ticker,
        fill="tozeroy",
        fillcolor=COLORS["bench_dd_fill"],
        line=dict(color=COLORS["benchmark"], width=1, dash="dot"),
        hovertemplate="%{y:.2%}<extra>" + bench_ticker + " DD</extra>",
    ))

fig_dd.update_layout(**styled_layout(
    title="Drawdown",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    height=320,
))
st.plotly_chart(fig_dd, use_container_width=True)

# ── Metrics comparison table ─────────────────────────────────────────────────
st.markdown("### Performance Summary")

table_data = {"Metric": metric_keys, "Portfolio": [port_metrics[k] for k in metric_keys]}
if bench_metrics:
    table_data[bench_ticker] = [bench_metrics[k] for k in metric_keys]

df_table = pd.DataFrame(table_data).set_index("Metric")
st.dataframe(df_table, use_container_width=True)

# ── Holdings table ───────────────────────────────────────────────────────────
with st.expander("📋 Full Rebalance Schedule", expanded=False):
    display_rows = []
    for date, tickers in schedule:
        for t in sorted(tickers):
            display_rows.append({
                "Rebalance Date": date.strftime("%Y-%m-%d"),
                "Ticker": t,
                "Weight": f"{1/len(tickers):.1%}",
            })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
