"""
Part B + C — Live Streamlit dashboard.

Deploy free at Streamlit Community Cloud:
  1. Push this repo to GitHub (public or private with access granted)
  2. Go to share.streamlit.io → New app → select repo → app.py
  3. Click Deploy

On every visit:
  - Fetches the latest closed BTCUSDT 1h bar from Binance
  - Runs 10,000-path GBM Monte Carlo
  - Shows current price, 95% forecast range, last-50-bar chart, backtest metrics
  - Appends prediction to live_predictions.jsonl (Part C persistence)
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import evaluate, winkler_score
from model import fetch_btc_hourly, predict_interval

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Forecast | AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
)

BACKTEST_FILE  = "backtest_results.jsonl"
LIVE_PRED_FILE = "live_predictions.jsonl"


# ── File I/O helpers ─────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_backtest_metrics() -> dict | None:
    preds = _read_jsonl(BACKTEST_FILE)
    if not preds:
        return None
    m = evaluate(preds)
    m["hits"]   = sum(p["covered"] for p in preds)
    m["misses"] = m["n_predictions"] - m["hits"]
    return m


def append_live_prediction(record: dict) -> None:
    with open(LIVE_PRED_FILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def load_live_history() -> list[dict]:
    return _read_jsonl(LIVE_PRED_FILE)


# ── Data fetch + model run (cached 55s so rapid reloads don't spam Binance) ──

@st.cache_data(ttl=55)
def get_data_and_forecast():
    prices = fetch_btc_hourly(limit=500)
    low_95, high_95, mu, sigma, nu = predict_interval(
        prices, confidence=0.95, n_sims=10_000, lookback=500
    )
    return prices, low_95, high_95, mu, sigma, nu


# ── Main layout ──────────────────────────────────────────────────────────────

st.title("BTC/USDT Next-Hour Forecast")
st.caption(
    "Geometric Brownian Motion · Student-t fat tails · 10,000-path Monte Carlo  |  "
    "AlphaI x Polaris Challenge"
)

with st.spinner("Fetching data and running Monte Carlo..."):
    prices, low_95, high_95, mu, sigma, nu = get_data_and_forecast()

current_price = float(prices.iloc[-1])
prev_price    = float(prices.iloc[-2])
pct_change    = (current_price - prev_price) / prev_price * 100

# Persist this visit's prediction (Part C)
live_rec = {
    "bar_time":      prices.index[-1].isoformat(),
    "current_price": current_price,
    "low_95":        low_95,
    "high_95":       high_95,
    "width":         high_95 - low_95,
    "visited_at":    datetime.now(timezone.utc).isoformat(),
}
append_live_prediction(live_rec)

# ── Section 1 — Backtest metrics (Part A) ────────────────────────────────────

st.subheader("Part A: 30-Day Backtest Metrics (720 bars)")
bt = load_backtest_metrics()
if bt:
    c1, c2, c3, c4, c5 = st.columns(5)
    delta_cov = f"{bt['coverage_95'] - 0.95:+.4f} vs target"
    c1.metric("Coverage 95%",       f"{bt['coverage_95']:.4f}", delta_cov)
    c2.metric("Hits / Misses",      f"{bt['hits']} / {bt['misses']}")
    c3.metric("Avg Width",          f"${bt['avg_width']:,.0f}")
    c4.metric("Mean Winkler Score", f"${bt['mean_winkler_95']:,.0f}")
    c5.metric("Predictions",        f"{bt['n_predictions']:,}")
else:
    st.info(
        "Backtest metrics not yet available. "
        "Run `python backtest.py` locally, commit `backtest_results.jsonl`, and redeploy."
    )

st.divider()

# ── Section 2 — Live forecast ─────────────────────────────────────────────────

st.subheader("Part B: Current Forecast")

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC Price",            f"${current_price:,.2f}",  f"{pct_change:+.2f}%")
m2.metric("Forecast Low  (95%)",  f"${low_95:,.2f}")
m3.metric("Forecast High (95%)",  f"${high_95:,.2f}")
m4.metric("Range Width",          f"${high_95 - low_95:,.2f}")

# ── Chart: last 50 bars + shaded forecast ribbon ─────────────────────────────

last_50 = prices.iloc[-50:]
ts_list  = last_50.index.tolist()
next_ts  = ts_list[-1] + pd.Timedelta(hours=1)

fig = go.Figure()

# Price history line
fig.add_trace(go.Scatter(
    x=ts_list, y=last_50.values,
    mode="lines", name="BTC Close (1h)",
    line=dict(color="#F7931A", width=2),
))

# Latest close dot
fig.add_trace(go.Scatter(
    x=[ts_list[-1]], y=[current_price],
    mode="markers", name="Latest close",
    marker=dict(color="#F7931A", size=9, symbol="circle"),
))

# Shaded 95% CI ribbon
fig.add_trace(go.Scatter(
    x=[ts_list[-1], next_ts, next_ts, ts_list[-1]],
    y=[high_95,     high_95, low_95,  low_95],
    fill="toself",
    fillcolor="rgba(30, 120, 255, 0.18)",
    line=dict(color="rgba(0,0,0,0)"),
    name=f"95% CI  [{low_95:,.0f} to {high_95:,.0f}]",
))

# Dashed boundary lines
for y_val in [low_95, high_95]:
    fig.add_trace(go.Scatter(
        x=[ts_list[-1], next_ts],
        y=[current_price, y_val],
        mode="lines",
        line=dict(color="#1E78FF", dash="dot", width=1.5),
        showlegend=False,
    ))

fig.update_layout(
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    template="plotly_dark",
    height=480,
    margin=dict(l=60, r=20, t=30, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model parameters (this run)"):
    p1, p2, p3 = st.columns(3)
    p1.metric("Hourly drift μ",     f"{mu:.6f}")
    p2.metric("EWMA volatility σ",  f"{sigma:.6f}")
    p3.metric("Student-t DoF ν",    f"{nu:.2f}")

st.divider()

# ── Section 3 — Live prediction history (Part C) ──────────────────────────────

history = load_live_history()

if len(history) > 1:
    st.subheader("Part C: Live Prediction History")

    df_h = pd.DataFrame(history)
    df_h["bar_time"] = pd.to_datetime(df_h["bar_time"])
    df_h = df_h.sort_values("bar_time").drop_duplicates("bar_time").reset_index(drop=True)

    fig2 = go.Figure()

    # BTC price at each prediction time
    fig2.add_trace(go.Scatter(
        x=df_h["bar_time"], y=df_h["current_price"],
        mode="lines+markers", name="BTC at prediction time",
        line=dict(color="#F7931A", width=2),
        marker=dict(size=5),
    ))

    # Confidence ribbon
    x_ribbon = list(df_h["bar_time"]) + list(df_h["bar_time"])[::-1]
    y_ribbon  = list(df_h["high_95"]) + list(df_h["low_95"])[::-1]
    fig2.add_trace(go.Scatter(
        x=x_ribbon, y=y_ribbon,
        fill="toself", fillcolor="rgba(30, 120, 255, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI ribbon",
    ))

    fig2.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="Bar time (UTC)",
        yaxis_title="BTC Price (USDT)",
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_cols = ["bar_time", "current_price", "low_95", "high_95", "width"]
    st.dataframe(
        df_h[display_cols]
        .tail(30)
        .set_index("bar_time")
        .rename(columns={
            "current_price": "BTC Price",
            "low_95": "Forecast Low",
            "high_95": "Forecast High",
            "width": "Width",
        })
        .style.format("${:,.2f}"),
        use_container_width=True,
    )
else:
    st.info("Live prediction history will appear here after more visits.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
    "Data: Binance Vision API (public, no API key required)"
)
