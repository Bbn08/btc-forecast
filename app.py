"""
Part B + C — Live Streamlit dashboard.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from backtest import evaluate, winkler_score
from model import fetch_btc_hourly, predict_interval

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Forecast | AlphaI x Polaris",
    page_icon="₿",
    layout="wide",
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="btc_refresh")

BACKTEST_FILE  = "backtest_results.jsonl"
LIVE_PRED_FILE = "live_predictions.jsonl"


# ── File I/O helpers ──────────────────────────────────────────────────────────

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


def resolve_predictions(preds: list[dict], prices: pd.Series) -> list[dict]:
    """
    For each prediction made at bar_time T, the actual is the close of bar T+1h.
    Look that up in the prices series (indexed by open_time UTC).
    """
    # Normalise price index to UTC-aware for reliable lookup
    idx = prices.index
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC")

    price_map = {ts: float(p) for ts, p in zip(idx, prices.values)}

    resolved = []
    for p in preds:
        entry = dict(p)
        bar_ts = pd.Timestamp(p["bar_time"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        actual_ts = bar_ts + pd.Timedelta(hours=1)

        # Look for actual close (allow 1-min slop for DST / ms rounding)
        actual = None
        for ts, price in price_map.items():
            if abs((ts - actual_ts).total_seconds()) < 120:
                actual = price
                break

        entry["actual"] = actual
        if actual is not None:
            entry["result"] = "Hit" if p["low_95"] <= actual <= p["high_95"] else "Miss"
        else:
            entry["result"] = "Pending"
        resolved.append(entry)

    return resolved


# ── Data fetch + model (cached 55s to avoid hammering Binance on every rerun) -

@st.cache_data(ttl=55)
def get_data_and_forecast():
    prices = fetch_btc_hourly(limit=500)
    low_95, high_95, mu, sigma, nu = predict_interval(
        prices, confidence=0.95, n_sims=10_000, lookback=500
    )
    return prices, low_95, high_95, mu, sigma, nu


# ── Main layout ───────────────────────────────────────────────────────────────

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

# ── Part A: Backtest metrics ──────────────────────────────────────────────────

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

# ── Part B: Current forecast ──────────────────────────────────────────────────

st.subheader("Part B: Current Forecast")

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC Price",           f"${current_price:,.2f}", f"{pct_change:+.2f}%")
m2.metric("Forecast Low (95%)",  f"${low_95:,.2f}")
m3.metric("Forecast High (95%)", f"${high_95:,.2f}")
m4.metric("Range Width",         f"${high_95 - low_95:,.2f}")

# Chart: last 50 bars + shaded forecast ribbon
last_50 = prices.iloc[-50:]
ts_list  = last_50.index.tolist()
next_ts  = ts_list[-1] + pd.Timedelta(hours=1)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts_list, y=last_50.values,
    mode="lines", name="BTC Close (1h)",
    line=dict(color="#F7931A", width=2),
))
fig.add_trace(go.Scatter(
    x=[ts_list[-1]], y=[current_price],
    mode="markers", name="Latest close",
    marker=dict(color="#F7931A", size=9, symbol="circle"),
))
fig.add_trace(go.Scatter(
    x=[ts_list[-1], next_ts, next_ts, ts_list[-1]],
    y=[high_95, high_95, low_95, low_95],
    fill="toself",
    fillcolor="rgba(30, 120, 255, 0.18)",
    line=dict(color="rgba(0,0,0,0)"),
    name=f"95% CI [{low_95:,.0f} to {high_95:,.0f}]",
))
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
    p1.metric("Hourly drift mu",    f"{mu:.6f}")
    p2.metric("EWMA volatility sigma", f"{sigma:.6f}")
    p3.metric("Student-t DoF nu",   f"{nu:.2f}")

st.divider()

# ── Part C: Live prediction history ──────────────────────────────────────────

raw_history = load_live_history()

# Deduplicate by bar_time, keep latest visited_at per bar
df_raw = pd.DataFrame(raw_history) if raw_history else pd.DataFrame()

if not df_raw.empty:
    df_raw["bar_time"] = pd.to_datetime(df_raw["bar_time"])
    df_raw = (
        df_raw.sort_values("visited_at")
              .drop_duplicates("bar_time", keep="last")
              .reset_index(drop=True)
    )
    preds_list = df_raw.to_dict("records")
    resolved   = resolve_predictions(preds_list, prices)

    resolved_only = [r for r in resolved if r["result"] != "Pending"]
    hits   = sum(1 for r in resolved_only if r["result"] == "Hit")
    misses = len(resolved_only) - hits
    live_cov = hits / len(resolved_only) if resolved_only else None

    st.subheader(f"Part C: Live Prediction History ({len(resolved)} entries)")

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Live Coverage",   f"{live_cov:.3f}" if live_cov is not None else "N/A")
    lc2.metric("Resolved Hits",   hits)
    lc3.metric("Resolved Misses", misses)

    # Build display table (newest first)
    rows = []
    for r in sorted(resolved, key=lambda x: x["bar_time"], reverse=True):
        bar_ts = pd.Timestamp(r["bar_time"])
        result = r["result"]
        result_label = (
            "Hit"     if result == "Hit"
            else "Miss"    if result == "Miss"
            else "Pending"
        )
        rows.append({
            "Timestamp (UTC)": bar_ts.strftime("%Y-%m-%d %H:%M"),
            "BTC at Pred":     f"${r['current_price']:,.2f}",
            "Lo":              f"${r['low_95']:,.2f}",
            "Hi":              f"${r['high_95']:,.2f}",
            "Actual":          f"${r['actual']:,.2f}" if r["actual"] else "-",
            "Result":          result_label,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.subheader("Part C: Live Prediction History")
    st.info("No predictions logged yet. Refresh the page to log the first one.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.caption(
    f"Auto-refreshes every 60s  |  "
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
    "Data: Binance BTCUSDT 1h"
)
