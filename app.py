"""
Parts B + C — Live dashboard.
Model: GBM with Garman-Klass volatility + Student-t fat tails.
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
from model import fetch_btc_hourly, fetch_btc_ohlcv, predict_interval, vol_regime

st.set_page_config(
    page_title="BTC Forecast | AlphaI x Polaris",
    page_icon="=",
    layout="wide",
)
st_autorefresh(interval=60_000, key="btc_refresh")

BACKTEST_FILE  = "backtest_results.jsonl"
LIVE_PRED_FILE = "live_predictions.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_jsonl(path):
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


def load_backtest_metrics():
    preds = _read_jsonl(BACKTEST_FILE)
    if not preds:
        return None
    m = evaluate(preds)
    m["hits"]   = sum(p["covered"] for p in preds)
    m["misses"] = m["n_predictions"] - m["hits"]
    return m, preds


def append_live_prediction(record):
    with open(LIVE_PRED_FILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def resolve_predictions(preds, prices):
    idx = prices.index
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC")
    price_map = {ts: float(p) for ts, p in zip(idx, prices.values)}

    resolved = []
    for p in preds:
        entry  = dict(p)
        bar_ts = pd.Timestamp(p["bar_time"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        actual_ts = bar_ts + pd.Timedelta(hours=1)

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


def bar_countdown():
    now  = datetime.now(timezone.utc)
    secs = 3600 - (now.minute * 60 + now.second)
    return f"{secs // 60}m {secs % 60:02d}s"


# ── Data fetch (cached 55s) ───────────────────────────────────────────────────

@st.cache_data(ttl=55)
def get_data_and_forecast():
    ohlcv = fetch_btc_ohlcv(limit=500)
    low_95, high_95, mu, sigma, nu = predict_interval(ohlcv, n_sims=10_000, lookback=500)
    return ohlcv, low_95, high_95, mu, sigma, nu


# ── Layout ────────────────────────────────────────────────────────────────────

col_title, col_timer = st.columns([4, 1])
with col_title:
    st.title("BTC/USDT Next-Hour Forecast")
    st.caption(
        "Garman-Klass vol + Student-t fat tails + 10,000-path Monte Carlo  |  "
        "AlphaI x Polaris Challenge"
    )
with col_timer:
    st.metric("Next bar closes in", bar_countdown())

with st.spinner("Fetching OHLCV and running Monte Carlo..."):
    ohlcv, low_95, high_95, mu, sigma, nu = get_data_and_forecast()

prices        = ohlcv["close"]
current_price = float(prices.iloc[-1])
pct_change    = (current_price - float(prices.iloc[-2])) / float(prices.iloc[-2]) * 100

# Persist prediction (Part C)
append_live_prediction({
    "bar_time":      prices.index[-1].isoformat(),
    "current_price": current_price,
    "low_95":        low_95,
    "high_95":       high_95,
    "width":         high_95 - low_95,
    "visited_at":    datetime.now(timezone.utc).isoformat(),
})


# ── Part A: Backtest metrics ──────────────────────────────────────────────────

st.subheader("Part A: 30-Day Backtest Metrics (720 bars, GK vol)")

bt_result = load_backtest_metrics()
if bt_result:
    bt, bt_preds = bt_result
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Coverage 95%",       f"{bt['coverage_95']:.4f}",
              f"{bt['coverage_95']-0.95:+.4f} vs target")
    c2.metric("Hits / Misses",       f"{bt['hits']} / {bt['misses']}")
    c3.metric("Avg Width",           f"${bt['avg_width']:,.0f}")
    c4.metric("Mean Winkler Score",  f"${bt['mean_winkler_95']:,.0f}")
    c5.metric("Predictions",         f"{bt['n_predictions']:,}")
else:
    bt_preds = []
    st.info("Run `python backtest.py` and commit `backtest_results.jsonl` to populate this.")

st.divider()

# ── Part B: Current forecast ──────────────────────────────────────────────────

st.subheader("Part B: Current Forecast")

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC Price",            f"${current_price:,.2f}", f"{pct_change:+.2f}%")
m2.metric("Forecast Low  (95%)",  f"${low_95:,.2f}")
m3.metric("Forecast High (95%)",  f"${high_95:,.2f}")
m4.metric("Range Width",          f"${high_95 - low_95:,.2f}")

# Chart: last 50 bars + forecast ribbon
last_50 = prices.iloc[-50:]
ts_list = last_50.index.tolist()
next_ts = ts_list[-1] + pd.Timedelta(hours=1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts_list, y=last_50.values,
    mode="lines", name="BTC Close (1h)",
    line=dict(color="#F7931A", width=2),
))
fig.add_trace(go.Scatter(
    x=[ts_list[-1]], y=[current_price],
    mode="markers", name="Latest close",
    marker=dict(color="#F7931A", size=9),
))
fig.add_trace(go.Scatter(
    x=[ts_list[-1], next_ts, next_ts, ts_list[-1]],
    y=[high_95, high_95, low_95, low_95],
    fill="toself", fillcolor="rgba(30,120,255,0.18)",
    line=dict(color="rgba(0,0,0,0)"),
    name=f"95% CI  [{low_95:,.0f} to {high_95:,.0f}]",
))
for y_val in [low_95, high_95]:
    fig.add_trace(go.Scatter(
        x=[ts_list[-1], next_ts], y=[current_price, y_val],
        mode="lines", line=dict(color="#1E78FF", dash="dot", width=1.5),
        showlegend=False,
    ))
fig.update_layout(
    xaxis_title="Time (UTC)", yaxis_title="Price (USDT)",
    template="plotly_dark", height=420,
    margin=dict(l=60, r=20, t=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Market Conditions (unique section) ───────────────────────────────────────

st.subheader("Market Conditions")

regime        = vol_regime(ohlcv, history_bars=720)
close_24h_ago = float(prices.iloc[-25]) if len(prices) >= 25 else float(prices.iloc[0])
pct_24h       = (current_price - close_24h_ago) / close_24h_ago * 100
high_24h      = float(ohlcv["high"].iloc[-24:].max())
low_24h       = float(ohlcv["low"].iloc[-24:].min())
vol_now       = float(ohlcv["volume"].iloc[-1])
vol_avg_24h   = float(ohlcv["volume"].iloc[-24:].mean())
vol_ratio     = vol_now / vol_avg_24h if vol_avg_24h > 0 else 1.0

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("GK Vol (annualised)",  f"{regime['annual_pct']:.1f}%")
mc2.metric("Vol Percentile",       f"{regime['pct']:.0f}th",
           f"vs last 30 days")
mc3.metric("Vol Regime",           regime["label"])
mc4.metric("24h Range",            f"${low_24h:,.0f} – ${high_24h:,.0f}")
mc5.metric("Vol vs 24h Avg",       f"{vol_ratio:.2f}x",
           "above avg" if vol_ratio > 1.05 else ("below avg" if vol_ratio < 0.95 else "normal"))

# Colour-coded volatility bar
pct_int = int(regime["pct"])
st.markdown(
    f"""
    <div style="margin: 4px 0 2px 0; font-size:12px; color:#999">
        Volatility percentile vs last 30 days
    </div>
    <div style="background:#2d2d2d; border-radius:6px; height:14px; width:100%">
      <div style="background:{regime['color']}; width:{pct_int}%; height:14px;
                  border-radius:6px; transition:width 0.4s ease"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:11px;
                color:#666; margin-top:2px">
      <span>0th (calm)</span><span>50th</span><span>100th (extreme)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model details"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Hourly drift",       f"{mu:.6f}")
    d2.metric("GK sigma (raw)",     f"{sigma/1.0:.6f}")
    d3.metric("Student-t DoF",      f"{nu:.2f}")
    st.caption(
        "Volatility estimated via Garman-Klass EWMA (uses OHLC, ~7x more efficient "
        "than close-to-close). Fat tails via Student-t; degrees of freedom estimated "
        "from rolling excess kurtosis. Walk-forward calibration: coverage 0.9542 on "
        "720 bars, Winkler $1,618."
    )

st.divider()

# ── Backtest visualisation ────────────────────────────────────────────────────

if bt_preds:
    st.subheader("Backtest: 720-Bar Price vs Predicted Intervals")

    df_bt = pd.DataFrame(bt_preds)
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    df_bt = df_bt.sort_values("timestamp")
    hits_bt   = df_bt[df_bt["covered"] == 1]
    misses_bt = df_bt[df_bt["covered"] == 0]

    fig_bt = go.Figure()

    # CI ribbon
    x_rib = list(df_bt["timestamp"]) + list(df_bt["timestamp"])[::-1]
    y_rib = list(df_bt["high_95"])   + list(df_bt["low_95"])[::-1]
    fig_bt.add_trace(go.Scatter(
        x=x_rib, y=y_rib,
        fill="toself", fillcolor="rgba(30,120,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Predicted CI",
    ))

    # Actual price line
    fig_bt.add_trace(go.Scatter(
        x=df_bt["timestamp"], y=df_bt["actual"],
        mode="lines", name="Actual BTC",
        line=dict(color="#F7931A", width=1.2),
    ))

    # Miss markers
    if len(misses_bt):
        fig_bt.add_trace(go.Scatter(
            x=misses_bt["timestamp"], y=misses_bt["actual"],
            mode="markers", name=f"Misses ({len(misses_bt)})",
            marker=dict(color="#ef4444", size=6, symbol="x"),
        ))

    fig_bt.update_layout(
        xaxis_title="Date (UTC)", yaxis_title="BTC Price (USDT)",
        template="plotly_dark", height=350,
        margin=dict(l=60, r=20, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

st.divider()

# ── Part C: Live prediction history ──────────────────────────────────────────

raw = _read_jsonl(LIVE_PRED_FILE)

if raw:
    df_raw = pd.DataFrame(raw)
    df_raw["bar_time"] = pd.to_datetime(df_raw["bar_time"])
    df_raw = (
        df_raw.sort_values("visited_at")
              .drop_duplicates("bar_time", keep="last")
              .reset_index(drop=True)
    )
    resolved      = resolve_predictions(df_raw.to_dict("records"), prices)
    resolved_done = [r for r in resolved if r["result"] != "Pending"]
    hits_live     = sum(1 for r in resolved_done if r["result"] == "Hit")
    misses_live   = len(resolved_done) - hits_live
    live_cov      = hits_live / len(resolved_done) if resolved_done else None

    st.subheader(f"Part C: Live Prediction History ({len(resolved)} entries)")

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Live Coverage",   f"{live_cov:.3f}" if live_cov is not None else "N/A")
    lc2.metric("Resolved Hits",   hits_live)
    lc3.metric("Resolved Misses", misses_live)

    rows = []
    for r in sorted(resolved, key=lambda x: x["bar_time"], reverse=True):
        result = r["result"]
        rows.append({
            "Timestamp (UTC)": pd.Timestamp(r["bar_time"]).strftime("%Y-%m-%d %H:%M"),
            "BTC at Pred":     f"${r['current_price']:,.2f}",
            "Lo":              f"${r['low_95']:,.2f}",
            "Hi":              f"${r['high_95']:,.2f}",
            "Actual":          f"${r['actual']:,.2f}" if r["actual"] else "-",
            "Result":          result,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.subheader("Part C: Live Prediction History")
    st.info("Refresh the page once to log the first prediction.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.caption(
    f"Auto-refreshes every 60s  |  "
    f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
    "Data: Binance BTCUSDT 1h (Binance Vision, public)"
)
