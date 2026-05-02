"""
Part A — 30-day rolling walk-forward backtest.

Run once:  python backtest.py
Outputs :  backtest_results.jsonl  (one prediction per line)
           Prints coverage, avg width, Winkler to stdout.

No-peek guarantee: at step i we use prices[:i] only.
Bar i's close (prices[i]) is the actual we verify against — it is NOT used to
compute the prediction; only prices[:i] (indices 0 … i-1) are seen by the model.
"""

import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import estimate_params, fetch_btc_bars, simulate_next

# ── Constants ────────────────────────────────────────────────────────────────

LOOKBACK = 168       # 1 week of hourly bars for parameter estimation
N_SIMS   = 10_000   # Monte Carlo paths per prediction
N_TEST   = 720       # 30 days of hourly predictions


# ── Scoring ──────────────────────────────────────────────────────────────────

def winkler_score(low: float, high: float, actual: float, alpha: float = 0.05) -> float:
    """
    Winkler interval score (lower = better).
    A hit costs the width; a miss costs width + a proportional penalty.
    """
    width = high - low
    if actual < low:
        return width + (2.0 / alpha) * (low - actual)
    if actual > high:
        return width + (2.0 / alpha) * (actual - high)
    return width


def evaluate(predictions: list[dict]) -> dict:
    covered  = [p["covered"] for p in predictions]
    widths   = [p["width"]   for p in predictions]
    winklers = [p["winkler"] for p in predictions]
    return {
        "coverage_95":     float(np.mean(covered)),
        "avg_width":       float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "n_predictions":   len(predictions),
    }


# ── Backtest loop ────────────────────────────────────────────────────────────

def run_backtest(prices: pd.Series, lookback: int = LOOKBACK, n_sims: int = N_SIMS) -> list[dict]:
    """
    Walk-forward backtest.

    For bar index i (0-based):
      history = prices[:i]        → everything before bar i (no peek)
      actual  = prices[i]         → the bar we're forecasting; revealed AFTER prediction

    The last bar in `prices` is never predicted (we need actual = prices[i+1] logic
    if framed differently, but here we predict bar i using prices[:i] — equivalent).
    """
    records: list[dict] = []
    rng = np.random.default_rng(42)

    # We start at `lookback` so we always have enough history
    start = lookback
    end   = len(prices)   # predict bars [start … end-1]

    for i in tqdm(range(start, end), desc="Backtesting", unit="bar"):
        # ── strictly past data only ──────────────────────────────────────────
        history = prices.iloc[:i]          # bars 0 … i-1
        actual  = float(prices.iloc[i])    # bar i (the target, not peeked during fit)

        log_ret = np.log(history / history.shift(1)).dropna()
        recent  = log_ret.iloc[-lookback:]  # rolling fixed window
        mu, sigma, nu = estimate_params(recent)
        sigma  *= 1.15   # calibration — matches predict_interval() exactly
        S0 = float(history.iloc[-1])

        simulated = simulate_next(S0, mu, sigma, nu, n_sims, rng)
        low_95  = float(np.percentile(simulated, 2.5))
        high_95 = float(np.percentile(simulated, 97.5))

        covered = int(low_95 <= actual <= high_95)
        width   = high_95 - low_95
        ws      = winkler_score(low_95, high_95, actual)

        records.append({
            "timestamp": prices.index[i].isoformat(),
            "actual":    actual,
            "low_95":    low_95,
            "high_95":   high_95,
            "covered":   covered,
            "width":     width,
            "winkler":   ws,
        })

    return records


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_needed = N_TEST + LOOKBACK
    print(f"Fetching {total_needed} BTCUSDT 1h bars from Binance...")
    t0 = time.time()
    prices = fetch_btc_bars(n_bars=total_needed)
    print(f"  Received {len(prices)} bars: {prices.index[0]} to {prices.index[-1]}")
    print(f"  Fetch took {time.time()-t0:.1f}s\n")

    if len(prices) < total_needed:
        print(f"WARNING: only got {len(prices)} bars, expected {total_needed}.")

    print(f"Running backtest ({N_TEST} test bars, {LOOKBACK}-bar rolling lookback)...")
    t0 = time.time()
    predictions = run_backtest(prices, lookback=LOOKBACK, n_sims=N_SIMS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/len(predictions)*1000:.1f} ms/bar)\n")

    with open("backtest_results.jsonl", "w") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")

    metrics = evaluate(predictions)
    sep = "-" * 42
    print(sep)
    print(f"  Predictions saved : {metrics['n_predictions']:,}")
    print(f"  Coverage 95%      : {metrics['coverage_95']:.4f}  (target ~0.95)")
    print(f"  Avg width         : ${metrics['avg_width']:>12,.2f}")
    print(f"  Mean Winkler      : ${metrics['mean_winkler_95']:>12,.2f}")
    print(sep)
    print("Done: backtest_results.jsonl written.")
