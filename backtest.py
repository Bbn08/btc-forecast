"""
Part A — 30-day walk-forward backtest with Garman-Klass volatility.

Run once:  python backtest.py
Output  :  backtest_results.jsonl  (720 predictions, one per line)

No-peek guarantee: at bar index i, only ohlcv[:i] is visible to the model.
ohlcv[i]["close"] is the actual we score against — revealed after prediction.
"""

import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import SIGMA_CALIB, estimate_params, fetch_btc_ohlcv_bars, simulate_next

LOOKBACK = 168      # 1 week of hourly bars for rolling parameter window
N_SIMS   = 10_000  # Monte Carlo paths per prediction
N_TEST   = 720      # 30 days of predictions


def winkler_score(low: float, high: float, actual: float, alpha: float = 0.05) -> float:
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


def run_backtest(ohlcv: pd.DataFrame, lookback: int = LOOKBACK, n_sims: int = N_SIMS) -> list[dict]:
    """
    Rolling walk-forward backtest using OHLCV data.
    At step i: uses ohlcv[:i] only (strict no-peek). ohlcv[i].close is the target.
    """
    records: list[dict] = []
    rng = np.random.default_rng(42)
    close = ohlcv["close"]

    for i in tqdm(range(lookback, len(ohlcv)), desc="Backtesting", unit="bar"):
        hist         = ohlcv.iloc[:i]
        actual       = float(close.iloc[i])
        recent_ohlcv = hist.iloc[-lookback:]
        recent_lr    = np.log(recent_ohlcv["close"] / recent_ohlcv["close"].shift(1)).dropna()

        mu, sigma, nu = estimate_params(recent_lr, ohlcv=recent_ohlcv)
        sigma *= SIGMA_CALIB
        S0 = float(hist["close"].iloc[-1])

        sims    = simulate_next(S0, mu, sigma, nu, n_sims, rng)
        low_95  = float(np.percentile(sims, 2.5))
        high_95 = float(np.percentile(sims, 97.5))

        covered = int(low_95 <= actual <= high_95)
        width   = high_95 - low_95
        ws      = winkler_score(low_95, high_95, actual)

        records.append({
            "timestamp": ohlcv.index[i].isoformat(),
            "actual":    actual,
            "low_95":    low_95,
            "high_95":   high_95,
            "covered":   covered,
            "width":     width,
            "winkler":   ws,
        })

    return records


if __name__ == "__main__":
    total = N_TEST + LOOKBACK
    print(f"Fetching {total} BTCUSDT 1h OHLCV bars from Binance...")
    t0 = time.time()
    ohlcv = fetch_btc_ohlcv_bars(n_bars=total)
    print(f"  Got {len(ohlcv)} bars: {ohlcv.index[0]} to {ohlcv.index[-1]}")
    print(f"  Fetch: {time.time()-t0:.1f}s\n")

    print(f"Running backtest ({N_TEST} test bars, {LOOKBACK}-bar rolling window, GK vol)...")
    t0 = time.time()
    predictions = run_backtest(ohlcv, lookback=LOOKBACK, n_sims=N_SIMS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/len(predictions)*1000:.1f} ms/bar)\n")

    with open("backtest_results.jsonl", "w") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")

    m = evaluate(predictions)
    print("-" * 44)
    print(f"  Predictions  : {m['n_predictions']:,}")
    print(f"  Coverage 95% : {m['coverage_95']:.4f}  (target ~0.95)")
    print(f"  Avg width    : ${m['avg_width']:>12,.2f}")
    print(f"  Mean Winkler : ${m['mean_winkler_95']:>12,.2f}")
    print("-" * 44)
    print("Done: backtest_results.jsonl written.")
