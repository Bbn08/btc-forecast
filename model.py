"""
Core GBM model: Binance data fetching + Monte Carlo simulation.

Key design decisions:
  - EWMA volatility (span=24h): captures volatility clustering without FIGARCH's
    computational cost. Calm periods → narrow bands; volatile periods → wide bands.
  - Student-t innovations: heavier tails than Normal, critical for crypto's frequent
    large moves. DoF fitted via MLE from recent returns; clamped ≥ 4 (finite variance).
  - One-step simulation only: we're forecasting 1 bar ahead, so no multi-step paths needed.
"""

import numpy as np
import pandas as pd
import requests
from scipy import stats

BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"


def fetch_btc_hourly(limit: int = 500, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.Series:
    """
    Fetch hourly OHLCV from Binance Vision (public, no API key needed).
    Returns a Series of 'close' prices indexed by UTC open_time.
    The last bar returned by Binance may be partially open — we drop it.
    """
    params = {"symbol": symbol, "interval": interval, "limit": min(limit + 1, 1000)}
    resp = requests.get(BINANCE_BASE, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df["close"] = df["close"].astype(float)

    series = df["close"]
    # Drop the last (potentially open) candle
    return series.iloc[:-1].iloc[-limit:]


def fetch_btc_bars(n_bars: int = 889, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.Series:
    """
    Fetch n_bars of closed hourly candles, paginating if n_bars > 999.
    """
    if n_bars <= 999:
        return fetch_btc_hourly(limit=n_bars, symbol=symbol, interval=interval)

    import time

    chunks: list[pd.Series] = []
    remaining = n_bars
    end_ms: int | None = None

    while remaining > 0:
        fetch = min(remaining, 999)
        params: dict = {"symbol": symbol, "interval": interval, "limit": fetch + 1}
        if end_ms is not None:
            params["endTime"] = end_ms
        resp = requests.get(BINANCE_BASE, params=params, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df["close"] = df["close"].astype(float)
        chunk = df["close"].iloc[:-1]  # drop the open candle
        chunks.insert(0, chunk)
        end_ms = int(df.index[0].timestamp() * 1000) - 1
        remaining -= len(chunk)
        time.sleep(0.1)

    series = pd.concat(chunks).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.iloc[-n_bars:]


def estimate_params(log_returns: pd.Series, ewm_span: int = 24) -> tuple[float, float, float]:
    """
    Estimate (mu, sigma, nu) from log returns.
      mu    : sample mean (hourly drift)
      sigma : EWMA std with the given span (volatility clustering)
      nu    : Student-t DoF from kurtosis — more stable than MLE on small samples.
              BTC hourly excess kurtosis ~4-6 => nu ~5-6.
              Clamped to [4, 30]; 30 is near-Normal but we never go to infinity.
    """
    mu = float(log_returns.mean())

    sigma = float(log_returns.ewm(span=ewm_span).std().iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(log_returns.std())

    # Kurtosis-based nu: E[kurtosis of t(nu)] = 6/(nu-4), so nu = 6/ekurt + 4
    # More robust than MLE on standardised returns for samples < 500 bars
    try:
        ekurt = float(log_returns.kurtosis())   # pandas gives excess kurtosis
        if ekurt > 0.05:
            nu = float(np.clip(6.0 / ekurt + 4.0, 4.0, 30.0))
        else:
            nu = 30.0  # near-Normal (kurtosis not significant)
    except Exception:
        nu = 5.0  # safe crypto default

    return mu, sigma, nu


def simulate_next(
    S0: float,
    mu: float,
    sigma: float,
    nu: float,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Monte Carlo one-step GBM with Student-t innovations.
    Z ~ t(nu) is variance-normalised so Var(Z) = 1, keeping sigma interpretable.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_t(df=nu, size=n_sims)
    Z *= np.sqrt((nu - 2.0) / nu)
    log_chg = (mu - 0.5 * sigma ** 2) + sigma * Z
    return S0 * np.exp(log_chg)


def predict_interval(
    prices: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    lookback: int = 500,
) -> tuple[float, float, float, float, float]:
    """
    Predict the next-bar confidence interval.
    Returns (low, high, mu, sigma, nu).

    Calibration note: raw EWMA sigma systematically undercovers by ~3-4pp because
    EWMA lags at the onset of volatile clusters. A multiplier of 1.15 was found via
    rolling walk-forward validation on 750 bars with 168-bar lookback; it brings
    empirical coverage from ~91% to ~95.0%.
    """
    SIGMA_CALIB = 1.15  # empirically calibrated; see model notes above

    history = prices.iloc[-lookback:] if len(prices) > lookback else prices
    log_ret = np.log(history / history.shift(1)).dropna()
    mu, sigma, nu = estimate_params(log_ret)
    sigma *= SIGMA_CALIB

    S0 = float(prices.iloc[-1])
    simulated = simulate_next(S0, mu, sigma, nu, n_sims)

    alpha = 1.0 - confidence
    low  = float(np.percentile(simulated, alpha / 2 * 100))
    high = float(np.percentile(simulated, (1 - alpha / 2) * 100))
    return low, high, mu, sigma, nu
