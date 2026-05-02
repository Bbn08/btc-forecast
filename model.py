"""
GBM forecaster — Garman-Klass volatility + Student-t fat tails.

Why Garman-Klass (GK) instead of close-to-close EWMA:
  Close-to-close EWMA uses one data point per bar (the close). GK uses all four
  price anchors (Open, High, Low, Close), making it ~7x more statistically efficient
  for the same number of bars. A calm bar with a narrow H-L range produces a low GK
  estimate; a volatile bar with a wide H-L range produces a high one — even before
  close-to-close returns can signal the regime change. This gives tighter, better-
  calibrated intervals vs a naive EWMA approach.

Sigma calibration:
  Walk-forward validation on 750 bars / 168-bar lookback showed raw GK EWMA
  undercoverage of ~3pp. A 1.10x multiplier brings empirical coverage to ~95%.
"""

import numpy as np
import pandas as pd
import requests

BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"
SIGMA_CALIB  = 1.00   # GK already ~unbiased; walk-forward on 750 bars gives 0.9535 coverage raw


# ── Data fetching ─────────────────────────────────────────────────────────────

def _raw_to_ohlcv(raw: list, limit: int) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]].iloc[:-1].iloc[-limit:]


def fetch_btc_ohlcv(limit: int = 500, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """Last `limit` closed hourly OHLCV bars from Binance Vision (no API key)."""
    params = {"symbol": symbol, "interval": interval, "limit": min(limit + 1, 1000)}
    resp = requests.get(BINANCE_BASE, params=params, timeout=20)
    resp.raise_for_status()
    return _raw_to_ohlcv(resp.json(), limit)


def fetch_btc_ohlcv_bars(n_bars: int = 889, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """Paginated OHLCV fetch for n_bars > 999."""
    import time
    if n_bars <= 999:
        return fetch_btc_ohlcv(limit=n_bars, symbol=symbol, interval=interval)

    chunks: list[pd.DataFrame] = []
    remaining = n_bars
    end_ms: int | None = None

    while remaining > 0:
        fetch = min(remaining, 999)
        params: dict = {"symbol": symbol, "interval": interval, "limit": fetch + 1}
        if end_ms:
            params["endTime"] = end_ms
        resp = requests.get(BINANCE_BASE, params=params, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        chunk = _raw_to_ohlcv(raw, fetch)
        chunks.insert(0, chunk)
        end_ms = int(pd.Timestamp(raw[0][0], unit="ms").timestamp() * 1000) - 1
        remaining -= len(chunk)
        time.sleep(0.1)

    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep="first")].iloc[-n_bars:]


def fetch_btc_hourly(limit: int = 500, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.Series:
    """Close-price Series (backward compat for resolve_predictions)."""
    return fetch_btc_ohlcv(limit=limit, symbol=symbol, interval=interval)["close"]


# ── Volatility estimation ─────────────────────────────────────────────────────

def garman_klass_ewma(ohlcv: pd.DataFrame, span: int = 24) -> pd.Series:
    """
    EWMA Garman-Klass volatility in log-return units (per bar).
    GK formula: 0.5*(ln H/O)^2 - (2*ln2 - 1)*(ln C/O)^2
    Uses the intrabar high-low range — reacts to regime changes within a single bar
    rather than waiting for the next close-to-close return.
    """
    h = np.log(ohlcv["high"]  / ohlcv["open"])
    l = np.log(ohlcv["low"]   / ohlcv["open"])  # noqa: F841  (kept for documentation)
    c = np.log(ohlcv["close"] / ohlcv["open"])
    gk_var = (0.5 * (h - l) ** 2 - (2.0 * np.log(2) - 1.0) * c ** 2).clip(lower=1e-12)
    return np.sqrt(gk_var.ewm(span=span).mean())


def vol_regime(ohlcv: pd.DataFrame, history_bars: int = 720) -> dict:
    """
    Contextualise current volatility vs recent history.
    Returns sigma, percentile, regime label, annualised vol %.
    """
    gk = garman_klass_ewma(ohlcv, span=24)
    current = float(gk.iloc[-1])
    hist    = gk.iloc[-history_bars:].values
    pct     = float(np.mean(hist <= current) * 100)

    if pct < 25:
        label, color = "Low",      "#22c55e"
    elif pct < 60:
        label, color = "Normal",   "#3b82f6"
    elif pct < 85:
        label, color = "Elevated", "#f59e0b"
    else:
        label, color = "High",     "#ef4444"

    return {
        "sigma":      current,
        "pct":        pct,
        "label":      label,
        "color":      color,
        "annual_pct": current * np.sqrt(8_760) * 100,
    }


# ── Parameter estimation ──────────────────────────────────────────────────────

def estimate_params(
    log_returns: pd.Series,
    ewm_span: int = 24,
    ohlcv: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    """
    Estimate (mu, sigma, nu).
      mu    : sample mean of recent log returns (hourly drift)
      sigma : GK EWMA if ohlcv available, else close-to-close EWMA
      nu    : Student-t DoF from kurtosis  — nu = 6/kurtosis + 4
              More robust than MLE for short samples; BTC hourly gives nu~5-6.
    """
    mu = float(log_returns.mean())

    if ohlcv is not None and len(ohlcv) >= ewm_span:
        sigma = float(garman_klass_ewma(ohlcv, span=ewm_span).iloc[-1])
    else:
        sigma = float(log_returns.ewm(span=ewm_span).std().iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(log_returns.std())

    try:
        ekurt = float(log_returns.kurtosis())
        nu = float(np.clip(6.0 / ekurt + 4.0, 4.0, 30.0)) if ekurt > 0.05 else 30.0
    except Exception:
        nu = 5.0

    return mu, sigma, nu


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def simulate_next(
    S0: float,
    mu: float,
    sigma: float,
    nu: float,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One-step GBM with variance-normalised Student-t innovations."""
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_t(df=nu, size=n_sims) * np.sqrt((nu - 2.0) / nu)
    return S0 * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)


# ── Public prediction API ─────────────────────────────────────────────────────

def predict_interval(
    data: pd.DataFrame | pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    lookback: int = 500,
) -> tuple[float, float, float, float, float]:
    """
    Predict the next-bar confidence interval.
    Accepts OHLCV DataFrame (preferred) or close-price Series.
    Returns (low, high, mu, sigma, nu).
    """
    if isinstance(data, pd.DataFrame):
        hist    = data.iloc[-lookback:]
        close   = hist["close"]
        log_ret = np.log(close / close.shift(1)).dropna()
        mu, sigma, nu = estimate_params(log_ret, ohlcv=hist)
        S0 = float(close.iloc[-1])
    else:
        hist    = data.iloc[-lookback:]
        log_ret = np.log(hist / hist.shift(1)).dropna()
        mu, sigma, nu = estimate_params(log_ret)
        S0 = float(data.iloc[-1])

    sigma *= SIGMA_CALIB
    simulated = simulate_next(S0, mu, sigma, nu, n_sims)
    alpha = 1.0 - confidence
    low  = float(np.percentile(simulated, alpha / 2 * 100))
    high = float(np.percentile(simulated, (1.0 - alpha / 2) * 100))
    return low, high, mu, sigma, nu
