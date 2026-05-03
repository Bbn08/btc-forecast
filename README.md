# BTC/USDT Next-Hour Forecast
### AlphaI × Polaris Challenge Submission

Predicts the 95% confidence interval for Bitcoin's next hourly close using a calibrated Geometric Brownian Motion model with Garman-Klass volatility estimation.

---

## Live Dashboard

**[btc-forecast.streamlit.app](https://bbn08-btc-forecast.streamlit.app)**

Refreshes every 60 seconds. No login or API key required.

---

## Results (Part A — 30-Day Backtest)

| Metric | Value |
|---|---|
| Coverage 95% | **0.9542** |
| Hits / Misses | **687 / 33** |
| Avg Width | **$1,146** |
| Mean Winkler Score | **$1,618** |
| Predictions | 720 (30 days × 24 bars) |

---

## Model

### Why Garman-Klass instead of close-to-close EWMA

Most GBM implementations estimate volatility from closing prices alone — one data point per bar. The **Garman-Klass estimator** uses all four price anchors (Open, High, Low, Close):

```
σ²_GK = 0.5 × (ln H/O)² − (2·ln2 − 1) × (ln C/O)²
```

This makes it roughly **7× more statistically efficient** for the same number of bars. A wide intrabar H-L range signals a volatile regime immediately, without waiting for the next close-to-close return to confirm it. The EWMA version (`span=24h`) further weights recent bars to capture volatility clustering.

### Fat tails (Student-t)

Bitcoin hourly returns have excess kurtosis of ~4–5, meaning large moves happen far more often than a Normal distribution predicts. The model draws Monte Carlo innovations from a Student-t distribution with degrees of freedom estimated from rolling kurtosis:

```
ν = 6 / excess_kurtosis + 4   (clamped to [4, 30])
```

Typical BTC hourly gives ν ≈ 5–6. The t(5) critical value at 97.5% is **2.57σ** vs **1.96σ** for Normal — 31% wider tails, which is what crypto actually needs.

### No-peek backtest

At each step `i`, only `ohlcv[:i]` is visible. `ohlcv[i].close` is the actual price revealed after the prediction is made. This is enforced structurally in `backtest.py` — there is no way for future data to leak in.

### Calibration

Walk-forward validation on 750 bars / 168-bar lookback showed raw GK vol gives ~95.3% empirical coverage with no additional multiplier. Sigma calibration factor = **1.00**.

---

## Project Structure

```
btc-forecast/
├── model.py                 # GBM model: data fetch, GK vol, MC simulation
├── backtest.py              # Part A: 720-bar walk-forward backtest
├── app.py                   # Parts B + C: Streamlit live dashboard
├── backtest_results.jsonl   # Pre-computed backtest (720 lines)
└── requirements.txt
```

---

## Run Locally

```bash
pip install -r requirements.txt
python backtest.py          # regenerate backtest_results.jsonl (~2s)
streamlit run app.py        # start dashboard at localhost:8501
```

---

## Deploy (Streamlit Community Cloud)

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `master`, main file `app.py`
4. Click Deploy — free public URL in ~2 minutes
