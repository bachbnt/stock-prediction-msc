# Vietnamese Stock Price Prediction

MSc thesis project — predicting closing prices for Vietnamese stocks (HOSE) using LSTM-based deep learning models.

---

## Models

| Model | Description |
|-------|-------------|
| LSTM | 2-layer LSTM + BatchNorm + L2 regularization |
| BiLSTM | Bidirectional LSTM — learns both forward and backward patterns |
| CNN-LSTM | Conv1D feature extraction → LSTM sequence learning |
| ARIMA(5,1,0) | Statistical baseline (walk-forward) |
| Ensemble | Dynamic weighted average of 3 DL models (weight ∝ 1/val_loss) |

**19 input features:** OHLCV + Return, SMA(20/50), EMA(12/26), RSI(14), MACD, Bollinger Bands, Volume Ratio, HL Ratio, CO Ratio

---

## Output

Each symbol produces a 6-panel PNG chart:

1. **Price prediction** — actual vs predicted on test set
2. **Portfolio value** — trading strategy vs buy & hold
3. **Drawdown** — max loss from peak
4. **MAPE & Directional Accuracy** — error metrics per model
5. **Return & Sharpe Ratio** — trading performance per model
6. **Future forecast** — predicted prices for next N trading days

Optional: `portfolio_optimization.png` — Markowitz efficient frontier + equity curves.

---

## Project Structure

```
stock-prediction-msc/
├── main.py                  # Entry point
├── requirements.txt
├── PLAN.md                  # Roadmap & extension ideas
│
├── src/
│   ├── config.py            # All constants (symbols, features, hyperparams)
│   ├── data_loader.py       # Load CSV, update from vnstock, prepare sequences
│   ├── features.py          # Technical indicator calculation
│   ├── models.py            # LSTM, BiLSTM, CNN-LSTM, ensemble
│   ├── evaluate.py          # Error & trading metrics
│   ├── visualize.py         # 6-panel chart generation
│   ├── portfolio.py         # Markowitz MVO optimization
│   └── sentiment/           # News sentiment pipeline (PhoBERT)
│       ├── crawler.py       # Google News + VnExpress scraper
│       ├── scorer.py        # PhoBERT sentiment scoring
│       └── pipeline.py      # Daily aggregation + feature merge
│
├── scripts/
│   ├── start.command        # Double-click to train & run
│   ├── plot-only.command    # Re-plot without retraining
│   ├── refresh-data.command # Update data only
│   ├── setup.command        # First-time environment setup
│   └── install-python.sh   # Install pyenv + Python 3.13
│
├── data/                    # CSV price data (auto-updated via vnstock)
├── models/                  # Versioned models, scaler, and metadata
└── plot/                    # Output charts (.png)
```

---

## Setup

**Requires Python 3.10–3.13.** TensorFlow does not support Python 3.14+.

### First time

```bash
# 1. Install correct Python version (skip if already on 3.10–3.13)
bash scripts/install-python.sh

# 2. Create virtual environment & install dependencies
bash scripts/setup.command   # or double-click the file
```

### macOS — double-click workflow

| File | Action |
|------|--------|
| `scripts/setup.command` | First-time setup |
| `scripts/refresh-data.command` | Update market data from vnstock |
| `scripts/start.command` | Train models + generate charts |
| `scripts/plot-only.command` | Re-generate charts (no retraining) |

> First run: right-click → Open (macOS Gatekeeper). Double-click works after that.

---

## Usage

```bash
source venv/bin/activate

# Run all 8 symbols
python main.py

# Run specific symbols
python main.py --symbols FPT VCB

# Update data from vnstock (no training)
python main.py --data-only

# Re-plot using saved models (no retraining)
python main.py --plot-only

# Forecast next N trading days (iterative auto-regressive, default 1)
python main.py --steps 10

# Include Markowitz portfolio optimization
python main.py --portfolio

# Include news sentiment feature
python main.py --sentiment

# Use fine-tuned PhoBERT for sentiment
python main.py --sentiment --sentiment-model models/phobert_sentiment

# Combine flags
python main.py --refresh --steps 10 --portfolio --sentiment
```

Each training configuration is stored separately under
`models/<symbol>/w<window>_s<steps>_f<features>_<hash>/`. The directory contains
the three `.keras` models, `scaler.joblib`, and `metadata.json` with the feature
order, split date, training date range, and validation losses. Legacy model
files without metadata are left untouched and are not loaded by `--plot-only`.

---

## Configuration

Edit `src/config.py` to change key parameters:

```python
SYMBOLS         = ['FPT', 'VCB', 'VIC', 'HPG', 'VNM', 'MWG', 'PNJ', 'SSI']
DATE_MILESTONE  = '2024-01-01'   # train/test split date — keep recent enough
                                 # so scaler covers the current price range
WINDOW_SIZE     = 60             # lookback window (trading days)
N_PREDICT_STEPS = 1              # forecast horizon: 1 = single-step (stable)
                                 # >1 = DIRECT multi-step (experimental)
MAX_EPOCHS      = 200
```

---

## Data

Price data covers **2015–present** (auto-updated via [vnstock](https://github.com/thinh-vu/vnstock) v4.x).

CSV files stored in `data/<SYMBOL>.csv` with columns: `date, open, high, low, close, volume`.

To update:
```bash
python main.py --data-only
# or double-click: scripts/refresh-data.command
```

**Unit mismatch auto-fix:** older CSV files from some providers store prices in a different unit than vnstock v4 (thousands VND). `load_data()` automatically detects any large internal price jump (ratio > 10x) and normalises the earlier segment in-place, then overwrites the CSV. No manual intervention needed.

---

## Sentiment Analysis (optional)

Requires additional setup:

```bash
# 1. Crawl news & cache locally
python main.py --sentiment

# 2. Label ~500 articles manually using Label Studio
#    Save to: data/labels/<SYMBOL>_labeled.csv  (columns: text, label)

# 3. Fine-tune PhoBERT (requires GPU or Apple Silicon MPS, ~30 min)
python -c "
from src.sentiment.scorer import fine_tune_phobert
fine_tune_phobert('data/labels/FPT_labeled.csv', 'models/phobert_sentiment')
"

# 4. Run with fine-tuned model
python main.py --sentiment --sentiment-model models/phobert_sentiment
```

Without fine-tuning, keyword-based scoring is used as a fallback.

---

## Research pipeline (`research.py`)

`main.py` is the original single-split demo. `research.py` is the evaluation
framework for the thesis. It fixes the main weaknesses of a price-level,
single-split setup:

| Issue in the original setup | What `research.py` does |
|---|---|
| Predicts the raw `close` (non-stationary, MinMax-scaled) | Predicts the **next-day log return** from stationary, scale-free features (`src/stationary.py`) |
| One train/test split | **Expanding-window walk-forward**, one test year per fold (2021–2025) |
| One run per model | **Multi-seed** (default 3): tables report mean ± std over seeds |
| No trivial baseline | **Naive** (random walk), Drift, ARIMA, gradient boosting |
| No significance testing | **Diebold-Mariano** (HLN-corrected), binomial test on directional accuracy, OOS R² vs random walk |
| Signal = "prediction t+1 > prediction t", no costs | Signal = forecast vs **today's close**, must clear the 0.40% round-trip cost, **T+2 settlement**, ±7% HOSE band, optional volatility-scaled position sizing |
| Point forecasts only | Probabilistic: **MC-dropout**, **quantile LSTM**, **TFT-lite** (80% intervals, coverage reported) and a direction classifier |

```bash
python research.py --fast --symbols FPT --models LSTM GRU Naive   # smoke test (~1 min)
python research.py                          # full grid: 8 symbols x 5 folds x 3 seeds x 14 models
python research.py --ablation               # drop one feature group at a time
python research.py --pooled                 # one network trained on all tickers
python research.py --tune --symbols FPT     # Optuna search (train data only) -> results/tuning/
python research.py --hparams results/tuning # reuse the tuned parameters
python research.py --rl                     # fitted-Q-iteration RL trading agent
python research.py --portfolio              # Markowitz / Black-Litterman / risk parity, with costs
python research.py --report                 # results/REPORT.md + tables + figures
```

Runs are cached per `(symbol, experiment)` in `results/` — re-running resumes;
use `--no-resume` to recompute. A full grid takes hours on CPU; start with
`--fast` or restrict `--symbols`, `--models`, `--seeds`, `--test-years`.

**Models:** LSTM, BiLSTM, CNN-LSTM, GRU, TCN, Transformer, LSTM-MC, LSTM-Q,
TFT-lite, LSTM-Dir, Ensemble (inverse-validation-loss weights); baselines Naive,
Drift, ARIMA(5,1,0) (rolled forward without re-fitting), HistGB.
*TFT-lite is a Keras approximation of the Temporal Fusion Transformer (variable
selection + LSTM + causal attention + quantile heads), not the
`pytorch-forecasting` implementation.*

**Feature groups** (ablation unit): returns, candles, ma, macd, rsi, bollinger,
volume (incl. OBV), volatility (incl. ATR), market (VN-Index return, beta, relative
strength), macro (USD/VND, gold, Brent — lagged one day), sentiment, regime
(3-state Gaussian-mixture posteriors fitted on training rows only).

**Leakage safeguards** (unit-tested in `tests/test_research.py`): features use only
data up to day *t*; the scaler, regime model and target scale are fitted on
training rows; training targets end strictly before the test block starts;
changing test-period data leaves training tensors unchanged.

**Not implemented / limits:** SJC gold, interbank rates and foreign net-flow have
no stable free API (see `src/macro_features.py` to plug them in). A proper HMM,
pytorch-forecasting TFT and stable-baselines3 agents are not used — the Gaussian
mixture, TFT-lite and fitted-Q agent are lightweight stand-ins with no extra
dependencies. PhoBERT fine-tuning still needs hand-labelled data
(`python -m sentiment.labeling export FPT 600`).

### Data repair

Old CSVs mixed two price-adjustment bases (e.g. FPT jumped −39% on 2022-06-01) plus
unadjusted bonus-share gaps. `python main.py --repair-data` stitches the CSVs onto
the provider history and neutralises any move beyond the ±7% band; originals are
kept in `data/raw_backup/`. Re-run it after a `--refresh` if new corporate actions
have occurred.

---

## Metrics

**Error metrics:** RMSE, MAE, MAPE, Directional Accuracy (DA)

**Trading metrics** (`main.py` now uses the edge-based backtest `evaluate.backtest_edge`: forecast vs today's close, 0.40% cost hurdle, T+2, ±7% band; the legacy rule is kept as `compute_trading_metrics`):
- Total Return vs Buy & Hold
- Sharpe Ratio (risk-free rate: 4.5%/year)
- Max Drawdown & Calmar Ratio

**HOSE transaction costs:** Buy tax 0.15% + Sell tax 0.15% + Brokerage 0.10%

---

## Requirements

```
tensorflow >= 2.13   # Python 3.10–3.13 only
scikit-learn, statsmodels, pandas, numpy, matplotlib
scipy                # portfolio optimization
vnstock >= 4.0       # market data
pygooglenews, beautifulsoup4  # news crawling
# optional: transformers, torch  (PhoBERT fine-tuning)
```

**Apple Silicon (M-series):** `setup.sh` pins TensorFlow to `2.18.x` on arm64. `tensorflow-metal` is no longer installed — it is incompatible with TF 2.19+ and largely unmaintained as of 2026. TF 2.18 runs on CPU via the standard package with no extra steps.
