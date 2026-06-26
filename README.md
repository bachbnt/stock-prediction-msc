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
├── models/                  # Saved trained models (.keras)
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

# Run all 5 symbols
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

---

## Configuration

Edit `src/config.py` to change key parameters:

```python
SYMBOLS         = ['FPT', 'VCB', 'VIC', 'HPG', 'VNM']
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

## Metrics

**Error metrics:** RMSE, MAE, MAPE, Directional Accuracy (DA)

**Trading metrics** (long-only strategy, HOSE fees applied):
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
