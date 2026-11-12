# config.py — Centralized configuration

# ── Symbols & Split ───────────────────────────────────────────────────────────
SYMBOLS        = ['FPT', 'VCB', 'VIC', 'HPG', 'VNM', 'MWG', 'PNJ', 'SSI']
SECTORS        = {            # used for sector analysis and pooled models
    'Technology': ['FPT'],
    'Banking':    ['VCB'],
    'Securities': ['SSI'],
    'Real estate': ['VIC'],
    'Materials':  ['HPG'],
    'Consumer':   ['VNM', 'MWG', 'PNJ'],
}
DATE_MILESTONE = '2024-01-01'   # train < date, test >= date

# ── Model ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 60
BATCH_SIZE       = 32
MAX_EPOCHS       = 200
PATIENCE_ES      = 20    # EarlyStopping patience
PATIENCE_LR      = 10    # ReduceLROnPlateau patience
SEED             = 42
N_PREDICT_STEPS  = 1     # 1 = single-step (stable); >1 = DIRECT multi-step (experimental)

# ── Features (lowercase — compatible with vnstock & pandas conventions) ───────
FEATURE_COLS = [
    'close', 'open', 'high', 'low', 'volume',
    'return', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_pct', 'volume_ratio',
    'hl_ratio', 'co_ratio',
]
# To enable sentiment analysis, append: FEATURE_COLS.append('sentiment')
TARGET_COL = 'close'
CLOSE_IDX  = FEATURE_COLS.index(TARGET_COL)   # = 0

# ── Trading ───────────────────────────────────────────────────────────────────
TAX_BUY         = 0.0015        # 0.15% buy fee
TAX_SELL        = 0.0015        # 0.15% sell fee
BROKERAGE_SELL  = 0.001         # 0.10% sell tax
ROUND_TRIP_COST = TAX_BUY + TAX_SELL + BROKERAGE_SELL   # 0.40%
HOSE_LIMIT      = 0.07          # HOSE daily price band (+/-7%)
SETTLEMENT_LAG  = 2             # T+2: shares bought on day t can be sold from t+2
RISK_FREE_RATE  = 0.045 / 252   # ~4.5% per year → converted to daily rate
INITIAL_CAPITAL = 10_000_000    # VND

# ── Portfolio Optimization ────────────────────────────────────────────────────
N_PORTFOLIO_SIMS = 10_000   # number of Monte Carlo simulations for efficient frontier

# ── Visualization ─────────────────────────────────────────────────────────────
COLORS = {
    'LSTM':      '#e74c3c',
    'BiLSTM':    '#3498db',
    'CNN-LSTM':  '#9b59b6',
    'ARIMA':     '#2ecc71',
    'Ensemble':  '#f39c12',
    'Portfolio': '#1abc9c',
}

# ── Research pipeline (research.py) ───────────────────────────────────────────
RESEARCH = dict(
    window          = 60,
    max_epochs      = 80,        # lighter than MAX_EPOCHS: many folds x seeds
    patience_es     = 10,
    patience_lr     = 5,
    batch_size      = 64,
    seeds           = [42, 43, 44],
    # Expanding-window walk-forward: each year below is a test block, the model
    # is trained on everything strictly before it.
    test_years      = [2021, 2022, 2023, 2024, 2025],
    mc_samples      = 30,        # MC-dropout forward passes
    quantiles       = [0.1, 0.5, 0.9],
    results_dir     = 'results',
)
