# config.py — Cấu hình tập trung
import math

# ── Symbols & Split ───────────────────────────────────────────────────────────
SYMBOLS        = ['FPT', 'VCB', 'VIC', 'HPG', 'VNM']
DATE_MILESTONE = '2021-01-01'   # train < date, test >= date

# ── Model ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 60
BATCH_SIZE  = 32
MAX_EPOCHS  = 200
PATIENCE_ES = 20   # EarlyStopping patience
PATIENCE_LR = 10   # ReduceLROnPlateau patience
SEED        = 42

# ── Features (lowercase — chuẩn vnstock & pandas) ────────────────────────────
FEATURE_COLS = [
    'close', 'open', 'high', 'low', 'volume',
    'return', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_pct', 'volume_ratio',
    'hl_ratio', 'co_ratio',
]
TARGET_COL = 'close'
CLOSE_IDX  = FEATURE_COLS.index(TARGET_COL)   # = 0

# ── Trading ───────────────────────────────────────────────────────────────────
TAX_BUY        = 0.0015        # 0.15% mua
TAX_SELL       = 0.0015        # 0.15% bán
BROKERAGE_SELL = 0.001         # 0.10% phí môi giới
RISK_FREE_RATE = 0.045 / 252   # ~4.5%/năm → ngày

# ── Visualization ─────────────────────────────────────────────────────────────
COLORS = {
    'LSTM':     '#e74c3c',
    'BiLSTM':   '#3498db',
    'CNN-LSTM': '#9b59b6',
    'ARIMA':    '#2ecc71',
    'Ensemble': '#f39c12',
}
