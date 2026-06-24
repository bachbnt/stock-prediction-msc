# data_loader.py — Load, cập nhật và chuẩn bị dữ liệu
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import FEATURE_COLS, TARGET_COL, CLOSE_IDX, WINDOW_SIZE, DATE_MILESTONE, N_PREDICT_STEPS


# ── Cột CSV gốc (uppercase) → lowercase chuẩn ────────────────────────────────
_CSV_RENAME = {
    'Date': 'date', 'Open': 'open', 'High': 'high',
    'Low': 'low', 'Close': 'close', 'Volume': 'volume',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Load dữ liệu OHLCV cho một mã.
    Ưu tiên: file CSV local → GitHub raw URL (fallback).
    Trả về DataFrame với cột lowercase và cột 'date' kiểu datetime.
    """
    local = os.path.join(data_dir, f'{symbol}.csv')
    if os.path.exists(local):
        df = pd.read_csv(local)
        src = f'local ({local})'
    else:
        url = (f'https://raw.githubusercontent.com/bachbnt/'
               f'learn-stock_prediction-python/dev/data/{symbol}.csv')
        df = pd.read_csv(url)
        src = 'GitHub'

    # Chuẩn hoá tên cột về lowercase
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns=_CSV_RENAME)   # xử lý nếu cột có tên khác
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  [{symbol}] Loaded [{src}]: {len(df)} rows "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UPDATE — Nối dữ liệu mới từ vnstock vào CSV local
# ═══════════════════════════════════════════════════════════════════════════════
def update_data(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Cập nhật CSV local bằng dữ liệu mới nhất từ vnstock (VCI → KBS).
    Nếu CSV chưa tồn tại, tải toàn bộ lịch sử từ 2012.
    Trả về DataFrame đã được cập nhật.
    """
    try:
        from vnstock import Quote
    except ImportError:
        print("  [update_data] vnstock chưa được cài. Chạy: pip install vnstock>=4.0.0")
        return load_data(symbol, data_dir)

    local = os.path.join(data_dir, f'{symbol}.csv')
    os.makedirs(data_dir, exist_ok=True)

    # Xác định ngày bắt đầu fetch
    if os.path.exists(local):
        existing = load_data(symbol, data_dir)
        last_date = existing['date'].max()
        start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"  [{symbol}] Đang fetch từ {start} đến nay...")
    else:
        existing = None
        start = '2012-01-01'
        print(f"  [{symbol}] Không có CSV local, fetch toàn bộ từ {start}...")

    end = pd.Timestamp.today().strftime('%Y-%m-%d')

    if start >= end:
        print(f"  [{symbol}] Dữ liệu đã cập nhật đến {end}, bỏ qua.")
        return existing

    # Thử VCI trước, fallback KBS
    new_df = None
    for source in ('VCI', 'KBS'):
        try:
            quote  = Quote(symbol=symbol, source=source)
            raw    = quote.history(start=start, end=end, interval='1D')
            if raw is not None and len(raw) > 0:
                new_df = raw
                print(f"  [{symbol}] Fetched {len(raw)} rows từ {source}.")
                break
        except Exception as e:
            print(f"  [{symbol}] {source} lỗi: {e}")

    if new_df is None or len(new_df) == 0:
        print(f"  [{symbol}] Không fetch được dữ liệu mới.")
        return existing if existing is not None else load_data(symbol, data_dir)

    # Chuẩn hoá cột vnstock → lowercase chuẩn
    new_df = new_df.rename(columns={'time': 'date'})
    new_df['date'] = pd.to_datetime(new_df['date']).dt.normalize()
    new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Nối với dữ liệu cũ
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = (combined
                .drop_duplicates(subset='date')
                .sort_values('date')
                .reset_index(drop=True))

    # Lưu lại CSV (với tên cột capitalize để tương thích ngược)
    save_df = combined.copy()
    save_df.columns = [c.capitalize() if c != 'date' else 'Date' for c in save_df.columns]
    save_df.to_csv(local, index=False)
    print(f"  [{symbol}] Đã lưu {len(combined)} rows → {local}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE — Tạo sequences cho model
# ═══════════════════════════════════════════════════════════════════════════════
def _make_sequences(data: np.ndarray, window: int, target_idx: int,
                    n_steps: int = 1):
    """
    Tạo X/y sequences.
    - n_steps=1  → y là scalar (single-step)
    - n_steps>1  → y là vector [t+1, t+2, ..., t+n_steps] (multi-step)
    """
    X, y = [], []
    max_i = len(data) - n_steps
    for i in range(window, max_i):
        X.append(data[i - window:i])
        if n_steps == 1:
            y.append(data[i, target_idx])
        else:
            y.append(data[i:i + n_steps, target_idx])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame,
                 date_milestone: str = DATE_MILESTONE,
                 window: int = WINDOW_SIZE,
                 n_steps: int = None):
    """
    Split train/test, scale (fit chỉ trên train), tạo sequences.

    Args:
        n_steps: số bước dự đoán (None → dùng N_PREDICT_STEPS từ config).

    Returns:
        X_train, y_train, X_test, y_test, scaler, df_train, df_test
        - y shape: (N,) nếu n_steps=1, (N, n_steps) nếu n_steps>1
    """
    if n_steps is None:
        n_steps = N_PREDICT_STEPS

    df_train = df[df['date'] < date_milestone].copy()
    df_test  = df[df['date'] >= date_milestone].copy()

    # Scaler fit chỉ trên training → tránh data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[FEATURE_COLS].values)

    combined    = np.vstack([df_train[FEATURE_COLS].values,
                              df_test[FEATURE_COLS].values])
    combined_sc = scaler.transform(combined)

    n_train  = len(df_train)
    train_sc = combined_sc[:n_train]
    test_sc  = combined_sc[n_train - window:]

    X_train, y_train = _make_sequences(train_sc, window, CLOSE_IDX, n_steps)
    X_test,  y_test  = _make_sequences(test_sc,  window, CLOSE_IDX, n_steps)

    return X_train, y_train, X_test, y_test, scaler, df_train, df_test


def inverse_close(scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse transform cho cột close (index CLOSE_IDX).
    Hỗ trợ cả 1D (single-step) và 2D (multi-step, shape N×n_steps).
    """
    scaled_flat = np.array(scaled)
    original_shape = scaled_flat.shape

    # Flatten về 1D
    flat = scaled_flat.flatten()
    dummy = np.zeros((len(flat), len(FEATURE_COLS)))
    dummy[:, CLOSE_IDX] = flat
    result = scaler.inverse_transform(dummy)[:, CLOSE_IDX]

    # Reshape lại nếu multi-step
    if len(original_shape) > 1:
        return result.reshape(original_shape)
    return result
