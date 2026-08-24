# data_loader.py — Load, update, and prepare data
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import FEATURE_COLS, TARGET_COL, WINDOW_SIZE, DATE_MILESTONE, N_PREDICT_STEPS


# ── Original CSV column names (uppercase) → normalized lowercase ──────────────
_CSV_RENAME = {
    'Date': 'date', 'Open': 'open', 'High': 'high',
    'Low': 'low', 'Close': 'close', 'Volume': 'volume',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Load OHLCV data for a given stock symbol.
    Priority: local CSV file → GitHub raw URL (fallback).
    Returns a DataFrame with lowercase column names and a 'date' column of datetime type.
    """
    local = os.path.join(data_dir, f'{symbol}.csv')
    if os.path.exists(local):
        df = pd.read_csv(local)
        src = f'local ({local})'
    else:
        # Fall back to fetching from the remote GitHub repository
        url = (f'https://raw.githubusercontent.com/bachbnt/'
               f'learn-stock_prediction-python/dev/data/{symbol}.csv')
        df = pd.read_csv(url)
        src = 'GitHub'

    # Normalize all column names to lowercase for consistency
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns=_CSV_RENAME)   # handle any alternate column names
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  [{symbol}] Loaded [{src}]: {len(df)} rows "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    # Fix any unit mismatch already baked into the CSV (e.g. from a previous
    # partial refresh that appended vnstock thousands-VND rows to an old CSV
    # that stored prices in a different unit). Scan every consecutive pair of
    # rows; if a jump > 10x is found, normalise the earlier segment in-place
    # and overwrite the CSV so subsequent runs are clean.
    df, fixed = _fix_internal_unit_mismatch(df, symbol)
    if fixed and os.path.exists(local):
        save_df = df.copy()
        save_df.columns = [c.capitalize() if c != 'date' else 'Date'
                           for c in save_df.columns]
        save_df.to_csv(local, index=False)
        print(f"  [{symbol}] CSV overwritten with normalised units.")

    return df


def _fix_internal_unit_mismatch(df: pd.DataFrame, symbol: str):
    """
    Scan the entire DataFrame for a large internal price discontinuity that
    indicates a unit mismatch baked into the CSV (e.g. old rows in full VND
    followed by new rows in thousands VND from a previous partial refresh).

    Strategy:
      - Compute the ratio between each consecutive close pair.
      - If any ratio > 10, treat it as a unit boundary.
      - Find the power-of-10 factor that makes that ratio < 3x (plausible move).
      - Divide all rows BEFORE the boundary by that factor so the whole series
        is in a consistent unit (matching the tail / vnstock standard).

    Returns (df, fixed: bool).
    """
    price_cols = ['open', 'high', 'low', 'close']
    closes = df['close'].values
    fixed = False

    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if b <= 0:
            continue
        ratio = a / b  # looking for old (large) → new (small) direction

        if ratio > 10:
            # Find the divisor that reduces the jump to < 3x
            factor = 1
            for f in (10, 100, 1000):
                if ratio / f < 3:
                    factor = f
                    break

            print(f"  [{symbol}] Internal unit mismatch at row {i} "
                  f"({a:,.1f} → {b:,.1f}, ratio={ratio:.0f}x). "
                  f"Normalising rows 0–{i-1} ÷{factor}.")
            df = df.copy()
            df.loc[:i - 1, price_cols] = df.loc[:i - 1, price_cols] / factor
            fixed = True
            break  # one boundary expected; stop after first fix

    return df, fixed


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UPDATE — Append new data from vnstock into the local CSV
# ═══════════════════════════════════════════════════════════════════════════════
def update_data(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Update the local CSV with the latest data from vnstock (VCI → KBS fallback).
    If no local CSV exists, fetch the full history starting from 2012.
    Returns the updated DataFrame.
    """
    try:
        from vnstock import Quote
    except ImportError:
        print("  [update_data] vnstock is not installed. Run: pip install vnstock>=4.0.0")
        return load_data(symbol, data_dir)

    local = os.path.join(data_dir, f'{symbol}.csv')
    os.makedirs(data_dir, exist_ok=True)

    # Determine the fetch start date based on the last row in the existing CSV
    if os.path.exists(local):
        existing = load_data(symbol, data_dir)
        last_date = existing['date'].max()
        # Start from the day after the last known date to avoid duplicates
        start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"  [{symbol}] Fetching from {start} to today...")
    else:
        existing = None
        start = '2012-01-01'
        print(f"  [{symbol}] No local CSV found, fetching full history from {start}...")

    end = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Skip if data is already up-to-date
    if start >= end:
        print(f"  [{symbol}] Data is already up-to-date as of {end}, skipping.")
        return existing

    # Try VCI first, then fall back to KBS if VCI fails or returns no data
    new_df = None
    for source in ('VCI', 'KBS'):
        try:
            quote  = Quote(symbol=symbol, source=source)
            raw    = quote.history(start=start, end=end, interval='1D')
            if raw is not None and len(raw) > 0:
                new_df = raw
                print(f"  [{symbol}] Fetched {len(raw)} rows from {source}.")
                break
        except Exception as e:
            print(f"  [{symbol}] {source} error: {e}")

    if new_df is None or len(new_df) == 0:
        print(f"  [{symbol}] No new data could be fetched.")
        return existing if existing is not None else load_data(symbol, data_dir)

    # Normalize vnstock column names to standard lowercase format
    new_df = new_df.rename(columns={'time': 'date'})
    new_df['date'] = pd.to_datetime(new_df['date']).dt.normalize()
    new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    # ── Unit mismatch detection ───────────────────────────────────────────────
    # Old CSVs from some data providers use different price units (e.g. ×10 or
    # full VND) while vnstock v4 returns prices in thousands VND. A real stock
    # price cannot change by 10x overnight, so any ratio > 10 at the join
    # boundary is treated as a unit mismatch rather than a genuine price move.
    # Fix: find the power-of-10 conversion factor that reduces the ratio to a
    # plausible range (< 3x), then rescale the old CSV in-place. This preserves
    # the full price history without needing to re-fetch from vnstock.
    if existing is not None and len(new_df) > 0:
        last_old  = existing['close'].iloc[-1]
        first_new = new_df['close'].iloc[0]
        ratio = max(last_old, first_new) / max(min(last_old, first_new), 1)

        if ratio > 10:
            # Try divisors 10, 100, 1000 in order; pick the first that brings
            # the boundary ratio below 3x (i.e. a plausible real price change).
            # Example: ratio=169 → 169/100=1.69x ✓ → factor=100.
            factor = 1
            for f in (10, 100, 1000):
                if ratio / f < 3:
                    factor = f
                    break

            print(f"  [{symbol}] Unit mismatch detected at join boundary "
                  f"({last_old:,.1f} → {first_new:,.1f}, ratio={ratio:.0f}x).")
            print(f"  [{symbol}] Converting old CSV prices ÷{factor} to match "
                  f"vnstock unit (thousands VND)...")

            price_cols = ['open', 'high', 'low', 'close']
            existing[price_cols] = existing[price_cols] / factor

            # Overwrite the old CSV with normalised values
            save_existing = existing.copy()
            save_existing.columns = [
                c.capitalize() if c != 'date' else 'Date'
                for c in save_existing.columns
            ]
            save_existing.to_csv(local, index=False)
            print(f"  [{symbol}] Old CSV normalised and saved.")

        elif ratio > 3:  # large but possibly real gap (e.g. trading halt, rights issue)
            print(f"  [{symbol}] WARNING: Large price gap at join boundary "
                  f"({last_old:,.0f} → {first_new:,.0f}, ratio={ratio:.1f}x). "
                  f"Check for stock split or unit mismatch in the CSV.")

    # Concatenate new data with the existing records
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Remove duplicate dates and re-sort chronologically
    combined = (combined
                .drop_duplicates(subset='date')
                .sort_values('date')
                .reset_index(drop=True))

    # Save back to CSV with capitalized column names for backward compatibility
    save_df = combined.copy()
    save_df.columns = [c.capitalize() if c != 'date' else 'Date' for c in save_df.columns]
    save_df.to_csv(local, index=False)
    print(f"  [{symbol}] Saved {len(combined)} rows → {local}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE — Build sequences for model training/evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def _make_sequences(data: np.ndarray, window: int, target_idx: int,
                    n_steps: int = 1):
    """
    Create sliding-window X/y sequences from scaled data.
    - n_steps=1  → y is a scalar (single-step prediction)
    - n_steps>1  → y is a vector [t+1, t+2, ..., t+n_steps] (multi-step prediction)
    """
    X, y = [], []
    # Include the final valid target window as well.
    max_i = len(data) - n_steps + 1
    for i in range(window, max_i):
        X.append(data[i - window:i])          # input window of length `window`
        if n_steps == 1:
            y.append(data[i, target_idx])      # single future target value
        else:
            y.append(data[i:i + n_steps, target_idx])  # multi-step target vector
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame,
                 date_milestone: str = DATE_MILESTONE,
                 window: int = WINDOW_SIZE,
                 n_steps: int = None,
                 feature_cols=None,
                 scaler: MinMaxScaler = None):
    """
    Split into train/test sets, fit-and-transform the scaler (on train only),
    and build sliding-window sequences.

    Args:
        n_steps: number of future steps to predict (None → use N_PREDICT_STEPS from config).
        feature_cols: ordered model features (None → use FEATURE_COLS from config).
        scaler: an already-fitted scaler from a matching model artifact. When
                omitted, a new scaler is fitted on training data only.

    Returns:
        X_train, y_train, X_test, y_test, scaler, df_train, df_test
        - y shape: (N,) if n_steps=1, (N, n_steps) if n_steps>1
    """
    if n_steps is None:
        n_steps = N_PREDICT_STEPS

    feature_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("feature_cols contains duplicate column names")
    if TARGET_COL not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COL}' must be present in feature_cols")
    if df is None or df.empty:
        raise ValueError("Cannot prepare data from an empty DataFrame")

    required_cols = ['date', *feature_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    feature_values = df[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        bad_cols = [col for col in feature_cols
                    if not np.isfinite(df[col].to_numpy(dtype=float)).all()]
        raise ValueError(f"Features contain NaN or infinite values: {', '.join(bad_cols)}")

    split_date = pd.Timestamp(date_milestone)
    df_train = df[df['date'] < split_date].copy()
    df_test  = df[df['date'] >= split_date].copy()

    if df_train.empty:
        raise ValueError(f"Training split is empty before {split_date.date()}")
    if df_test.empty:
        raise ValueError(f"Test split is empty on or after {split_date.date()}")
    min_train_rows = window + n_steps
    if len(df_train) < min_train_rows:
        raise ValueError(
            f"Training split has {len(df_train)} rows; at least {min_train_rows} "
            f"are required for window={window}, n_steps={n_steps}"
        )
    if len(df_test) < n_steps:
        raise ValueError(
            f"Test split has {len(df_test)} rows; at least {n_steps} "
            f"are required for n_steps={n_steps}"
        )

    target_idx = feature_cols.index(TARGET_COL)

    # Fit the scaler exclusively on training data to prevent data leakage
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_train[feature_cols].values)
    elif getattr(scaler, 'n_features_in_', None) != len(feature_cols):
        raise ValueError(
            "Artifact scaler feature count does not match the active feature set: "
            f"{getattr(scaler, 'n_features_in_', 'unknown')} != {len(feature_cols)}"
        )

    # ── Scaler range validation ────────────────────────────────────────────────
    # If test prices fall outside the training price range, the scaler will
    # extrapolate beyond [0, 1] and inverse_transform will produce garbage.
    # Detect this early and warn the user with a clear fix suggestion.
    train_close_max = df_train['close'].max()
    train_close_min = df_train['close'].min()
    test_close_max  = df_test['close'].max()
    test_close_min  = df_test['close'].min()

    if test_close_max > train_close_max * 1.15:
        print(f"\n  [WARNING] Scaler out-of-range detected!")
        print(f"    Training close range : {train_close_min:,.0f} – {train_close_max:,.0f}")
        print(f"    Test close range     : {test_close_min:,.0f} – {test_close_max:,.0f}")
        print(f"    Test max is {test_close_max/train_close_max:.1f}x the training max.")
        print(f"    FIX: Move DATE_MILESTONE forward in src/config.py so that")
        print(f"         training data covers the current price range.")
        suggestion = df_test['date'].quantile(0.5).strftime('%Y-%m-%d')
        print(f"         Suggestion: DATE_MILESTONE = '{suggestion}'\n")

    # Stack train and test arrays, then transform the entire combined set
    # using the train-fitted scaler (test data uses the same scale as training)
    combined    = np.vstack([df_train[feature_cols].values,
                              df_test[feature_cols].values])
    combined_sc = scaler.transform(combined)

    n_train  = len(df_train)
    train_sc = combined_sc[:n_train]
    # Include the last `window` rows of training data at the start of the test slice
    # so that the very first test sequence has a full lookback window
    test_sc  = combined_sc[n_train - window:]

    X_train, y_train = _make_sequences(train_sc, window, target_idx, n_steps)
    X_test,  y_test  = _make_sequences(test_sc,  window, target_idx, n_steps)

    return X_train, y_train, X_test, y_test, scaler, df_train, df_test


def inverse_close(scaled: np.ndarray, scaler: MinMaxScaler,
                  feature_cols=None) -> np.ndarray:
    """
    Inverse-transform scaled values back to the original price domain for the
    close-price column.
    Supports both 1D arrays (single-step) and 2D arrays (multi-step, shape N×n_steps).
    """
    feature_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
    if TARGET_COL not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COL}' must be present in feature_cols")
    n_scaler_features = getattr(scaler, 'n_features_in_', None)
    if n_scaler_features != len(feature_cols):
        raise ValueError(
            "Scaler feature count does not match feature_cols: "
            f"{n_scaler_features} != {len(feature_cols)}"
        )

    target_idx = feature_cols.index(TARGET_COL)
    scaled_flat = np.asarray(scaled)
    original_shape = scaled_flat.shape

    # Flatten to 1D so we can build a full-feature dummy matrix for inverse_transform
    flat = scaled_flat.flatten()
    if flat.size == 0:
        return flat

    # Detect out-of-range scaled values — these indicate the scaler is extrapolating
    # and the resulting prices will be incorrect (classic symptom: MAPE > 1000%)
    out_of_range = np.sum((flat < -0.1) | (flat > 1.1))
    if out_of_range > 0:
        pct = out_of_range / len(flat) * 100
        print(f"  [WARNING] {out_of_range} scaled values ({pct:.1f}%) are outside [0, 1]. "
              f"Inverse transform will produce unreliable prices. "
              f"Check DATE_MILESTONE in src/config.py.")

    # Build a zero-filled dummy array with the same number of feature columns,
    # then insert the scaled close values at the correct column index
    dummy = np.zeros((len(flat), len(feature_cols)))
    dummy[:, target_idx] = flat
    result = scaler.inverse_transform(dummy)[:, target_idx]

    # Restore the original 2D shape for multi-step predictions (n_steps > 1)
    # If the last dimension equals 1 (e.g., model.predict output), keep as 1D
    if len(original_shape) > 1 and original_shape[-1] > 1:
        return result.reshape(original_shape)
    return result
