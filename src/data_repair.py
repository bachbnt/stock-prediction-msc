# data_repair.py — Repair price-adjustment discontinuities in the local CSVs
"""
Problem
-------
The CSVs were built by appending vnstock rows to an older file that used a
different corporate-action adjustment basis. The result is a series with
artificial jumps (e.g. FPT: -39% on 2022-06-01) and unadjusted bonus-share /
stock-dividend gaps in the older part. Price-LEVEL models tolerate this badly;
RETURN-based models are corrupted by it.

Repair strategy
---------------
1. Fetch the provider history (KBS) — it is internally consistent (every daily
   move is within the HOSE ±7% band).
2. Where the provider covers a date, use the provider OHLCV.
3. Older rows that only exist in the local CSV are multiplied by the ratio
   provider/CSV measured on the first overlapping date. That ratio is constant
   across the old segment (verified), so its relative moves are preserved.
4. Any remaining daily move beyond HOSE limit + tolerance (an unadjusted split /
   bonus issue in the CSV-only region) is neutralised by rescaling all earlier
   rows so that the move becomes ~0. Every such fix is logged.

The original file is copied to data/raw_backup/ first, so the operation is
reversible.
"""
import os
import shutil
import numpy as np
import pandas as pd

PRICE_COLS = ['open', 'high', 'low', 'close']
HOSE_LIMIT = 0.07
JUMP_TOLERANCE = 0.115     # |daily return| above this is treated as an artifact


def _to_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df.sort_values('date').drop_duplicates('date').reset_index(drop=True)


def find_jumps(df: pd.DataFrame, tol: float = JUMP_TOLERANCE) -> list:
    """Return [(date, return)] for daily moves that exceed the plausible band."""
    ret = df['close'].pct_change()
    mask = ret.abs() > tol
    return [(d, float(r)) for d, r in zip(df.loc[mask, 'date'], ret[mask])]


def neutralise_jumps(df: pd.DataFrame, tol: float = JUMP_TOLERANCE,
                     log=print) -> pd.DataFrame:
    """Rescale rows before every implausible jump so the jump becomes ~0."""
    df = df.copy()
    for _ in range(50):                      # a handful of fixes at most
        jumps = find_jumps(df, tol)
        if not jumps:
            break
        date, ret = jumps[0]
        idx = df.index[df['date'] == date][0]
        factor = df.loc[idx, 'close'] / df.loc[idx - 1, 'close']
        df.loc[:idx - 1, PRICE_COLS] = df.loc[:idx - 1, PRICE_COLS] * factor
        if 'volume' in df.columns:
            df.loc[:idx - 1, 'volume'] = df.loc[:idx - 1, 'volume'] / factor
        log(f"    neutralised jump on {date.date()} ({ret*100:+.1f}%), "
            f"earlier rows x{factor:.4f}")
    return df


def fetch_provider_history(symbol: str, start: str = '2012-01-01',
                           sources=('KBS', 'VCI')) -> pd.DataFrame:
    from vnstock import Quote
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    for src in sources:
        try:
            raw = Quote(symbol=symbol, source=src).history(
                start=start, end=end, interval='1D')
            if raw is not None and len(raw) > 0:
                raw = raw.rename(columns={'time': 'date'})
                raw = _to_lower(raw[['date', 'open', 'high', 'low', 'close', 'volume']])
                return raw
        except Exception as exc:                      # network / provider error
            print(f"    {src} failed: {str(exc)[:80]}")
    raise RuntimeError(f"Could not fetch provider history for {symbol}")


def stitch(csv_df: pd.DataFrame, provider: pd.DataFrame, log=print) -> pd.DataFrame:
    """Combine provider rows (preferred) with ratio-aligned older CSV rows."""
    csv_df, provider = _to_lower(csv_df), _to_lower(provider)
    first = provider['date'].min()
    overlap = csv_df.merge(provider[['date', 'close']], on='date',
                           suffixes=('_csv', '_prov'))
    old = csv_df[csv_df['date'] < first].copy()
    if old.empty:
        return provider
    if overlap.empty:
        log("    no overlap with provider — keeping CSV-only rows unscaled")
        ratio = 1.0
    else:
        head = overlap.head(5)
        ratio = float((head['close_prov'] / head['close_csv']).median())
        log(f"    CSV→provider ratio on first overlap: {ratio:.4f} "
            f"(applied to {len(old)} older rows)")
    old[PRICE_COLS] = old[PRICE_COLS] * ratio
    old['volume'] = old['volume'] / ratio          # keep traded value consistent
    return pd.concat([old, provider], ignore_index=True)


def _save(df: pd.DataFrame, path: str):
    out = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    out.to_csv(path, index=False)


def repair_symbol(symbol: str, data_dir: str = 'data') -> dict:
    path = os.path.join(data_dir, f'{symbol}.csv')
    backup_dir = os.path.join(data_dir, 'raw_backup')
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f'{symbol}.csv')
    if not os.path.exists(backup):                    # never overwrite the first backup
        shutil.copy2(path, backup)
    csv_df = _to_lower(pd.read_csv(backup))           # always repair from the raw file

    before = find_jumps(csv_df)
    print(f"  [{symbol}] raw rows={len(csv_df)} jumps={len(before)}")
    provider = fetch_provider_history(symbol)
    combined = stitch(csv_df, provider)
    combined = neutralise_jumps(combined)
    after = find_jumps(combined)
    _save(combined, path)
    print(f"  [{symbol}] repaired rows={len(combined)} remaining jumps={len(after)} "
          f"({combined['date'].min().date()} → {combined['date'].max().date()})")
    return dict(symbol=symbol, rows=len(combined), jumps_before=len(before),
                jumps_after=len(after))


def repair_all(symbols, data_dir: str = 'data') -> list:
    out = []
    for s in symbols:
        try:
            out.append(repair_symbol(s, data_dir))
        except Exception as exc:
            print(f"  [{s}] repair failed: {exc}")
    return out
