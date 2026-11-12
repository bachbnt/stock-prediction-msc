# macro_features.py — Optional market & macro data (VN-Index, USD/VND, gold, oil)
"""
Everything here is OPTIONAL and cached under data/macro/. If a source is not
reachable the corresponding columns are simply absent and the feature groups
'market' / 'macro' are skipped by the research pipeline.

  VN-Index   : vnstock Quote('VNINDEX')       -> data/macro/VNINDEX.csv
  USD/VND    : yfinance 'VND=X'               -> data/macro/macro.csv  (col usdvnd)
  Gold       : yfinance 'GC=F' (USD/oz)       -> col gold
  Brent oil  : yfinance 'BZ=F'                -> col oil

Not included (no free stable API): SJC gold price, interbank rates, and daily
foreign net buy/sell. Drop a CSV with a 'date' column into data/macro/ and extend
`load_macro` if you obtain them.
"""
import os
import pandas as pd

MACRO_DIR = os.path.join('data', 'macro')


def _read(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
        return df.sort_values('date').reset_index(drop=True)
    return None


def load_market(refresh: bool = False, symbol: str = 'VNINDEX') -> pd.DataFrame:
    """VN-Index daily closes (cached). Returns None when unavailable."""
    os.makedirs(MACRO_DIR, exist_ok=True)
    path = os.path.join(MACRO_DIR, f'{symbol}.csv')
    cached = _read(path)
    if cached is not None and not refresh:
        return cached
    try:
        from data_repair import fetch_provider_history
        df = fetch_provider_history(symbol, start='2014-01-01')
        df[['date', 'close', 'volume']].to_csv(path, index=False)
        print(f"  [macro] {symbol}: {len(df)} rows cached")
        return df[['date', 'close', 'volume']]
    except Exception as exc:
        print(f"  [macro] {symbol} unavailable ({str(exc)[:60]})")
        return cached


def load_macro(refresh: bool = False) -> pd.DataFrame:
    """USD/VND, gold, Brent oil levels (cached). Returns None when unavailable."""
    os.makedirs(MACRO_DIR, exist_ok=True)
    path = os.path.join(MACRO_DIR, 'macro.csv')
    cached = _read(path)
    if cached is not None and not refresh:
        return cached
    try:
        import yfinance as yf
        frames = {}
        for col, ticker in (('usdvnd', 'VND=X'), ('gold', 'GC=F'), ('oil', 'BZ=F')):
            raw = yf.download(ticker, start='2014-01-01', progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            s = raw['Close']
            s = s.iloc[:, 0] if hasattr(s, 'columns') else s
            frames[col] = s
        if not frames:
            raise RuntimeError('no series downloaded')
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename_axis('date').reset_index().ffill()
        df.to_csv(path, index=False)
        print(f"  [macro] downloaded {list(frames)} ({len(df)} rows)")
        return df
    except Exception as exc:
        print(f"  [macro] yfinance data unavailable ({str(exc)[:60]})")
        return cached
