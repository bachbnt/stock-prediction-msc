# stationary.py — Stationary, causal feature set for return forecasting
"""
Why this exists
---------------
The legacy pipeline scales raw price levels with MinMaxScaler and predicts the
next close. Prices are non-stationary: once the test period leaves the training
price range the scaler extrapolates and the model degrades (see the warning in
data_loader.prepare_data). Here every feature is a ratio / return / z-score, and
the target is the NEXT-DAY LOG RETURN, so the same model can be evaluated in any
period and even pooled across tickers.

Causality contract: feature values on row t use data up to and including the
close of day t only. The target on row t is log(close[t+1] / close[t]).

Features are organised in named groups so the ablation study can switch groups
on/off. See FEATURE_GROUPS.
"""
import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    'returns':    ['ret_1', 'ret_5', 'ret_10', 'ret_20'],
    'candles':    ['gap', 'intraday', 'hl_range'],
    'ma':         ['px_sma20', 'px_sma50', 'sma20_sma50', 'ema12_ema26'],
    'macd':       ['macd_n', 'macd_signal_n', 'macd_hist_n'],
    'rsi':        ['rsi_n'],
    'bollinger':  ['bb_width', 'bb_pct'],
    'volume':     ['vol_ratio', 'obv_mom'],
    'volatility': ['vol_10', 'vol_20', 'atr_n'],
    'market':     ['mkt_ret_1', 'mkt_ret_5', 'beta_60', 'rel_5'],
    'macro':      ['usdvnd_5', 'gold_5', 'oil_5'],
    'sentiment':  ['sentiment'],
    'regime':     ['reg_0', 'reg_1', 'reg_2'],
}
# Groups that are always computed from the price file alone.
CORE_GROUPS = ['returns', 'candles', 'ma', 'macd', 'rsi', 'bollinger',
               'volume', 'volatility']
OPTIONAL_GROUPS = ['market', 'macro', 'sentiment', 'regime']

FEATURE_NAMES = [c for g in FEATURE_GROUPS.values() for c in g]


def columns_for(groups) -> list:
    """Ordered feature columns for a list of group names."""
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")
    return [c for g in FEATURE_GROUPS if g in groups for c in FEATURE_GROUPS[g]]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(loss != 0, 100.0).where(~(gain.isna() | loss.isna()))


def build_features(df: pd.DataFrame,
                   market: pd.DataFrame = None,
                   macro: pd.DataFrame = None,
                   sentiment: pd.Series = None,
                   drop_na: bool = True) -> pd.DataFrame:
    """
    Build the stationary feature table.

    Args:
        df:        OHLCV with lowercase columns and a 'date' column.
        market:    optional frame with ['date', 'close'] of the VN-Index.
        macro:     optional frame with ['date', 'usdvnd', 'gold', 'oil'] levels.
        sentiment: optional Series aligned to df.index.
        drop_na:   drop warm-up rows (and the final row has y=NaN, kept so the
                   last feature row can be used for a live forecast).

    Returns a frame with: date, close, y (next-day log return), + feature columns.
    """
    d = df.sort_values('date').reset_index(drop=True).copy()
    c, o, h, l, v = d['close'], d['open'], d['high'], d['low'], d['volume']
    prev = c.shift(1)
    lc = np.log(c)

    out = pd.DataFrame({'date': d['date'], 'close': c})

    # returns
    out['ret_1'] = lc.diff()
    for k in (5, 10, 20):
        out[f'ret_{k}'] = lc.diff(k)

    # candles
    out['gap'] = np.log(o / prev)
    out['intraday'] = np.log(c / o)
    out['hl_range'] = np.log(h / l.replace(0, np.nan))

    # moving averages (price relative to average => stationary)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out['px_sma20'] = c / sma20 - 1
    out['px_sma50'] = c / sma50 - 1
    out['sma20_sma50'] = sma20 / sma50 - 1
    out['ema12_ema26'] = ema12 / ema26 - 1

    # MACD normalised by price
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out['macd_n'] = macd / c
    out['macd_signal_n'] = signal / c
    out['macd_hist_n'] = (macd - signal) / c

    # RSI centred
    out['rsi_n'] = _rsi(c) / 100.0 - 0.5

    # Bollinger
    std20 = c.rolling(20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    out['bb_width'] = (upper - lower) / sma20
    out['bb_pct'] = (c - lower) / (upper - lower).replace(0, np.nan) - 0.5

    # volume
    vol_ma = v.rolling(20).mean()
    out['vol_ratio'] = np.log((v + 1) / (vol_ma + 1))
    obv = (np.sign(out['ret_1'].fillna(0)) * v).cumsum()
    out['obv_mom'] = (obv - obv.shift(20)) / (v.rolling(20).sum() + 1)

    # volatility
    out['vol_10'] = out['ret_1'].rolling(10).std()
    out['vol_20'] = out['ret_1'].rolling(20).std()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out['atr_n'] = tr.rolling(14).mean() / c

    # market (VN-Index) — beta and relative strength
    if market is not None and len(market):
        m = market[['date', 'close']].copy()
        m['date'] = pd.to_datetime(m['date']).dt.normalize()
        m = m.drop_duplicates('date').set_index('date')['close'].sort_index()
        mk = pd.Series(m.reindex(pd.to_datetime(d['date']).dt.normalize()).values, index=d.index)
        mk = mk.ffill()
        mret = np.log(mk).diff()
        out['mkt_ret_1'] = mret
        out['mkt_ret_5'] = np.log(mk).diff(5)
        cov = out['ret_1'].rolling(60).cov(mret)
        out['beta_60'] = cov / mret.rolling(60).var().replace(0, np.nan)
        out['rel_5'] = out['ret_5'] - out['mkt_ret_5']

    # macro — lagged one day because FX / commodity quotes close at different
    # times from HOSE; using same-day values would risk look-ahead.
    if macro is not None and len(macro):
        mac = macro.copy()
        mac['date'] = pd.to_datetime(mac['date']).dt.normalize()
        mac = mac.drop_duplicates('date').set_index('date').sort_index()
        mac = mac.reindex(pd.to_datetime(d['date']).dt.normalize()).ffill()
        mac.index = d.index
        for col, name in (('usdvnd', 'usdvnd_5'), ('gold', 'gold_5'), ('oil', 'oil_5')):
            if col in mac.columns:
                out[name] = np.log(mac[col]).diff(5).shift(1)

    if sentiment is not None:
        out['sentiment'] = pd.Series(np.asarray(sentiment, dtype=float), index=d.index)

    # target: next-day log return (NaN on the last row)
    out['y'] = lc.shift(-1) - lc

    out = out.replace([np.inf, -np.inf], np.nan)
    if drop_na:
        feat_cols = [col for col in out.columns if col in FEATURE_NAMES]
        out = out.dropna(subset=feat_cols).reset_index(drop=True)
    return out


def available_groups(feat: pd.DataFrame) -> list:
    """Groups whose columns are all present in the frame."""
    return [g for g, cols in FEATURE_GROUPS.items()
            if g != 'regime' and all(c in feat.columns for c in cols)]
