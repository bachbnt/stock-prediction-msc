# features.py — Feature engineering: Technical Indicators
import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    # Replace zero losses with NaN to avoid division by zero in RSI formula
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _macd(series: pd.Series, fast=12, slow=26, sig=9):
    ema_f  = series.ewm(span=fast, adjust=False).mean()
    ema_s  = series.ewm(span=slow, adjust=False).mean()
    macd   = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def _bollinger(series: pd.Series, period=20, n_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + n_std * std, sma, sma - n_std * std


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 14 technical indicators to the DataFrame.
    All input and output column names are lowercase.
    Drops NaN rows introduced by rolling windows and resets the index.
    """
    df = df.copy()

    # Returns
    df['return']       = df['close'].pct_change()

    # Moving averages
    df['sma_20']       = df['close'].rolling(20).mean()
    df['sma_50']       = df['close'].rolling(50).mean()
    df['ema_12']       = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26']       = df['close'].ewm(span=26, adjust=False).mean()

    # RSI
    df['rsi']          = _rsi(df['close'])

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = _macd(df['close'])

    # Bollinger Bands
    bb_u, bb_m, bb_l   = _bollinger(df['close'])
    df['bb_width']     = (bb_u - bb_l) / bb_m          # band width relative to midline
    df['bb_pct']       = (df['close'] - bb_l) / (bb_u - bb_l).replace(0, np.nan)  # price position within band (0=lower, 1=upper)

    # Volume
    vol_ma             = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)  # volume relative to 20-day average

    # Candlestick ratios
    df['hl_ratio']     = df['high'] / df['low'].replace(0, np.nan)   # high-to-low range ratio
    df['co_ratio']     = df['close'] / df['open'].replace(0, np.nan) # close-to-open ratio (>1 = bullish candle)

    return df.dropna().reset_index(drop=True)
