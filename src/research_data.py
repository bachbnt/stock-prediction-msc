# research_data.py — Walk-forward folds and leakage-safe window construction
"""
Convention (one sample per trading day t):
    input  = standardised features of days t-window+1 .. t   (known at close of t)
    target = log(close[t+1] / close[t])                      (realised at t+1)

Leakage guarantees (covered by tests/test_research.py):
  * a training sample's TARGET date is strictly before the test block starts;
  * the scaler, regime model and target scale are fitted on training rows only;
  * test windows may read earlier (training) rows as inputs — that is legitimate,
    those days were already observed when the forecast was made.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stationary import FEATURE_GROUPS, columns_for
from regime import RegimeModel


@dataclass
class Fold:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(dates: pd.Series, test_years, min_train_rows: int = 500) -> list:
    """Expanding-window folds, one calendar year per test block."""
    dates = pd.to_datetime(dates)
    folds = []
    for y in test_years:
        start, end = pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y}-12-31')
        if (dates < start).sum() < min_train_rows or ((dates >= start) & (dates <= end)).sum() < 20:
            continue
        folds.append(Fold(str(y), start, end))
    return folds


def single_split_fold(dates: pd.Series, milestone: str) -> Fold:
    """One train/test split (legacy behaviour): test = everything from the milestone on."""
    dates = pd.to_datetime(dates)
    return Fold(f'>={milestone}', pd.Timestamp(milestone), dates.max())


def make_windows(x: np.ndarray, idx: np.ndarray, window: int) -> np.ndarray:
    """Stack windows ending at each index in `idx` -> (len(idx), window, n_features)."""
    return np.stack([x[i - window + 1:i + 1] for i in idx]) if len(idx) else \
        np.empty((0, window, x.shape[1]))


@dataclass
class FoldData:
    fold: Fold
    feature_cols: list
    X_train: np.ndarray
    y_train: np.ndarray            # scaled log returns (y / y_scale)
    X_test: np.ndarray
    y_test: np.ndarray             # raw log returns
    y_scale: float
    prev_close: np.ndarray         # close[t]   for each test sample
    real_close: np.ndarray         # close[t+1]
    dates: pd.Series               # date of close[t+1] (the forecast target date)
    vol: np.ndarray                # trailing 20d volatility at t (for sizing)
    regime: np.ndarray             # hard regime label at t
    scaler: StandardScaler
    train_target_dates: pd.Series  # target dates of training samples (leak checks)
    flat_train: np.ndarray         # last-row features of training samples (tabular models)
    flat_test: np.ndarray


def prepare_fold(feat: pd.DataFrame, groups, fold: Fold, window: int = 60,
                 seed: int = 0) -> FoldData:
    """
    Build train/test tensors for one fold.

    `feat` is the output of stationary.build_features (+ optional sentiment).
    `groups` is the list of feature groups to use ('regime' adds GMM posteriors
    fitted on training rows only).
    """
    feat = feat.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(feat['date'])
    a_idx = np.flatnonzero(dates.values >= np.datetime64(fold.test_start))
    if len(a_idx) == 0:
        raise ValueError("test block starts after the last available date")
    a = int(a_idx[0])                                    # first index in the test block
    b = int(np.flatnonzero(dates.values <= np.datetime64(fold.test_end))[-1])
    n = len(feat)
    if a < window + 50:
        raise ValueError(f"not enough training rows before {fold.test_start.date()}")

    regime_model = None
    if 'regime' in groups:
        regime_model = RegimeModel(seed=seed).fit(feat.iloc[:a])
        feat = regime_model.add_features(feat)
    cols = columns_for(groups)
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise ValueError(f"feature columns missing from frame: {missing}")

    values = feat[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in cols if not np.isfinite(feat[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"non-finite values in features: {bad}")

    scaler = StandardScaler().fit(values[:a])            # training rows only
    xs = scaler.transform(values)
    xs = np.clip(xs, -8, 8)                              # tame extreme outliers

    y = feat['y'].to_numpy(dtype=float)
    close = feat['close'].to_numpy(dtype=float)

    # training samples: t in [window-1, a-2]  => target index t+1 <= a-1
    tr_idx = np.arange(window - 1, a - 1)
    tr_idx = tr_idx[np.isfinite(y[tr_idx])]
    # test samples: t in [a-1, b-1]  => target index t+1 in [a, b]
    te_idx = np.arange(a - 1, min(b, n - 1))
    te_idx = te_idx[np.isfinite(y[te_idx])]

    y_scale = float(np.std(y[tr_idx]) + 1e-12)
    vol = feat['vol_20'].to_numpy(dtype=float) if 'vol_20' in feat else \
        pd.Series(feat['ret_1']).rolling(20).std().bfill().to_numpy()
    reg = (regime_model.labels(feat) if regime_model is not None
           else np.zeros(n, dtype=int))

    return FoldData(
        fold=fold, feature_cols=cols,
        X_train=make_windows(xs, tr_idx, window), y_train=y[tr_idx] / y_scale,
        X_test=make_windows(xs, te_idx, window), y_test=y[te_idx], y_scale=y_scale,
        prev_close=close[te_idx], real_close=close[te_idx + 1],
        dates=dates.iloc[te_idx + 1].reset_index(drop=True),
        vol=vol[te_idx], regime=reg[te_idx], scaler=scaler,
        train_target_dates=dates.iloc[tr_idx + 1].reset_index(drop=True),
        flat_train=xs[tr_idx], flat_test=xs[te_idx],
    )


def tabular_features(xs_rows: np.ndarray, X_windows: np.ndarray, lags=(1, 2, 3, 5, 10)) -> np.ndarray:
    """Flatten a window into [current row | a few lagged rows | window mean] for tree models."""
    parts = [xs_rows]
    for k in lags:
        if k < X_windows.shape[1]:
            parts.append(X_windows[:, -1 - k, :])
    parts.append(X_windows.mean(axis=1))
    return np.concatenate(parts, axis=1)
