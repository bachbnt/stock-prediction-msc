# regime.py — Market regime detection (Gaussian mixture on trend & volatility)
"""
A lightweight stand-in for an HMM (hmmlearn is not a dependency): a 3-component
Gaussian mixture is fitted on [20-day mean return, 20-day volatility] using
TRAINING rows only. Components are re-ordered by volatility so that
reg_0 = calm, reg_1 = normal, reg_2 = turbulent. The posterior probabilities
are used as features (group 'regime'), and the hard label is used to report
metrics per regime.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REGIME_COLS = ['reg_0', 'reg_1', 'reg_2']
REGIME_NAMES = ['calm', 'normal', 'turbulent']


def _regime_inputs(feat: pd.DataFrame) -> np.ndarray:
    mean20 = feat['ret_1'].rolling(20).mean()
    vol20 = feat['vol_20'] if 'vol_20' in feat else feat['ret_1'].rolling(20).std()
    return np.column_stack([mean20.values, vol20.values])


class RegimeModel:
    def __init__(self, n_regimes: int = 3, seed: int = 0):
        self.n = n_regimes
        self.seed = seed
        self.gmm = None
        self.order = None
        self.mu = self.sd = None

    def fit(self, feat_train: pd.DataFrame) -> 'RegimeModel':
        x = _regime_inputs(feat_train)
        x = x[np.isfinite(x).all(axis=1)]
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-12
        self.gmm = GaussianMixture(self.n, covariance_type='full',
                                   random_state=self.seed, n_init=3).fit((x - self.mu) / self.sd)
        # order components by volatility (2nd input dimension, un-standardised)
        vol_means = self.gmm.means_[:, 1] * self.sd[1] + self.mu[1]
        self.order = np.argsort(vol_means)
        return self

    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        x = _regime_inputs(feat)
        ok = np.isfinite(x).all(axis=1)
        proba = np.zeros((len(feat), self.n))
        if ok.any():
            p = self.gmm.predict_proba((x[ok] - self.mu) / self.sd)
            proba[ok] = p[:, self.order]
        # warm-up rows (no 20-day stats): assume the middle regime
        proba[~ok, self.n // 2] = 1.0
        return proba

    def add_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat.copy()
        proba = self.predict_proba(feat)
        for i, col in enumerate(REGIME_COLS[:self.n]):
            out[col] = proba[:, i]
        return out

    def labels(self, feat: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(feat).argmax(axis=1)
