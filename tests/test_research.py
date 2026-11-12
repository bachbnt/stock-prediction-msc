import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data_repair import find_jumps, neutralise_jumps, stitch
from evaluate import backtest_edge
from portfolio import (black_litterman_returns, max_utility_weights,
                       min_variance_weights, risk_parity_weights)
from regime import RegimeModel
from research_data import prepare_fold, walk_forward_folds, Fold
from stationary import FEATURE_GROUPS, build_features, columns_for
from stats_tests import dm_test, directional_accuracy_test, oos_r2


def synthetic_prices(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame(dict(date=pd.bdate_range('2016-01-01', periods=n),
                             open=open_, high=high, low=low, close=close, volume=vol))


class FeatureCausalityTests(unittest.TestCase):
    """Features at row t must not change when future rows are removed."""

    def test_features_are_causal(self):
        df = synthetic_prices()
        full = build_features(df)
        cut = 1200
        part = build_features(df.iloc[:cut])
        cols = [c for g in FEATURE_GROUPS.values() for c in g if c in full.columns]
        merged = full.merge(part, on='date', suffixes=('_f', '_p'))
        for c in cols:
            np.testing.assert_allclose(merged[f'{c}_f'], merged[f'{c}_p'], rtol=1e-9, atol=1e-12,
                                       err_msg=f'feature {c} depends on future data')

    def test_target_is_next_day_log_return(self):
        df = synthetic_prices(400)
        f = build_features(df)
        merged = f.merge(df[['date', 'close']].rename(columns={'close': 'c'}), on='date')
        nxt = np.log(df['close'].shift(-1) / df['close'])
        expected = df[['date']].assign(e=nxt).merge(f[['date', 'y']], on='date')
        np.testing.assert_allclose(expected['e'].iloc[:-1], expected['y'].iloc[:-1])

    def test_features_are_scale_free(self):
        df = synthetic_prices(600)
        a = build_features(df)
        scaled = df.copy()
        scaled[['open', 'high', 'low', 'close']] *= 1000
        b = build_features(scaled)
        for c in ('ret_1', 'px_sma20', 'macd_n', 'bb_pct', 'atr_n', 'rsi_n'):
            np.testing.assert_allclose(a[c], b[c], rtol=1e-6, atol=1e-9)


class FoldLeakageTests(unittest.TestCase):
    def setUp(self):
        self.feat = build_features(synthetic_prices(1500))
        self.groups = ['returns', 'candles', 'ma', 'volatility']

    def test_training_targets_precede_test_block(self):
        fold = Fold('t', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
        fd = prepare_fold(self.feat, self.groups, fold, window=30)
        self.assertTrue((fd.train_target_dates < fold.test_start).all())
        self.assertTrue((fd.dates >= fold.test_start).all())
        self.assertTrue((fd.dates <= fold.test_end).all())

    def test_scaler_fitted_on_training_rows_only(self):
        fold = Fold('t', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
        fd = prepare_fold(self.feat, self.groups, fold, window=30)
        cols = columns_for(self.groups)
        train_rows = self.feat[self.feat['date'] < fold.test_start][cols]
        np.testing.assert_allclose(fd.scaler.mean_, train_rows.mean().values, rtol=1e-9)

    def test_altering_test_data_does_not_change_training_tensors(self):
        fold = Fold('t', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
        a = prepare_fold(self.feat, self.groups, fold, window=30)
        feat2 = self.feat.copy()
        mask = feat2['date'] >= fold.test_start
        feat2.loc[mask, 'ret_1'] *= 5
        feat2.loc[mask, 'y'] *= 5
        b = prepare_fold(feat2, self.groups, fold, window=30)
        np.testing.assert_allclose(a.X_train, b.X_train)
        np.testing.assert_allclose(a.y_train, b.y_train)

    def test_target_alignment(self):
        fold = Fold('t', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
        fd = prepare_fold(self.feat, self.groups, fold, window=30)
        np.testing.assert_allclose(np.log(fd.real_close / fd.prev_close), fd.y_test)

    def test_regime_model_fit_on_train_only(self):
        fold = Fold('t', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
        fd = prepare_fold(self.feat, self.groups + ['regime'], fold, window=30)
        self.assertEqual(fd.X_train.shape[2], len(columns_for(self.groups + ['regime'])))
        self.assertTrue(set(np.unique(fd.regime)) <= {0, 1, 2})

    def test_walk_forward_folds_expand(self):
        folds = walk_forward_folds(self.feat['date'], [2018, 2019, 2020, 2021])
        self.assertGreaterEqual(len(folds), 2)
        starts = [f.test_start for f in folds]
        self.assertEqual(starts, sorted(starts))


class StatsTests(unittest.TestCase):
    def test_dm_detects_better_model(self):
        rng = np.random.default_rng(1)
        e_base = rng.normal(0, 1.0, 500)
        e_good = rng.normal(0, 0.6, 500)
        r = dm_test(e_good, e_base)
        self.assertLess(r['stat'], 0)
        self.assertLess(r['p_value'], 0.01)
        self.assertTrue(r['model_better'])

    def test_dm_no_difference(self):
        rng = np.random.default_rng(2)
        r = dm_test(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        self.assertGreater(r['p_value'], 0.05)

    def test_dm_identical_errors(self):
        e = np.random.default_rng(3).normal(size=100)
        r = dm_test(e, e)
        self.assertEqual(r['p_value'], 1.0)

    def test_da_ignores_zero_forecasts(self):
        a = np.array([0.01, -0.02, 0.03])
        r = directional_accuracy_test(a, np.zeros(3))
        self.assertEqual(r['n'], 0)
        self.assertTrue(np.isnan(r['da']))

    def test_oos_r2_zero_forecast_is_zero(self):
        a = np.random.default_rng(0).normal(size=50)
        self.assertEqual(oos_r2(a, np.zeros(50)), 0.0)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.r = rng.normal(0.0005, 0.015, 300)
        self.P = 100 * np.cumprod(np.r_[1, 1 + self.r])

    def test_no_signal_means_no_trades(self):
        b = backtest_edge(self.P, np.zeros(300), verbose=False)
        self.assertEqual(b['n_trades'], 0)
        self.assertAlmostEqual(b['total_return'], 0.0)

    def test_signal_below_cost_threshold_is_ignored(self):
        b = backtest_edge(self.P, np.full(300, 0.003), verbose=False)   # < 0.4% round trip
        self.assertEqual(b['n_trades'], 0)

    def test_costs_reduce_returns(self):
        sig = np.full(300, 0.02)
        b = backtest_edge(self.P, sig, verbose=False)
        gross = self.P[-1] / self.P[0] - 1
        self.assertLess(b['total_return'], gross)

    def test_t_plus_2_blocks_immediate_sale(self):
        # buy on day 0, strongly negative signal on day 1: cannot sell until day 2
        P = np.array([100.0, 90.0, 80.0, 80.0])
        sig = np.array([0.02, -0.05, -0.05])
        b = backtest_edge(P, sig, verbose=False)
        self.assertLess(b['port_vals'][2], b['port_vals'][1])   # still holding at day 1 -> 2 drop
        b0 = backtest_edge(P, sig, settlement_lag=0, verbose=False)
        self.assertGreater(b0['port_vals'][2], b['port_vals'][2])

    def test_forecast_clipped_to_price_band(self):
        a = backtest_edge(self.P, np.full(300, 0.5), verbose=False)
        b = backtest_edge(self.P, np.full(300, 0.07), verbose=False)
        np.testing.assert_allclose(a['port_vals'], b['port_vals'])

    def test_sized_requires_vol(self):
        with self.assertRaises(ValueError):
            backtest_edge(self.P, np.zeros(300), mode='sized', verbose=False)

    def test_length_validation(self):
        with self.assertRaises(ValueError):
            backtest_edge(self.P[:-2], np.zeros(300), verbose=False)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(500, 4)) * [0.01, 0.02, 0.015, 0.03]
        self.cov = np.cov(a, rowvar=False) * 252

    def test_weights_valid(self):
        mu = np.array([0.05, 0.1, 0.07, 0.12])
        for w in (min_variance_weights(self.cov), risk_parity_weights(self.cov),
                  max_utility_weights(mu, self.cov)):
            self.assertAlmostEqual(w.sum(), 1.0, places=6)
            self.assertTrue((w >= -1e-9).all())

    def test_risk_parity_equalises_contributions(self):
        w = risk_parity_weights(self.cov)
        rc = w * (self.cov @ w)
        self.assertLess(rc.std() / rc.mean(), 0.05)

    def test_bl_limits(self):
        views = np.array([0.2, -0.1, 0.05, 0.3])
        prior = black_litterman_returns(self.cov, views, confidence=1e-5)
        full = black_litterman_returns(self.cov, views, confidence=1 - 1e-5)
        np.testing.assert_allclose(prior, 2.5 * self.cov @ (np.ones(4) / 4), rtol=1e-3)
        np.testing.assert_allclose(full, views, rtol=1e-2, atol=1e-3)


class DataRepairTests(unittest.TestCase):
    def test_neutralise_jump_restores_continuity(self):
        df = synthetic_prices(200)
        df.loc[100:, ['open', 'high', 'low', 'close']] *= 0.6        # artificial -40% step
        self.assertEqual(len(find_jumps(df)), 1)
        fixed = neutralise_jumps(df, log=lambda *_: None)
        self.assertEqual(len(find_jumps(fixed)), 0)
        self.assertAlmostEqual(fixed['close'].iloc[-1], df['close'].iloc[-1])  # newest rows unchanged

    def test_stitch_aligns_old_rows_to_provider(self):
        old = synthetic_prices(300)
        provider = old.iloc[150:].copy()
        provider[['open', 'high', 'low', 'close']] *= 0.5            # provider on another basis
        out = stitch(old, provider, log=lambda *_: None)
        self.assertEqual(len(find_jumps(out, 0.05)), 0)
        self.assertEqual(len(out), len(old))


if __name__ == '__main__':
    unittest.main()
