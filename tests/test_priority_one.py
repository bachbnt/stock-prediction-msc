import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from artifacts import (artifact_directory, build_artifact_config,
                       load_artifact_bundle, save_artifact_bundle)
from config import INITIAL_CAPITAL
from data_loader import inverse_close, prepare_data
from evaluate import compute_trading_metrics


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.features = ['close', 'volume', 'sentiment']
        self.df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'close': np.arange(100.0, 110.0),
            'volume': np.arange(1_000.0, 1_010.0),
            'sentiment': np.linspace(-1.0, 1.0, 10),
        })

    def test_dynamic_features_include_final_single_step_sample(self):
        result = prepare_data(
            self.df, '2024-01-07', window=3, n_steps=1,
            feature_cols=self.features,
        )
        X_train, y_train, X_test, y_test, scaler, _, df_test = result

        self.assertEqual(X_train.shape, (3, 3, 3))
        self.assertEqual(y_train.shape, (3,))
        self.assertEqual(X_test.shape, (4, 3, 3))
        restored = inverse_close(y_test, scaler, self.features)
        np.testing.assert_allclose(restored, df_test['close'].to_numpy())

    def test_dynamic_features_support_direct_multi_step_targets(self):
        result = prepare_data(
            self.df, '2024-01-07', window=3, n_steps=2,
            feature_cols=self.features,
        )
        X_train, y_train, X_test, y_test, scaler, _, _ = result

        self.assertEqual(X_train.shape, (2, 3, 3))
        self.assertEqual(y_train.shape, (2, 2))
        self.assertEqual(X_test.shape, (3, 3, 3))
        restored = inverse_close(y_test, scaler, self.features)
        expected = np.array([[106.0, 107.0],
                             [107.0, 108.0],
                             [108.0, 109.0]])
        np.testing.assert_allclose(restored, expected)

    def test_rejects_missing_features_and_short_training_split(self):
        with self.assertRaisesRegex(ValueError, 'Missing required columns'):
            prepare_data(
                self.df.drop(columns='sentiment'), '2024-01-07',
                window=3, feature_cols=self.features,
            )

        with self.assertRaisesRegex(ValueError, 'Training split has'):
            prepare_data(
                self.df, '2024-01-03', window=3, n_steps=1,
                feature_cols=self.features,
            )


class ArtifactTests(unittest.TestCase):
    def test_round_trip_preserves_scaler_config_and_validation_losses(self):
        config = build_artifact_config(
            symbol='fpt', date_milestone='2024-01-01', window=3,
            n_steps=1, feature_cols=['close', 'sentiment'],
            seed=42, batch_size=32,
        )
        df = pd.DataFrame({
            'date': pd.date_range('2023-12-20', periods=16, freq='D'),
            'close': np.arange(16.0),
            'sentiment': np.linspace(-1.0, 1.0, 16),
        })
        *_, scaler, df_train, _ = prepare_data(
            df, '2024-01-01', window=3, n_steps=1,
            feature_cols=config['feature_cols'],
        )

        with tempfile.TemporaryDirectory() as model_root:
            artifact_dir = artifact_directory(config, model_root=model_root)
            alternate_dir = artifact_directory(
                dict(config, n_steps=5), model_root=model_root
            )
            self.assertNotEqual(artifact_dir, alternate_dir)
            save_artifact_bundle(
                artifact_dir, config, scaler, {'LSTM': 0.0123},
                {'rows': len(df_train)},
            )
            metadata, loaded_scaler, error = load_artifact_bundle(
                artifact_dir, config
            )

            self.assertIsNone(error)
            self.assertEqual(metadata['config'], config)
            self.assertEqual(metadata['training_data']['rows'], len(df_train))
            self.assertAlmostEqual(metadata['val_losses']['LSTM'], 0.0123)
            self.assertEqual(loaded_scaler.n_features_in_, 2)

            incompatible = dict(config, n_steps=5)
            _, _, error = load_artifact_bundle(artifact_dir, incompatible)
            self.assertIn('does not match', error)


class EvaluationConfigTests(unittest.TestCase):
    def test_trading_metrics_use_configured_initial_capital(self):
        metrics = compute_trading_metrics(
            np.array([100.0, 101.0]), np.array([100.0, 101.0]), 'test'
        )
        self.assertEqual(metrics['port_vals'][0], INITIAL_CAPITAL)


if __name__ == '__main__':
    unittest.main()
