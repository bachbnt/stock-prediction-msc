# tuning.py — Hyper-parameter search (Optuna; random-search fallback)
"""
Tuning uses ONLY training data of the first walk-forward fold: its chronological
last 20% is the validation block, so the test years are never touched.
Output: results/tuning/<SYMBOL>.json  ->  pass to research.py with --hparams.
"""
import json
import os
import time

import numpy as np

from research_data import prepare_fold, walk_forward_folds
from models_research import fit_keras
import keras


def _objective_factory(fd, cfg, model_name='LSTM'):
    from models_research import build_gru
    from models import build_lstm
    n = len(fd.X_train)
    cut = int(n * 0.8)
    Xtr, ytr, Xva, yva = fd.X_train[:cut], fd.y_train[:cut], fd.X_train[cut:], fd.y_train[cut:]
    shape = Xtr.shape[1:]

    def evaluate(params) -> float:
        keras.utils.set_random_seed(0)
        keras.backend.clear_session()
        if model_name == 'GRU':
            m = build_gru(shape, units=params['units'], dropout=params['dropout'],
                          l2=params['l2'], lr=params['lr'])
        else:
            m = build_lstm(shape, units=params['units'], dropout=params['dropout'],
                           l2=params['l2'], n_steps=1)
            m.optimizer.learning_rate.assign(params['lr'])
        m.fit(Xtr, ytr, epochs=cfg['max_epochs'], batch_size=cfg['batch_size'],
              validation_data=(Xva, yva), verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=cfg['patience_es'],
                                                       restore_best_weights=True)])
        return float(np.mean((m.predict(Xva, verbose=0).reshape(-1) - yva) ** 2))
    return evaluate


def tune_symbol(symbol: str, cfg: dict, out_dir: str, n_trials: int = 25,
                model_name: str = 'LSTM') -> dict:
    from research_pipeline import default_groups, load_feature_table
    feat = load_feature_table(symbol)
    folds = walk_forward_folds(feat['date'], cfg['test_years'])
    fd = prepare_fold(feat, default_groups(feat), folds[0], cfg['window'])
    evaluate = _objective_factory(fd, cfg, model_name)
    print(f"\n  [tune] {symbol} {model_name}: {n_trials} trials, "
          f"train={len(fd.X_train)} (val = last 20%)")

    def sample(trial=None, rng=None):
        if trial is not None:
            return dict(units=trial.suggest_categorical('units', [32, 48, 64, 96, 128]),
                        dropout=trial.suggest_float('dropout', 0.1, 0.5),
                        l2=trial.suggest_float('l2', 1e-5, 1e-3, log=True),
                        lr=trial.suggest_float('lr', 1e-4, 5e-3, log=True))
        return dict(units=int(rng.choice([32, 48, 64, 96, 128])),
                    dropout=float(rng.uniform(0.1, 0.5)),
                    l2=float(10 ** rng.uniform(-5, -3)),
                    lr=float(10 ** rng.uniform(-4, np.log10(5e-3))))

    t0 = time.time()
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(lambda tr: evaluate(sample(trial=tr)), n_trials=n_trials)
        best, best_val, method = study.best_params, study.best_value, 'optuna-tpe'
    except ImportError:
        rng = np.random.default_rng(0)
        trials = [sample(rng=rng) for _ in range(n_trials)]
        scores = [evaluate(t) for t in trials]
        i = int(np.argmin(scores))
        best, best_val, method = trials[i], scores[i], 'random-search'

    result = dict(symbol=symbol, model=model_name, method=method, n_trials=n_trials,
                  val_mse_scaled=best_val, seconds=round(time.time() - t0, 1), **best)
    os.makedirs(os.path.join(out_dir, 'tuning'), exist_ok=True)
    path = os.path.join(out_dir, 'tuning', f'{symbol}.json')
    json.dump(result, open(path, 'w'), indent=2)
    print(f"  [tune] best={best}  val_mse={best_val:.4f}  -> {path}")
    return result
