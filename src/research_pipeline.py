# research_pipeline.py — Walk-forward, multi-seed, ablation and pooled experiments
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from config import HOSE_LIMIT, RESEARCH
from data_loader import load_data
from evaluate import backtest_edge
from macro_features import load_macro, load_market
from models_research import (BASELINES, DL_POINT, ensemble_returns, fit_predict)
from research_data import (FoldData, prepare_fold, walk_forward_folds)
from regime import REGIME_NAMES
from stationary import (CORE_GROUPS, FEATURE_GROUPS, available_groups, build_features)
from stats_tests import directional_accuracy_test, interval_metrics, oos_r2

LOG_LIMIT = float(np.log1p(HOSE_LIMIT))


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════
def load_feature_table(symbol: str, use_market=True, use_macro=True,
                       use_sentiment=False, sentiment_model=None,
                       refresh_macro=False) -> pd.DataFrame:
    df = load_data(symbol)
    market = load_market(refresh_macro) if use_market else None
    macro = load_macro(refresh_macro) if use_macro else None
    sent = None
    if use_sentiment:
        sent = _sentiment_series(df, symbol, sentiment_model)
    return build_features(df, market=market, macro=macro, sentiment=sent)


def _sentiment_series(df, symbol, model_path):
    try:
        from sentiment.pipeline import run_sentiment_pipeline, add_sentiment_feature
        sent_df = run_sentiment_pipeline(symbol, str(df['date'].min().date()),
                                         str(df['date'].max().date()),
                                         price_df=df, model_path=model_path)
        return add_sentiment_feature(df, sent_df)['sentiment'].values
    except Exception as exc:
        print(f"  [Sentiment] unavailable ({exc}) — group skipped")
        return None


def default_groups(feat: pd.DataFrame, include_regime=True) -> list:
    groups = [g for g in available_groups(feat)]
    if include_regime:
        groups.append('regime')
    return groups


def ablation_variants(groups: list) -> dict:
    """full model + 'drop one group' variants (core price groups and optional ones)."""
    variants = {'full': list(groups)}
    for g in groups:
        rest = [x for x in groups if x != g]
        if rest:
            variants[f'no_{g}'] = rest
    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def returns_to_prices(prev_close: np.ndarray, pred_log_ret: np.ndarray) -> np.ndarray:
    return prev_close * np.exp(np.clip(pred_log_ret, -LOG_LIMIT, LOG_LIMIT))


def evaluate_forecast(fd: FoldData, pred: np.ndarray, extras: dict = None,
                      verbose=False, label='') -> dict:
    """Error, directional, trading and (optional) interval metrics for one forecast."""
    extras = extras or {}
    pred = np.asarray(pred, float)
    pred_price = returns_to_prices(fd.prev_close, pred)
    err_price = fd.real_close - pred_price
    m = dict(
        n=int(len(pred)),
        rmse_price=float(np.sqrt(np.mean(err_price ** 2))),
        mae_price=float(np.mean(np.abs(err_price))),
        mape=float(np.mean(np.abs(err_price / fd.real_close))),
        rmse_ret=float(np.sqrt(np.mean((fd.y_test - pred) ** 2))),
        oos_r2=oos_r2(fd.y_test, pred),
    )
    da = directional_accuracy_test(fd.y_test, pred)
    m.update(da=da['da'], da_p=da['p_value'])

    # per-regime directional accuracy
    for k, name in enumerate(REGIME_NAMES):
        mask = fd.regime == k
        if mask.sum() >= 10:
            m[f'da_{name}'] = directional_accuracy_test(fd.y_test[mask], pred[mask])['da']
            m[f'n_{name}'] = int(mask.sum())

    prices = np.concatenate([[fd.prev_close[0]], fd.real_close])
    exp_ret = np.expm1(np.clip(pred, -LOG_LIMIT, LOG_LIMIT))
    for mode in ('binary', 'sized'):
        bt = backtest_edge(prices, exp_ret, mode=mode, vol=fd.vol,
                           label=f'{label}/{mode}', verbose=verbose)
        for key in ('total_return', 'bh_return', 'sharpe', 'max_drawdown', 'calmar',
                    'n_trades', 'exposure'):
            m[f'{mode}_{key}'] = float(bt[key])

    if 'lo' in extras and 'hi' in extras:
        m.update({f'int_{k}': v for k, v in
                  interval_metrics(fd.y_test, extras['lo'], extras['hi'], nominal=0.8).items()})
    if 'p_up' in extras:
        m['brier'] = float(np.mean((extras['p_up'] - (fd.y_test > 0)) ** 2))
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# One symbol
# ═══════════════════════════════════════════════════════════════════════════════
def run_symbol(symbol: str, models: list, seeds: list, test_years: list, groups: list,
               cfg: dict, out_dir: str, experiment: str = 'main',
               hparams: dict = None, feat: pd.DataFrame = None,
               use_sentiment=False, sentiment_model=None, resume=True) -> list:
    """Run every (fold x seed x model) for one symbol; write metrics + predictions."""
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, f'metrics_{experiment}_{symbol}.json')
    if resume and os.path.exists(mpath):
        print(f"  [{symbol}/{experiment}] cached — skipping (use --no-resume to redo)")
        return json.load(open(mpath))

    print(f"\n{'#'*60}\n  {symbol}  experiment={experiment}  groups={len(groups)}\n{'#'*60}")
    if feat is None:
        feat = load_feature_table(symbol, use_sentiment=use_sentiment,
                                  sentiment_model=sentiment_model)
    groups = [g for g in groups if g == 'regime' or
              all(c in feat.columns for c in FEATURE_GROUPS[g])]
    folds = walk_forward_folds(feat['date'], test_years)
    if not folds:
        raise ValueError(f"{symbol}: no usable walk-forward fold")

    records, pred_rows = [], []
    importances = {}
    for fold in folds:
        fd = prepare_fold(feat, groups, fold, cfg['window'], seed=0)
        train_close = feat.loc[feat['date'] < fold.test_start, 'close'].to_numpy()
        base = pd.DataFrame(dict(date=fd.dates, fold=fold.name, prev_close=fd.prev_close,
                                 real_close=fd.real_close, y=fd.y_test, vol=fd.vol,
                                 regime=fd.regime))
        print(f"  fold {fold.name}: train={len(fd.X_train)} test={len(fd.X_test)} "
              f"features={len(fd.feature_cols)}")
        cols = {}
        for seed in seeds:
            point_preds, point_losses = {}, {}
            for name in models:
                if name in BASELINES and name in ('Naive', 'Drift', 'ARIMA') and seed != seeds[0]:
                    continue                       # deterministic — run once per fold
                t0 = time.time()
                try:
                    res = fit_predict(name, fd, cfg, seed, hparams, train_close=train_close)
                except Exception as exc:
                    print(f"    [{name}] FAILED: {str(exc)[:100]}")
                    continue
                cols[f'{name}|{seed}'] = res['pred']
                for ex in ('lo', 'hi'):
                    if ex in res:
                        cols[f'{name}|{seed}|{ex}'] = res[ex]
                if 'p_up' in res:
                    cols[f'{name}|{seed}|p_up'] = res['p_up']
                if name in DL_POINT:
                    point_preds[name], point_losses[name] = res['pred'], res['val_loss']
                if 'importance' in res:
                    importances.setdefault(fold.name, {})[seed] = res['importance']
                met = evaluate_forecast(fd, res['pred'], res)
                records.append(dict(symbol=symbol, experiment=experiment, fold=fold.name,
                                    seed=seed, model=name, val_loss=res['val_loss'],
                                    seconds=round(time.time() - t0, 1), **met))
                print(f"    [{name:<11}] seed={seed} DA={met['da']*100:5.1f}% "
                      f"R2={met['oos_r2']*100:6.2f}% MAPE={met['mape']*100:.2f}% "
                      f"Sharpe(bin)={met['binary_sharpe']:5.2f} ({res['val_loss']:.4f}) "
                      f"{time.time()-t0:.0f}s")
            if len(point_preds) >= 2:
                ens, w = ensemble_returns(point_preds, point_losses)
                cols[f'Ensemble|{seed}'] = ens
                met = evaluate_forecast(fd, ens)
                records.append(dict(symbol=symbol, experiment=experiment, fold=fold.name,
                                    seed=seed, model='Ensemble', val_loss=float('nan'),
                                    seconds=0.0, weights=w, **met))
        pred_rows.append(pd.concat([base, pd.DataFrame(cols)], axis=1))

    pd.concat(pred_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, f'preds_{experiment}_{symbol}.csv'), index=False)
    if importances:
        json.dump(importances, open(os.path.join(out_dir, f'importance_{experiment}_{symbol}.json'), 'w'),
                  indent=2)
    json.dump(records, open(mpath, 'w'), indent=2, default=_json_default)
    return records


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ═══════════════════════════════════════════════════════════════════════════════
# Pooled model (one network trained on all tickers)
# ═══════════════════════════════════════════════════════════════════════════════
def pool_fold_data(fds: dict) -> tuple:
    """Concatenate per-symbol FoldData into one pooled FoldData + slices for splitting."""
    syms = list(fds)
    first = fds[syms[0]]
    if any(fds[s].feature_cols != first.feature_cols for s in syms):
        raise ValueError("all symbols must share the same feature columns to pool")
    raw_train = np.concatenate([fds[s].y_train * fds[s].y_scale for s in syms])
    scale = float(np.std(raw_train) + 1e-12)
    slices, start = {}, 0
    for s in syms:
        slices[s] = slice(start, start + len(fds[s].y_test))
        start += len(fds[s].y_test)
    cat = lambda attr: np.concatenate([getattr(fds[s], attr) for s in syms])
    pooled = FoldData(
        fold=first.fold, feature_cols=first.feature_cols,
        X_train=cat('X_train'), y_train=raw_train / scale,
        X_test=cat('X_test'), y_test=cat('y_test'), y_scale=scale,
        prev_close=cat('prev_close'), real_close=cat('real_close'),
        dates=pd.concat([fds[s].dates for s in syms], ignore_index=True),
        vol=cat('vol'), regime=cat('regime'), scaler=None,
        train_target_dates=pd.concat([fds[s].train_target_dates for s in syms], ignore_index=True),
        flat_train=cat('flat_train'), flat_test=cat('flat_test'),
    )
    return pooled, slices


def run_pooled(symbols: list, models: list, seeds: list, test_years: list, groups: list,
               cfg: dict, out_dir: str, experiment: str = 'pooled', hparams: dict = None,
               resume=True) -> list:
    """
    One model per (fold, seed) trained on the concatenated training windows of all
    symbols, evaluated per symbol. Return features are stationary and standardised
    per symbol, so pooling is meaningful. ARIMA is skipped (per-series model).
    """
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, f'metrics_{experiment}_ALL.json')
    if resume and os.path.exists(mpath):
        print(f"  [pooled/{experiment}] cached — skipping")
        return json.load(open(mpath))
    feats = {s: load_feature_table(s) for s in symbols}
    models = [m for m in models if m != 'ARIMA']
    folds = walk_forward_folds(min(feats.values(), key=lambda f: len(f))['date'], test_years)
    records = []
    pred_tables = {s: [] for s in symbols}
    for fold in folds:
        fds = {}
        for s in symbols:
            try:
                fds[s] = prepare_fold(feats[s], groups, fold, cfg['window'], seed=0)
            except ValueError as exc:
                print(f"  skip {s} in fold {fold.name}: {exc}")
        if len(fds) < 2:
            continue
        pooled, slices = pool_fold_data(fds)
        print(f"  pooled fold {fold.name}: train={len(pooled.X_train)} test={len(pooled.X_test)}")
        cols = {s: {} for s in fds}
        for seed in seeds:
            for name in models:
                if name in ('Naive', 'Drift') and seed != seeds[0]:
                    continue
                try:
                    res = fit_predict(name, pooled, cfg, seed, hparams)
                except Exception as exc:
                    print(f"    [{name}] FAILED: {str(exc)[:100]}")
                    continue
                for s, sl in slices.items():
                    sub_res = {k: v[sl] for k, v in res.items()
                               if isinstance(v, np.ndarray) and len(v) == len(pooled.y_test)}
                    met = evaluate_forecast(fds[s], res['pred'][sl], sub_res)
                    cols[s][f'{name}|{seed}'] = res['pred'][sl]
                    records.append(dict(symbol=s, experiment=experiment, fold=fold.name,
                                        seed=seed, model=name, val_loss=res['val_loss'],
                                        seconds=0.0, **met))
                print(f"    [{name:<11}] seed={seed} done")
        for s in fds:
            fd = fds[s]
            base = pd.DataFrame(dict(date=fd.dates, fold=fold.name, prev_close=fd.prev_close,
                                     real_close=fd.real_close, y=fd.y_test, vol=fd.vol,
                                     regime=fd.regime))
            pred_tables[s].append(pd.concat([base, pd.DataFrame(cols[s])], axis=1))
    for s, tabs in pred_tables.items():
        if tabs:
            pd.concat(tabs, ignore_index=True).to_csv(
                os.path.join(out_dir, f'preds_{experiment}_{s}.csv'), index=False)
    json.dump(records, open(mpath, 'w'), indent=2, default=_json_default)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Load saved results
# ═══════════════════════════════════════════════════════════════════════════════
def load_records(out_dir: str, experiment: str = None) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith('metrics_') and f.endswith('.json'):
            rows += json.load(open(os.path.join(out_dir, f)))
    df = pd.DataFrame(rows)
    if experiment and len(df):
        df = df[df['experiment'] == experiment]
    return df


def load_preds(out_dir: str, experiment: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(out_dir, f'preds_{experiment}_{symbol}.csv')
    return pd.read_csv(path, parse_dates=['date']) if os.path.exists(path) else None
