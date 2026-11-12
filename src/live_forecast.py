# live_forecast.py — Next-trading-day forecast trained on ALL available data
"""
Trains on every labelled day up to the last close and forecasts the next day's
log return from the final window. Reports:
  * the Naive forecast (tomorrow = today) — the benchmark that walk-forward tests
    have so far NOT beaten
  * a seed-averaged model median and an 80% interval from quantile models
This is a forecast, not advice: next-day moves are close to unpredictable
(see results/*/REPORT.md for the walk-forward evidence).
"""
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

from config import HOSE_LIMIT, RESEARCH
from models_research import build_lstm_quantile, build_tft_lite, fit_keras
from research_data import make_windows
from stationary import columns_for


def live_forecast(symbol: str, seeds=(42, 43, 44, 45, 46), cfg: dict = None) -> dict:
    from research_pipeline import default_groups, load_feature_table
    cfg = cfg or RESEARCH
    feat = load_feature_table(symbol).sort_values('date').reset_index(drop=True)
    groups = [g for g in default_groups(feat) if g != 'regime']       # regime needs a train/test fit
    cols = columns_for(groups)
    W = cfg['window']
    last = len(feat) - 1
    y = feat['y'].to_numpy()
    lab = np.flatnonzero(np.isfinite(y))                             # rows with a known next-day return
    scaler = StandardScaler().fit(feat.loc[lab, cols].values)        # scale on labelled history only
    xs = np.clip(scaler.transform(feat[cols].values), -8, 8)
    idx = lab[lab >= W - 1]
    X, ys = make_windows(xs, idx, W), y[idx]
    scale = float(ys.std())
    x_live = make_windows(xs, np.array([last]), W)
    qs = cfg['quantiles']
    lo_, med_, hi_ = [], [], []
    for seed in seeds:
        for builder in (build_lstm_quantile, build_tft_lite):
            keras.utils.set_random_seed(seed); keras.backend.clear_session()
            m = builder((W, len(cols)), qs)
            fit_keras(m, X, ys / scale, cfg)
            q = np.sort(m.predict(x_live, verbose=0)[0]) * scale
            lo_.append(q[0]); med_.append(q[len(q) // 2]); hi_.append(q[-1])
    lim = np.log1p(HOSE_LIMIT)
    close = float(feat.loc[last, 'close'])
    price = lambda r: close * float(np.exp(np.clip(r, -lim, lim)))
    med = float(np.mean(med_))
    return dict(symbol=symbol, last_date=feat.loc[last, 'date'].date().isoformat(),
                last_close=close, naive=close, model_median=price(med),
                model_return_pct=(np.exp(med) - 1) * 100,
                lo80=price(np.mean(lo_)), hi80=price(np.mean(hi_)),
                floor=close * (1 - HOSE_LIMIT), ceiling=close * (1 + HOSE_LIMIT),
                models_run=len(med_), spread_of_medians_pct=float(np.std(med_)) * 100)
