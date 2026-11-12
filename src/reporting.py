# reporting.py — Aggregate saved experiments into thesis tables and figures
"""
Reads results/metrics_*.json and results/preds_*.csv and writes:

  tables_summary.csv     mean ± std over seeds / folds / symbols, per model
  tables_dm.csv          Diebold-Mariano vs Naive (seed-bagged forecasts), per symbol
  tables_regime.csv      directional accuracy per market regime
  tables_intervals.csv   PICP / width of probabilistic models vs nominal 80%
  tables_ablation.csv    feature-group ablation (Δ vs full)
  fig_*.png              model comparison, CI band, feature importance, ablation
  REPORT.md              all of the above in one markdown file

Conventions: forecasts are next-day log returns; "price MAPE" is computed on the
reconstructed price; DA excludes zero forecasts (so Naive has no DA).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research_pipeline import load_preds, load_records, returns_to_prices
from stats_tests import dm_test

# Validated categorical order (dataviz reference palette, light surface)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

METRIC_COLS = ['mape', 'oos_r2', 'da', 'binary_sharpe', 'binary_total_return',
               'binary_max_drawdown', 'sized_sharpe', 'sized_total_return']


def _style(ax):
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.title.set_color(INK)


def _fmt(mean, std, pct=False, nd=2):
    if pd.isna(mean):
        return '—'
    k = 100 if pct else 1
    return f'{mean*k:.{nd}f} ± {0 if pd.isna(std) else std*k:.{nd}f}'


# ═══════════════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════════════
def summary_table(rec: pd.DataFrame) -> pd.DataFrame:
    """Per model: average over symbols/folds, std across SEEDS of those averages."""
    rows = []
    for model, g in rec.groupby('model'):
        per_seed = g.groupby('seed')[METRIC_COLS].mean(numeric_only=True)
        mean, std = per_seed.mean(), per_seed.std()
        row = dict(model=model, runs=len(g), seeds=g['seed'].nunique())
        for c in METRIC_COLS:
            row[c] = mean[c]
            row[c + '_sd'] = std[c]
        rows.append(row)
    return pd.DataFrame(rows).set_index('model').sort_values('oos_r2', ascending=False)


def bagged_forecast(pred_df: pd.DataFrame, model: str):
    cols = [c for c in pred_df.columns if c.startswith(f'{model}|') and c.count('|') == 1]
    return pred_df[cols].mean(axis=1).values if cols else None


def dm_table(out_dir: str, symbols: list, experiment='main', baseline='Naive') -> pd.DataFrame:
    """DM test of each model's seed-bagged forecast vs the baseline (squared error of returns)."""
    rows = []
    for s in symbols:
        p = load_preds(out_dir, experiment, s)
        if p is None:
            continue
        base = bagged_forecast(p, baseline)
        if base is None:
            continue
        models = sorted({c.split('|')[0] for c in p.columns if '|' in c} - {baseline})
        for m in models:
            f = bagged_forecast(p, m)
            if f is None or len(f) != len(base):
                continue
            r = dm_test(p['y'].values - f, p['y'].values - base, h=1, power=2)
            rows.append(dict(symbol=s, model=m, baseline=baseline, **r))
    return pd.DataFrame(rows)


def dm_overview(dm: pd.DataFrame) -> pd.DataFrame:
    if dm.empty:
        return dm
    g = dm.groupby('model')
    return pd.DataFrame(dict(
        symbols=g.size(),
        better_sig=g['model_better'].sum(),
        worse_sig=g.apply(lambda d: int(((d['mean_diff'] > 0) & (d['p_value'] < 0.05)).sum())),
        mean_loss_diff=g['mean_diff'].mean(),
        median_p_two_sided=g['p_value'].median(),
    )).sort_values('median_p_two_sided')


def regime_table(rec: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in rec.columns if c.startswith('da_') and c not in ('da_p',)]
    if not cols:
        return pd.DataFrame()
    return rec.groupby('model')[cols].mean().rename(columns=lambda c: c.replace('da_', 'DA '))


def interval_table(rec: pd.DataFrame) -> pd.DataFrame:
    if 'int_picp' not in rec.columns:
        return pd.DataFrame()
    g = rec.dropna(subset=['int_picp']).groupby('model')[['int_picp', 'int_mean_width']].mean()
    g['nominal'] = 0.8
    g['coverage_gap'] = g['int_picp'] - g['nominal']
    return g


def ablation_table(rec: pd.DataFrame, model: str = 'Ensemble') -> pd.DataFrame:
    ab = rec[rec['experiment'].str.startswith('ablation_')].copy()
    if ab.empty:
        return pd.DataFrame()
    ab['variant'] = ab['experiment'].str.replace('ablation_', '', regex=False)
    sub = ab[ab['model'] == model]
    if sub.empty:
        sub = ab[ab['model'] == ab['model'].iloc[0]]
        model = ab['model'].iloc[0]
    t = sub.groupby('variant')[['mape', 'oos_r2', 'da', 'binary_sharpe']].mean()
    if 'full' in t.index:
        for c in ('mape', 'oos_r2', 'da', 'binary_sharpe'):
            t[f'd_{c}'] = t[c] - t.loc['full', c]
    t.attrs['model'] = model
    return t.sort_values('d_oos_r2') if 'd_oos_r2' in t else t


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
def fig_model_comparison(summary: pd.DataFrame, path: str):
    metrics = [('oos_r2', 'Out-of-sample R² vs random walk (%)', 100),
               ('da', 'Directional accuracy (%)', 100),
               ('binary_sharpe', 'Sharpe ratio (net of costs)', 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    order = summary.index.tolist()[::-1]
    for ax, (col, title, k) in zip(axes, metrics):
        vals = summary.loc[order, col] * k
        err = summary.loc[order, col + '_sd'].fillna(0) * k
        ax.barh(order, vals if col != 'da' else vals - 50, left=0 if col != 'da' else 50,
                xerr=err, color=SERIES[0], height=.55,
                error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        if col == 'da':
            ax.axvline(50, color=SERIES[1], lw=1.2, ls='--')
            ax.set_xlim(40, 60)
        else:
            ax.axvline(0, color=INK2, lw=.8)
        ax.set_title(title, fontsize=10, loc='left'); _style(ax)
        ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False)
    fig.suptitle('Walk-forward comparison (mean over symbols & folds; whiskers = std over seeds)',
                 fontsize=11, color=INK, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_interval(out_dir: str, symbol: str, path: str, model='TFT-lite', seed=None):
    p = load_preds(out_dir, 'main', symbol)
    if p is None:
        return False
    cols = [c for c in p.columns if c.startswith(f'{model}|') and c.endswith('|lo')]
    if not cols:
        return False
    seed = cols[0].split('|')[1] if seed is None else str(seed)
    pred, lo, hi = p[f'{model}|{seed}'], p[f'{model}|{seed}|lo'], p[f'{model}|{seed}|hi']
    last = p[p['fold'] == p['fold'].iloc[-1]].reset_index(drop=True)
    sl = slice(len(p) - len(last), None)
    pr = returns_to_prices(last['prev_close'].values, pred.values[sl])
    plo = returns_to_prices(last['prev_close'].values, lo.values[sl])
    phi = returns_to_prices(last['prev_close'].values, hi.values[sl])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(last['date'], plo, phi, color=SERIES[0], alpha=.18, lw=0, label='80% interval')
    ax.plot(last['date'], last['real_close'], color=INK, lw=1.4, label='Actual close')
    ax.plot(last['date'], pr, color=SERIES[0], lw=1.2, label=f'{model} median')
    cov = ((last['real_close'] >= plo) & (last['real_close'] <= phi)).mean()
    ax.set_title(f'{symbol} — {model} next-day forecast, fold {last["fold"].iloc[0]} '
                 f'(empirical coverage {cov*100:.0f}% vs 80% nominal)', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_importance(out_dir: str, path: str):
    agg = {}
    for f in os.listdir(out_dir):
        if f.startswith('importance_') and f.endswith('.json'):
            for fold_d in json.load(open(os.path.join(out_dir, f))).values():
                for imp in fold_d.values():
                    for k, v in imp.items():
                        agg.setdefault(k, []).append(v)
    if not agg:
        return False
    s = pd.Series({k: np.mean(v) for k, v in agg.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=SERIES[0], height=.6)
    ax.axvline(1 / len(s), color=SERIES[1], ls='--', lw=1)
    ax.set_title('TFT-lite — mean variable-selection weight (dashed = uniform)', fontsize=10, loc='left')
    _style(ax); ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_ablation(abl: pd.DataFrame, path: str):
    if abl.empty or 'd_oos_r2' not in abl:
        return False
    a = abl.drop(index='full', errors='ignore')
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.3 * len(a) + 1.6), sharey=True)
    for ax, (c, t, k) in zip(axes, [('d_oos_r2', 'Δ OOS R² vs full (pp)', 100),
                                    ('d_da', 'Δ directional accuracy vs full (pp)', 100)]):
        ax.barh(a.index, a[c] * k, color=[SERIES[7] if v < 0 else SERIES[2] for v in a[c]], height=.55)
        ax.axvline(0, color=INK2, lw=.8)
        ax.set_title(t, fontsize=10, loc='left'); _style(ax)
        ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False)
    fig.suptitle('Ablation — removing a feature group (negative = group helps)',
                 fontsize=11, color=INK, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════
def _md(df: pd.DataFrame, floatfmt='.4g') -> str:
    if df is None or df.empty:
        return '_no data_\n'
    d = df.copy()
    d.index.name = d.index.name or ''
    header = '| ' + ' | '.join([d.index.name] + [str(c) for c in d.columns]) + ' |'
    sep = '|' + '---|' * (len(d.columns) + 1)
    lines = [header, sep]
    for idx, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append('—' if pd.isna(v) else
                             (f'{int(v)}' if float(v).is_integer() and abs(v) < 1e6 else f'{v:{floatfmt}}'))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join([str(idx)] + cells) + ' |')
    return '\n'.join(lines) + '\n'


def build_report(out_dir: str) -> str:
    rec = load_records(out_dir)
    if rec.empty:
        print(f"  [report] no metrics found in {out_dir}")
        return None
    main = rec[rec['experiment'].isin(['main'])]
    symbols = sorted(rec['symbol'].unique())
    md = [f'# Experiment report\n\nSymbols: {", ".join(symbols)}  \n'
          f'Folds: {", ".join(sorted(main["fold"].unique())) if len(main) else "—"}  \n'
          f'Seeds: {sorted(main["seed"].unique().tolist()) if len(main) else "—"}\n']

    if len(main):
        s = summary_table(main)
        s.to_csv(os.path.join(out_dir, 'tables_summary.csv'))
        show = pd.DataFrame({
            'runs': s['runs'],
            'MAPE % (price)': [_fmt(a, b, True) for a, b in zip(s['mape'], s['mape_sd'])],
            'OOS R² %': [_fmt(a, b, True) for a, b in zip(s['oos_r2'], s['oos_r2_sd'])],
            'DA %': [_fmt(a, b, True, 1) for a, b in zip(s['da'], s['da_sd'])],
            'Sharpe (binary)': [_fmt(a, b) for a, b in zip(s['binary_sharpe'], s['binary_sharpe_sd'])],
            'Sharpe (sized)': [_fmt(a, b) for a, b in zip(s['sized_sharpe'], s['sized_sharpe_sd'])],
            'Return % (binary)': [_fmt(a, b, True, 1) for a, b in zip(s['binary_total_return'], s['binary_total_return_sd'])],
        }, index=s.index)
        md += ['\n## 1. Model comparison (walk-forward, net of costs)\n',
               'Mean over symbols and folds; ± is the std across seeds. '
               'OOS R² is relative to the random-walk (Naive) forecast: '
               '**positive means the model beats Naive on squared return error**.\n',
               _md(show)]
        fig_model_comparison(s, os.path.join(out_dir, 'fig_model_comparison.png'))
        md.append('\n![model comparison](fig_model_comparison.png)\n')

        bh = main.groupby('model')['binary_bh_return'].mean().mean()
        md.append(f'\nBuy & hold over the same blocks: {bh*100:.1f}% mean return per fold.\n')

        dm = dm_table(out_dir, symbols)
        if not dm.empty:
            dm.to_csv(os.path.join(out_dir, 'tables_dm.csv'), index=False)
            md += ['\n## 2. Diebold-Mariano test vs Naive\n',
                   'Seed-bagged forecasts, squared return error, HLN-corrected. '
                   '`better_sig` = number of symbols where the model is significantly '
                   'better than Naive at 5% (one-sided).\n', _md(dm_overview(dm))]

        reg = regime_table(main)
        if not reg.empty:
            reg.to_csv(os.path.join(out_dir, 'tables_regime.csv'))
            md += ['\n## 3. Directional accuracy by market regime\n', _md(reg)]

        itv = interval_table(main)
        if not itv.empty:
            itv.to_csv(os.path.join(out_dir, 'tables_intervals.csv'))
            md += ['\n## 4. Probabilistic forecasts (80% interval)\n', _md(itv)]
            for s_ in symbols:
                if fig_interval(out_dir, s_, os.path.join(out_dir, f'fig_interval_{s_}.png')):
                    md.append(f'\n![interval {s_}](fig_interval_{s_}.png)\n')
                    break
        if fig_importance(out_dir, os.path.join(out_dir, 'fig_importance.png')):
            md.append('\n![importance](fig_importance.png)\n')

    abl = ablation_table(rec)
    if not abl.empty:
        abl.to_csv(os.path.join(out_dir, 'tables_ablation.csv'))
        md += [f'\n## 5. Ablation ({abl.attrs.get("model", "")})\n', _md(abl)]
        if fig_ablation(abl, os.path.join(out_dir, 'fig_ablation.png')):
            md.append('\n![ablation](fig_ablation.png)\n')

    pooled = rec[rec['experiment'] == 'pooled']
    if len(pooled):
        ps = summary_table(pooled)[['oos_r2', 'da', 'binary_sharpe']]
        md += ['\n## 6. Pooled model (one network, all tickers)\n', _md(ps)]

    rl = rec[rec['experiment'] == 'rl']
    if len(rl):
        r = rl.groupby('symbol')[['binary_total_return', 'binary_bh_return', 'binary_sharpe',
                                  'binary_max_drawdown']].mean()
        md += ['\n## 7. RL agent (fitted Q-iteration)\n', _md(r)]

    for f in sorted(os.listdir(out_dir)):
        if f.startswith('portfolio_') and f.endswith('.csv'):
            md += [f'\n## 8. Portfolio study ({f[10:-4]})\n', _md(pd.read_csv(os.path.join(out_dir, f), index_col=0)),
                   f'\n![portfolio]({f[:-4]}.png)\n']

    tdir = os.path.join(out_dir, 'tuning')
    if os.path.isdir(tdir):
        rows = [json.load(open(os.path.join(tdir, f))) for f in sorted(os.listdir(tdir))]
        md += ['\n## 9. Tuned hyper-parameters\n', _md(pd.DataFrame(rows).set_index('symbol'))]

    path = os.path.join(out_dir, 'REPORT.md')
    open(path, 'w').write('\n'.join(md))
    print(f"  [report] written {path}")
    return path
