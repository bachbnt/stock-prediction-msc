# portfolio_backtest.py — Walk-forward portfolio study with transaction costs
"""
Compares allocation rules on the walk-forward test period, using the saved
return forecasts of one model (default 'Ensemble', seed-averaged):

  EqualWeight   1/N, rebalanced
  MinVariance   Markowitz minimum variance (no forecasts)
  HistMVO       mean-variance with trailing historical mean returns (no ML control)
  MVO-Model     mean-variance with the MODEL forecasts as expected returns
  BL-Model      Black-Litterman: equilibrium prior + model forecasts as views
  RiskParity    equal risk contribution (no forecasts)

Every `rebalance_days` trading days the weights are recomputed from information
available at that close: trailing 252-day covariance and the average of the most
recent daily forecasts (assumed to persist over the holding period). Turnover is
charged at 0.15% on buys and 0.25% on sells.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (INITIAL_CAPITAL, RISK_FREE_RATE, TAX_BUY, TAX_SELL, BROKERAGE_SELL)
from data_loader import load_data
from portfolio import (black_litterman_returns, max_utility_weights,
                       min_variance_weights, risk_parity_weights)
from research_pipeline import load_preds

STRATEGIES = ['EqualWeight', 'MinVariance', 'HistMVO', 'MVO-Model', 'BL-Model', 'RiskParity']


def forecast_matrix(preds: dict, model: str) -> pd.DataFrame:
    """Seed-averaged forecast (simple return) per date x symbol."""
    cols = {}
    for s, df in preds.items():
        cand = [c for c in df.columns if c.startswith(f'{model}|') and c.count('|') == 1]
        if not cand:
            continue
        r = np.expm1(df[cand].mean(axis=1).clip(-0.07, 0.07))
        cols[s] = pd.Series(r.values, index=pd.to_datetime(df['date']))
    return pd.DataFrame(cols).sort_index()


def weights_for(strategy: str, hist_ret: pd.DataFrame, fcst: pd.DataFrame, i: int,
                rebalance_days: int, view_window: int = 5, confidence: float = 0.5) -> np.ndarray:
    n = hist_ret.shape[1]
    cov = hist_ret.cov().values * 252 + 1e-8 * np.eye(n)
    if strategy == 'EqualWeight':
        return np.ones(n) / n
    if strategy == 'MinVariance':
        return min_variance_weights(cov)
    if strategy == 'RiskParity':
        return risk_parity_weights(cov)
    if strategy == 'HistMVO':
        return max_utility_weights(hist_ret.mean().values * 252, cov)
    # forecast-based: average of the last `view_window` forecasts available at close i-1
    recent = fcst.iloc[max(0, i - view_window):i]
    mu = recent.mean().values * 252                       # annualised daily forecast
    if strategy == 'MVO-Model':
        return max_utility_weights(mu, cov)
    if strategy == 'BL-Model':
        post = black_litterman_returns(cov, mu, confidence=confidence)
        return max_utility_weights(post, cov)
    raise ValueError(strategy)


def simulate(returns: pd.DataFrame, fcst: pd.DataFrame, strategy: str, start: int,
             rebalance_days: int = 5, lookback: int = 252, confidence: float = 0.5) -> dict:
    """returns / fcst: aligned (dates x symbols); trading starts at row `start`."""
    n_days, n = returns.shape
    value, w = float(INITIAL_CAPITAL), np.ones(n) / n
    equity, turnover, weights_hist = [value], [], []
    for i in range(start, n_days):
        if (i - start) % rebalance_days == 0:
            hist = returns.iloc[max(0, i - lookback):i]
            target = weights_for(strategy, hist, fcst, i, rebalance_days, confidence=confidence)
            delta = target - w
            buys, sells = delta.clip(min=0).sum(), (-delta).clip(min=0).sum()
            cost = buys * TAX_BUY + sells * (TAX_SELL + BROKERAGE_SELL)
            value *= (1 - cost)
            turnover.append(buys + sells)
            w = target
            weights_hist.append((returns.index[i], target.copy()))
        day = returns.iloc[i].values
        growth = float(w @ (1 + day))
        value *= growth
        w = w * (1 + day) / growth                      # weights drift between rebalances
        equity.append(value)
    eq = pd.Series(equity[1:], index=returns.index[start:])
    daily = eq.pct_change().fillna(eq.iloc[0] / INITIAL_CAPITAL - 1)
    peak = eq.cummax()
    years = len(eq) / 252
    ann = (eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    return dict(strategy=strategy, equity=eq,
                total_return=eq.iloc[-1] / INITIAL_CAPITAL - 1, ann_return=ann, ann_vol=vol,
                sharpe=(daily.mean() - RISK_FREE_RATE) / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0,
                max_drawdown=float(((eq - peak) / peak).min()),
                avg_turnover=float(np.mean(turnover)) if turnover else 0.0,
                weights=weights_hist)


def run_portfolio_study(symbols: list, out_dir: str, model: str = 'Ensemble',
                        experiment: str = 'main', rebalance_days: int = 5,
                        confidence: float = 0.5) -> pd.DataFrame:
    preds = {s: load_preds(out_dir, experiment, s) for s in symbols}
    preds = {s: p for s, p in preds.items() if p is not None}
    if len(preds) < 3:
        print("  [portfolio] need saved forecasts for >= 3 symbols — run research.py first")
        return None
    fcst = forecast_matrix(preds, model)
    if fcst.empty:
        print(f"  [portfolio] model {model!r} not found in saved predictions")
        return None

    prices = {}
    for s in fcst.columns:
        d = load_data(s)
        prices[s] = d.set_index('date')['close']
    rets = pd.DataFrame(prices).sort_index().pct_change().dropna(how='all').fillna(0.0)
    rets = rets.loc[:fcst.index.max()]                    # stop where forecasts stop
    fcst = fcst.reindex(rets.index).fillna(0.0)
    test_dates = fcst.index[fcst.abs().sum(axis=1) > 0]
    start = rets.index.get_loc(test_dates[0])
    start = max(start, 252)
    print(f"\n{'#'*60}\n  PORTFOLIO STUDY  model={model}  symbols={list(fcst.columns)}\n"
          f"  {rets.index[start].date()} -> {rets.index[-1].date()}  "
          f"rebalance every {rebalance_days} days\n{'#'*60}")

    results = [simulate(rets, fcst, s, start, rebalance_days, confidence=confidence)
               for s in STRATEGIES]
    table = pd.DataFrame([{k: r[k] for k in ('strategy', 'total_return', 'ann_return', 'ann_vol',
                                              'sharpe', 'max_drawdown', 'avg_turnover')}
                          for r in results]).set_index('strategy')
    print(table.round(3).to_string())
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, f'portfolio_{model}.csv'))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in results:
        ax.plot(r['equity'].index, r['equity'] / INITIAL_CAPITAL, label=f"{r['strategy']} "
                f"(Sharpe {r['sharpe']:.2f})", lw=1.6)
    ax.set_title(f'Portfolio strategies — forecasts from {model}, costs included')
    ax.set_ylabel('Growth of 1 VND'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'portfolio_{model}.png'), dpi=140)
    plt.close(fig)
    return table
