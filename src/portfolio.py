# portfolio.py — Markowitz Mean-Variance Portfolio Optimization
"""
Uses predicted returns from the LSTM ensemble to estimate expected returns,
combined with historical covariance to find the optimal portfolio allocation.

Two optimization strategies:
  1. Min-Variance Portfolio  — lowest possible risk
  2. Max-Sharpe Portfolio    — best risk-adjusted return

Compared against:
  - Equal-weight (1/N) baseline
  - Single-stock Ensemble strategy (from evaluate.py)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from config import (SYMBOLS, RISK_FREE_RATE, INITIAL_CAPITAL,
                    N_PORTFOLIO_SIMS, TAX_BUY, TAX_SELL, BROKERAGE_SELL)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Expected Returns & Covariance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_expected_returns(all_results: list) -> np.ndarray:
    """
    Use the average predicted return from the Ensemble model on the test set
    as the expected annualized return for each stock symbol.

    Returns: array of shape (n_symbols,)
    """
    exp_rets = []
    for r in all_results:
        pred  = r['predictions']['Ensemble']
        # Compute daily percentage returns from the predicted price series
        daily_rets = np.diff(pred) / pred[:-1]
        # Annualize by multiplying the mean daily return by 252 trading days
        ann_ret    = float(np.mean(daily_rets)) * 252
        exp_rets.append(ann_ret)
    return np.array(exp_rets)


def compute_cov_matrix(all_results: list) -> np.ndarray:
    """
    Compute the annualized covariance matrix from actual daily returns on the test set.

    Returns: array of shape (n_symbols, n_symbols)
    """
    daily_rets_list = []
    for r in all_results:
        real       = r['real_price']
        daily_rets = np.diff(real) / real[:-1]
        daily_rets_list.append(daily_rets)

    # Align all return series to the shortest length before stacking into a matrix
    min_len = min(len(x) for x in daily_rets_list)
    mat     = np.column_stack([x[:min_len] for x in daily_rets_list])
    # Multiply by 252 to convert the daily covariance to an annualized figure
    cov     = np.cov(mat, rowvar=False) * 252

    return cov


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _portfolio_stats(weights: np.ndarray, exp_rets: np.ndarray,
                     cov: np.ndarray) -> tuple:
    """Compute annualized return, volatility, and Sharpe ratio for a given weight vector."""
    ret  = float(np.dot(weights, exp_rets))
    vol  = float(np.sqrt(weights @ cov @ weights))
    rf   = RISK_FREE_RATE * 252   # annualized risk-free rate
    # Guard against near-zero volatility to avoid division-by-zero
    shrp = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, shrp


def optimize_portfolio(exp_rets: np.ndarray, cov: np.ndarray) -> dict:
    """
    Markowitz Mean-Variance Optimization (MVO):
      - Min-Variance: minimize portfolio variance subject to Σw=1, w>=0
      - Max-Sharpe:   minimize negative Sharpe ratio subject to Σw=1, w>=0

    Returns a dict with weights and performance stats for each strategy.
    """
    n = len(exp_rets)
    # Equality constraint: weights must sum to 1 (fully invested portfolio)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    # Long-only constraint: no short selling allowed
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    w0          = np.ones(n) / n   # start from equal-weight as the initial guess

    # ── Min-Variance ──────────────────────────────────────────────────────────
    def neg_variance(w):
        # Objective: minimize portfolio variance (w^T * Cov * w)
        return float(w @ cov @ w)

    res_mv = minimize(neg_variance, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})
    w_mv    = res_mv.x
    r_mv, v_mv, s_mv = _portfolio_stats(w_mv, exp_rets, cov)

    # ── Max-Sharpe ────────────────────────────────────────────────────────────
    def neg_sharpe(w):
        # Objective: minimize negative Sharpe (equivalent to maximizing Sharpe)
        _, vol, shrp = _portfolio_stats(w, exp_rets, cov)
        return -shrp

    res_ms = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})
    w_ms    = res_ms.x
    r_ms, v_ms, s_ms = _portfolio_stats(w_ms, exp_rets, cov)

    # ── Equal-weight baseline ─────────────────────────────────────────────────
    w_ew    = np.ones(n) / n
    r_ew, v_ew, s_ew = _portfolio_stats(w_ew, exp_rets, cov)

    return {
        'min_variance': dict(weights=w_mv, ret=r_mv, vol=v_mv, sharpe=s_mv),
        'max_sharpe':   dict(weights=w_ms, ret=r_ms, vol=v_ms, sharpe=s_ms),
        'equal_weight': dict(weights=w_ew, ret=r_ew, vol=v_ew, sharpe=s_ew),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Portfolio Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_portfolio(weights: np.ndarray, all_results: list) -> np.ndarray:
    """
    Backtest: allocate INITIAL_CAPITAL across symbols according to the given weights.
    Each symbol uses the Ensemble trading strategy (signal derived from predictions).
    Total portfolio value = Σ weight_i × portfolio_value_i

    Returns: array of portfolio values over time.
    """
    port_series = []
    for r, w in zip(all_results, weights):
        td   = r['trading']['Ensemble']
        pv   = td['port_vals']                       # raw portfolio value for symbol i
        # Re-scale so that the weighted contribution starts at w * INITIAL_CAPITAL
        pv_w = w * (pv / pv[0]) * INITIAL_CAPITAL
        port_series.append(pv_w)

    # Align all series to the same length before summing across symbols
    min_len = min(len(p) for p in port_series)
    total   = np.sum([p[:min_len] for p in port_series], axis=0)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Monte Carlo Efficient Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_frontier(exp_rets: np.ndarray, cov: np.ndarray,
                          n_sims: int = N_PORTFOLIO_SIMS) -> dict:
    """
    Simulate random portfolios to approximate the efficient frontier.

    Returns: dict with arrays 'returns', 'vols', 'sharpes' — one entry per simulation.
    """
    n        = len(exp_rets)
    rf       = RISK_FREE_RATE * 252
    rets_v   = np.zeros(n_sims)
    vols_v   = np.zeros(n_sims)
    sharps_v = np.zeros(n_sims)

    rng = np.random.default_rng(42)   # fixed seed for reproducibility
    for i in range(n_sims):
        # Dirichlet distribution naturally produces weights that sum to 1
        w = rng.dirichlet(np.ones(n))
        r, v, s = _portfolio_stats(w, exp_rets, cov)
        rets_v[i]   = r
        vols_v[i]   = v
        sharps_v[i] = s

    return dict(returns=rets_v, vols=vols_v, sharpes=sharps_v)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_portfolio(all_results: list, opt_result: dict,
                   mc_frontier: dict, save_dir: str = 'plot'):
    """
    Generate two charts:
      1. Efficient Frontier (Monte Carlo scatter plot + optimal portfolio markers)
      2. Portfolio equity curves comparing Min-Variance, Max-Sharpe, and Equal-Weight
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    symbols = [r['symbol'] for r in all_results]

    # Layout: 2 rows — top row has frontier + equity curve, bottom row has weights table
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Portfolio Optimization — Markowitz MVO', fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.45, wspace=0.35)

    # ── 1. Efficient Frontier ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(mc_frontier['vols'] * 100,
                    mc_frontier['returns'] * 100,
                    c=mc_frontier['sharpes'], cmap='viridis',
                    alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    markers = {'min_variance': ('*', '#e74c3c', 'Min-Variance'),
               'max_sharpe':   ('D', '#3498db', 'Max-Sharpe'),
               'equal_weight': ('s', '#2ecc71', 'Equal-Weight')}

    # Build legend entries with vol/return info embedded in the label so the
    # chart needs no separate annotations. Line2D with color='w' (white body)
    # gives a clean marker-only appearance in the legend.
    from matplotlib.lines import Line2D
    legend_elements = []
    for key, (mk, col, lbl) in markers.items():
        v = opt_result[key]['vol'] * 100
        r = opt_result[key]['ret'] * 100
        ax.scatter(v, r, marker=mk, s=200, color=col, zorder=5)
        legend_elements.append(
            Line2D([0], [0], marker=mk, color='w', markerfacecolor=col,
                   markersize=9, label=f'{lbl}  ({v:.1f}%, {r:.1f}%)')
        )

    ax.set_xlabel('Annual Volatility (%)'); ax.set_ylabel('Annual Return (%)')
    ax.set_title('Efficient Frontier (Monte Carlo)')
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.margins(0.1)  # add padding so markers near the edge don't get clipped

    # ── 2. Portfolio Equity Curves ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors_map = {'min_variance': '#e74c3c', 'max_sharpe': '#3498db',
                  'equal_weight': '#2ecc71'}
    labels_map = {'min_variance': 'Min-Variance', 'max_sharpe': 'Max-Sharpe',
                  'equal_weight': 'Equal-Weight'}

    for key in ['min_variance', 'max_sharpe', 'equal_weight']:
        pv = backtest_portfolio(opt_result[key]['weights'], all_results)
        ax2.plot(pv / 1e6, color=colors_map[key], lw=1.5,
                 label=f"{labels_map[key]} ({(pv[-1]/pv[0]-1)*100:.1f}%)")

    ax2.set_title('Portfolio Value (VND)'); ax2.set_ylabel('Million VND')
    ax2.set_xlabel('Trading Day')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}M'))
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── 3. Weights table — dedicated subplot row, no overlap ─────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    w_data = [[f'{w*100:.1f}%' for w in opt_result[k]['weights']]
              for k in ['min_variance', 'max_sharpe', 'equal_weight']]
    row_labels = ['Min-Var', 'Max-Sharpe', 'Equal-Weight']
    tbl = ax3.table(cellText=w_data, rowLabels=row_labels,
                    colLabels=symbols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax3.set_title('Weight Allocation', fontsize=10, pad=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'portfolio_optimization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Portfolio plot: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Print Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_portfolio_summary(opt_result: dict, all_results: list):
    symbols = [r['symbol'] for r in all_results]
    print(f"\n{'═'*65}")
    print("  PORTFOLIO OPTIMIZATION — RESULTS")
    print(f"{'═'*65}")
    print(f"  {'Strategy':<15} {'Ann.Ret%':>9} {'Ann.Vol%':>9} "
          f"{'Sharpe':>8}  {'Weights'}")
    print(f"  {'-'*60}")
    for key, lbl in [('min_variance', 'Min-Variance'),
                     ('max_sharpe',   'Max-Sharpe'),
                     ('equal_weight', 'Equal-Weight')]:
        p = opt_result[key]
        w_str = '  '.join(f'{s}={p["weights"][i]*100:.0f}%'
                          for i, s in enumerate(symbols))
        print(f"  {lbl:<15} {p['ret']*100:>9.2f} {p['vol']*100:>9.2f} "
              f"{p['sharpe']:>8.2f}  {w_str}")
    print(f"{'═'*65}")
