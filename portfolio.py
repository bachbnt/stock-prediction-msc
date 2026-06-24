# portfolio.py — Markowitz Mean-Variance Portfolio Optimization
"""
Dùng predicted returns từ LSTM ensemble để ước lượng expected returns,
kết hợp với historical covariance → tìm danh mục tối ưu.

Hai chiến lược:
  1. Min-Variance Portfolio  — rủi ro thấp nhất
  2. Max-Sharpe Portfolio    — hiệu quả vốn cao nhất

So sánh với:
  - Equal-weight (1/N)
  - Single-stock Ensemble strategy (từ evaluate.py)
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
    Dùng predicted return trung bình của Ensemble trên test set
    làm expected annual return cho từng mã.

    Returns: array shape (n_symbols,)
    """
    exp_rets = []
    for r in all_results:
        pred  = r['predictions']['Ensemble']
        # Daily return trung bình từ predicted prices
        daily_rets = np.diff(pred) / pred[:-1]
        ann_ret    = float(np.mean(daily_rets)) * 252
        exp_rets.append(ann_ret)
    return np.array(exp_rets)


def compute_cov_matrix(all_results: list) -> np.ndarray:
    """
    Tính annualized covariance matrix từ actual daily returns của test set.

    Returns: array shape (n_symbols, n_symbols)
    """
    daily_rets_list = []
    for r in all_results:
        real       = r['real_price']
        daily_rets = np.diff(real) / real[:-1]
        daily_rets_list.append(daily_rets)

    # Align lengths
    min_len = min(len(x) for x in daily_rets_list)
    mat     = np.column_stack([x[:min_len] for x in daily_rets_list])
    cov     = np.cov(mat, rowvar=False) * 252   # annualize

    return cov


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _portfolio_stats(weights: np.ndarray, exp_rets: np.ndarray,
                     cov: np.ndarray) -> tuple:
    """Tính return, volatility, sharpe cho một bộ weights."""
    ret  = float(np.dot(weights, exp_rets))
    vol  = float(np.sqrt(weights @ cov @ weights))
    rf   = RISK_FREE_RATE * 252
    shrp = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, shrp


def optimize_portfolio(exp_rets: np.ndarray, cov: np.ndarray) -> dict:
    """
    Markowitz MVO:
      - Min-Variance: minimize σ subject to Σw=1, w>=0
      - Max-Sharpe:   minimize -Sharpe subject to Σw=1, w>=0

    Returns dict với weights + stats cho mỗi chiến lược.
    """
    n = len(exp_rets)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    w0          = np.ones(n) / n   # equal-weight khởi đầu

    # ── Min-Variance ──────────────────────────────────────────────────────────
    def neg_variance(w):
        return float(w @ cov @ w)

    res_mv = minimize(neg_variance, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})
    w_mv    = res_mv.x
    r_mv, v_mv, s_mv = _portfolio_stats(w_mv, exp_rets, cov)

    # ── Max-Sharpe ────────────────────────────────────────────────────────────
    def neg_sharpe(w):
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
    Backtest: phân bổ INITIAL_CAPITAL theo weights cho từng mã.
    Mỗi mã áp dụng chiến lược Ensemble (signal từ dự đoán).
    Tổng giá trị danh mục = Σ weight_i × portfolio_value_i

    Returns: array portfolio values theo thời gian.
    """
    port_series = []
    for r, w in zip(all_results, weights):
        td   = r['trading']['Ensemble']
        pv   = td['port_vals']                 # giá trị danh mục mã i
        pv_w = w * (pv / pv[0]) * INITIAL_CAPITAL   # re-scale
        port_series.append(pv_w)

    # Align lengths
    min_len = min(len(p) for p in port_series)
    total   = np.sum([p[:min_len] for p in port_series], axis=0)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Monte Carlo Efficient Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_frontier(exp_rets: np.ndarray, cov: np.ndarray,
                          n_sims: int = N_PORTFOLIO_SIMS) -> dict:
    """
    Random portfolio simulation để vẽ efficient frontier.

    Returns: dict với arrays 'returns', 'vols', 'sharpes'
    """
    n        = len(exp_rets)
    rf       = RISK_FREE_RATE * 252
    rets_v   = np.zeros(n_sims)
    vols_v   = np.zeros(n_sims)
    sharps_v = np.zeros(n_sims)

    rng = np.random.default_rng(42)
    for i in range(n_sims):
        w = rng.dirichlet(np.ones(n))    # random weights sum to 1
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
    2 biểu đồ:
      1. Efficient Frontier (scatter Monte Carlo + điểm tối ưu)
      2. Portfolio equity curve so sánh Min-Var, Max-Sharpe, Equal-Weight
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    symbols = [r['symbol'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Portfolio Optimization — Markowitz MVO', fontsize=14, fontweight='bold')

    # ── 1. Efficient Frontier ─────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(mc_frontier['vols'] * 100,
                    mc_frontier['returns'] * 100,
                    c=mc_frontier['sharpes'], cmap='viridis',
                    alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    markers = {'min_variance': ('*', '#e74c3c', 'Min-Variance'),
               'max_sharpe':   ('D', '#3498db', 'Max-Sharpe'),
               'equal_weight': ('s', '#2ecc71', 'Equal-Weight')}
    for key, (mk, col, lbl) in markers.items():
        v = opt_result[key]['vol'] * 100
        r = opt_result[key]['ret'] * 100
        ax.scatter(v, r, marker=mk, s=200, color=col, zorder=5, label=lbl)
        ax.annotate(f'{lbl}\n({v:.1f}%, {r:.1f}%)',
                    (v, r), textcoords='offset points',
                    xytext=(8, 4), fontsize=8)

    ax.set_xlabel('Annual Volatility (%)'); ax.set_ylabel('Annual Return (%)')
    ax.set_title('Efficient Frontier (Monte Carlo)'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Portfolio Equity Curves ────────────────────────────────────────────
    ax2 = axes[1]
    colors_map = {'min_variance': '#e74c3c', 'max_sharpe': '#3498db',
                  'equal_weight': '#2ecc71'}
    labels_map = {'min_variance': 'Min-Variance', 'max_sharpe': 'Max-Sharpe',
                  'equal_weight': 'Equal-Weight'}

    for key in ['min_variance', 'max_sharpe', 'equal_weight']:
        pv = backtest_portfolio(opt_result[key]['weights'], all_results)
        ax2.plot(pv / 1e6, color=colors_map[key], lw=1.5,
                 label=f"{labels_map[key]} ({(pv[-1]/pv[0]-1)*100:.1f}%)")

    ax2.set_title('Portfolio Value (VND)'); ax2.set_ylabel('Triệu VND')
    ax2.set_xlabel('Ngày giao dịch')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}M'))
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # Weight table dưới plot
    w_data = [[f'{w*100:.1f}%' for w in opt_result[k]['weights']]
              for k in ['min_variance', 'max_sharpe', 'equal_weight']]
    row_labels = ['Min-Var', 'Max-Sharpe', 'EW']
    tbl = ax2.table(cellText=w_data, rowLabels=row_labels,
                    colLabels=symbols, loc='bottom', bbox=[0, -0.4, 1, 0.25])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    ax2.set_title('Portfolio Value (VND)\n\nWeights:', pad=10)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
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
    print("  PORTFOLIO OPTIMIZATION — KẾT QUẢ")
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
