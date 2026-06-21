# visualize.py — Charts và plots
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

from config import COLORS, DATE_MILESTONE


def plot_symbol(symbol: str, df_test: pd.DataFrame, real_price: np.ndarray,
                predictions: dict, trading: dict, error_metrics: dict,
                save_dir: str = 'plot'):

    dates = pd.to_datetime(df_test['date'].values)
    n     = len(real_price)
    fig   = plt.figure(figsize=(18, 16))
    gs    = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── 1. Price prediction ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates[:n], real_price, label='Actual', color='black', lw=1.8, zorder=5)
    for name, pred in predictions.items():
        ax1.plot(dates[:len(pred)], pred, label=name,
                 color=COLORS.get(name, 'gray'), alpha=0.8, lw=1.2)
    ax1.set_title(f'{symbol} — Dự đoán giá đóng cửa (Test từ {DATE_MILESTONE})',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Ngày'); ax1.set_ylabel('Giá (VND)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # ── 2. Portfolio value ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bh_done = False
    for name, td in trading.items():
        ax2.plot(td['port_vals'], label=name, color=COLORS.get(name, 'gray'), lw=1.3)
        if not bh_done:
            ax2.plot(td['bh_vals'], label='Buy & Hold',
                     color='#7f8c8d', ls='--', lw=1.2)
            bh_done = True
    ax2.set_title('Giá trị danh mục (VND)', fontsize=11)
    ax2.set_ylabel('VND'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # ── 3. Drawdown ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for name, td in trading.items():
        pv = td['port_vals']; peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak * 100
        ax3.fill_between(range(len(dd)), dd, 0,
                         alpha=0.45, color=COLORS.get(name, 'gray'), label=name)
    ax3.set_title('Drawdown (%)', fontsize=11)
    ax3.set_ylabel('%'); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # ── 4. MAPE & DA ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    models = list(error_metrics.keys()); x = np.arange(len(models)); w = 0.35
    mape_v = [error_metrics[m]['mape'] * 100 for m in models]
    da_v   = [error_metrics[m]['da'] * 100 for m in models]
    b1 = ax4.bar(x - w/2, mape_v, w,
                 color=[COLORS.get(m, 'gray') for m in models], alpha=0.8)
    ax4r = ax4.twinx()
    b2 = ax4r.bar(x + w/2, da_v, w,
                  color=[COLORS.get(m, 'gray') for m in models], alpha=0.4)
    ax4.set_title('MAPE & Directional Accuracy', fontsize=11)
    ax4.set_xticks(x); ax4.set_xticklabels(models, fontsize=9)
    ax4.set_ylabel('MAPE (%)'); ax4r.set_ylabel('DA (%)')
    for bar, v in zip(b1, mape_v):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for bar, v in zip(b2, da_v):
        ax4r.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3, f'{v:.1f}', ha='center', fontsize=8)

    # ── 5. Return & Sharpe ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ret_v    = [trading[m]['total_return'] * 100 for m in models]
    sharpe_v = [trading[m]['sharpe'] for m in models]
    b3 = ax5.bar(x - w/2, ret_v, w,
                 color=[COLORS.get(m, 'gray') for m in models], alpha=0.8)
    ax5r = ax5.twinx()
    ax5r.bar(x + w/2, sharpe_v, w,
             color=[COLORS.get(m, 'gray') for m in models], alpha=0.4)
    ax5.set_title('Return & Sharpe Ratio', fontsize=11)
    ax5.set_xticks(x); ax5.set_xticklabels(models, fontsize=9)
    ax5.set_ylabel('Return (%)'); ax5r.set_ylabel('Sharpe')
    ax5.axhline(0, color='black', lw=0.8, ls='--')
    for bar, v in zip(b3, ret_v):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if v >= 0 else -2),
                 f'{v:.1f}%', ha='center', fontsize=8)

    fig.suptitle(f'{symbol} — Phân tích toàn diện', fontsize=15, fontweight='bold')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{symbol}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Plot: {path}")
