# visualize.py — Charts and plots
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
                save_dir: str = 'plot', future_prices: dict = None):

    dates = pd.to_datetime(df_test['date'].values)
    n     = len(real_price)
    fig   = plt.figure(figsize=(18, 20))
    gs    = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)

    # ── 1. Price prediction ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates[:n], real_price, label='Actual', color='black', lw=1.8, zorder=5)
    for name, pred in predictions.items():
        ax1.plot(dates[:len(pred)], pred, label=name,
                 color=COLORS.get(name, 'gray'), alpha=0.8, lw=1.2)
    ax1.set_title(f'{symbol} — Closing Price Prediction (Test from {DATE_MILESTONE})',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Price (VND)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # ── 2. Portfolio value ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    # Plot Buy & Hold first so it appears at the top of the legend
    first_td = next(iter(trading.values()))
    ax2.plot(first_td['bh_vals'], label='Buy & Hold',
             color='#7f8c8d', ls='--', lw=1.2)
    for name, td in trading.items():
        ax2.plot(td['port_vals'], label=name, color=COLORS.get(name, 'gray'), lw=1.3)
    ax2.set_title('Portfolio Value (VND)', fontsize=11)
    ax2.set_ylabel('VND'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # ── 3. Drawdown ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for name, td in trading.items():
        pv = td['port_vals']; peak = np.maximum.accumulate(pv)
        # Drawdown: percentage decline from the running peak
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
    # MAPE bars on the primary y-axis; DA bars on the secondary y-axis
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
    # Return bars on the primary y-axis; Sharpe ratio bars on the secondary y-axis
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
        # Offset label above or below the bar depending on whether return is positive
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if v >= 0 else -2),
                 f'{v:.1f}%', ha='center', fontsize=8)

    # ── 6. Future Forecast ───────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])

    # Show the last 30 actual trading days as context before the forecast starts
    context_n    = min(30, n)
    context_dates = dates[n - context_n: n]
    context_price = real_price[n - context_n: n]
    ax6.plot(context_dates, context_price,
             color='black', lw=1.8, label='Actual (last 30 days)', zorder=5)

    # Generate future business-day dates starting the day after the last known date
    last_date    = pd.Timestamp(dates[n - 1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=max((len(list(future_prices.values())[0])
                                               if future_prices else 1), 1))

    if future_prices:
        # Connect the last actual price to the first forecast point for a continuous line
        for name, fp in future_prices.items():
            connect_dates  = [last_date] + list(future_dates[:len(fp)])
            connect_prices = [real_price[-1]] + list(fp)
            ax6.plot(connect_dates, connect_prices,
                     color=COLORS.get(name, 'gray'), lw=1.5,
                     marker='o', markersize=5, alpha=0.85,
                     linestyle='--', label=f'{name}: {fp[0]:,.0f}')

    # Vertical red dotted line marking the boundary between actual and forecast data
    ax6.axvline(x=last_date, color='red', lw=1.2, ls=':', alpha=0.7)
    ax6.text(last_date, ax6.get_ylim()[0] if ax6.get_ylim()[0] != 0 else real_price[-1] * 0.99,
             ' Today', color='red', fontsize=8, va='bottom')

    ax6.set_title(f'{symbol} — Future Price Forecast', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Price (VND)')
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    fig.suptitle(f'{symbol} — Comprehensive Analysis', fontsize=15, fontweight='bold')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{symbol}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Plot: {path}")
