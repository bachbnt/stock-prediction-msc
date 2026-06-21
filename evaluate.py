# evaluate.py — Error metrics và trading strategy backtesting
import math
import numpy as np

from config import TAX_BUY, TAX_SELL, BROKERAGE_SELL, RISK_FREE_RATE


def compute_error_metrics(real: np.ndarray, pred: np.ndarray, label='') -> dict:
    """RMSE, MAE, MAPE, Directional Accuracy."""
    rmse = math.sqrt(np.mean((real - pred) ** 2))
    mae  = np.mean(np.abs(real - pred))
    mape = np.mean(np.abs((real - pred) / real))
    da   = np.mean(np.sign(np.diff(real)) == np.sign(np.diff(pred)))

    print(f"  [{label}]  RMSE={rmse:,.0f}  MAE={mae:,.0f}  "
          f"MAPE={mape*100:.2f}%  DA={da*100:.1f}%")

    return dict(rmse=rmse, mae=mae, mape=mape, da=da)


def compute_trading_metrics(real: np.ndarray, pred: np.ndarray, label='') -> dict:
    """
    Chiến lược long-only: mua khi model dự báo tăng, bán khi giảm.
    Vốn ban đầu: 10,000,000 VND. Tính đủ thuế + phí môi giới.
    """
    initial  = 10_000_000
    cash, shares = float(initial), 0.0
    port_vals, daily_rets = [initial], []

    for i in range(len(real) - 1):
        if pred[i + 1] > pred[i] and shares == 0 and cash > 0:
            shares = cash / (real[i] * (1 + TAX_BUY)); cash = 0.0
        elif pred[i + 1] <= pred[i] and shares > 0:
            cash = shares * real[i] * (1 - TAX_SELL - BROKERAGE_SELL); shares = 0.0

        pv = cash + shares * real[i + 1]
        port_vals.append(pv)
        daily_rets.append((pv - port_vals[-2]) / port_vals[-2])

    port_vals  = np.array(port_vals)
    daily_rets = np.array(daily_rets)

    bh_vals      = initial / (real[0] * (1 + TAX_BUY)) * real
    total_return = (port_vals[-1] - initial) / initial
    bh_return    = (bh_vals[-1] - initial) / initial

    sharpe = ((daily_rets.mean() - RISK_FREE_RATE) / daily_rets.std()
              * math.sqrt(252)) if daily_rets.std() > 1e-10 else 0.0

    peak         = np.maximum.accumulate(port_vals)
    max_drawdown = float(((port_vals - peak) / peak).min())
    calmar       = total_return / abs(max_drawdown) if max_drawdown else 0.0

    print(f"  [{label}]  Return={total_return*100:.1f}%  "
          f"B&H={bh_return*100:.1f}%  Sharpe={sharpe:.2f}  "
          f"MaxDD={max_drawdown*100:.1f}%")

    return dict(
        total_return=total_return, bh_return=bh_return,
        sharpe=sharpe, max_drawdown=max_drawdown, calmar=calmar,
        port_vals=port_vals, bh_vals=bh_vals, daily_rets=daily_rets,
    )
