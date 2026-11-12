# evaluate.py — Error metrics and trading strategy backtesting
import math
import numpy as np

from config import (TAX_BUY, TAX_SELL, BROKERAGE_SELL, RISK_FREE_RATE,
                    INITIAL_CAPITAL, ROUND_TRIP_COST, SETTLEMENT_LAG, HOSE_LIMIT)


def compute_error_metrics(real: np.ndarray, pred: np.ndarray, label='') -> dict:
    """Compute RMSE, MAE, MAPE, and Directional Accuracy."""
    rmse = math.sqrt(np.mean((real - pred) ** 2))
    mae  = np.mean(np.abs(real - pred))
    mape = np.mean(np.abs((real - pred) / real))
    # Directional accuracy: fraction of steps where predicted direction matches actual
    da   = np.mean(np.sign(np.diff(real)) == np.sign(np.diff(pred)))

    print(f"  [{label}]  RMSE={rmse:,.0f}  MAE={mae:,.0f}  "
          f"MAPE={mape*100:.2f}%  DA={da*100:.1f}%")

    return dict(rmse=rmse, mae=mae, mape=mape, da=da)


def compute_trading_metrics(real: np.ndarray, pred: np.ndarray, label='') -> dict:
    """
    Long-only trading strategy: buy when model predicts price increase, sell when decrease.
    Uses INITIAL_CAPITAL from config. Accounts for buy/sell tax and brokerage fees.
    """
    initial = INITIAL_CAPITAL
    cash, shares = float(initial), 0.0
    port_vals, daily_rets = [initial], []

    for i in range(len(real) - 1):
        # Buy signal: predicted price rises and we are currently flat (no shares held)
        if pred[i + 1] > pred[i] and shares == 0 and cash > 0:
            shares = cash / (real[i] * (1 + TAX_BUY)); cash = 0.0
        # Sell signal: predicted price falls or stays flat and we hold shares
        elif pred[i + 1] <= pred[i] and shares > 0:
            cash = shares * real[i] * (1 - TAX_SELL - BROKERAGE_SELL); shares = 0.0

        # Mark-to-market portfolio value at end of period i+1
        pv = cash + shares * real[i + 1]
        port_vals.append(pv)
        daily_rets.append((pv - port_vals[-2]) / port_vals[-2])

    port_vals  = np.array(port_vals)
    daily_rets = np.array(daily_rets)

    # Buy-and-hold benchmark: invest all capital at t=0 and hold through the period
    bh_vals      = initial / (real[0] * (1 + TAX_BUY)) * real
    total_return = (port_vals[-1] - initial) / initial
    bh_return    = (bh_vals[-1] - initial) / initial

    # Annualized Sharpe ratio (252 trading days per year); guard against zero std
    sharpe = ((daily_rets.mean() - RISK_FREE_RATE) / daily_rets.std()
              * math.sqrt(252)) if daily_rets.std() > 1e-10 else 0.0

    # Maximum drawdown: largest peak-to-trough decline in portfolio value
    peak         = np.maximum.accumulate(port_vals)
    max_drawdown = float(((port_vals - peak) / peak).min())
    # Calmar ratio: annualized return divided by absolute max drawdown
    calmar       = total_return / abs(max_drawdown) if max_drawdown else 0.0

    print(f"  [{label}]  Return={total_return*100:.1f}%  "
          f"B&H={bh_return*100:.1f}%  Sharpe={sharpe:.2f}  "
          f"MaxDD={max_drawdown*100:.1f}%")

    return dict(
        total_return=total_return, bh_return=bh_return,
        sharpe=sharpe, max_drawdown=max_drawdown, calmar=calmar,
        port_vals=port_vals, bh_vals=bh_vals, daily_rets=daily_rets,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Realistic edge-based backtest
# ═══════════════════════════════════════════════════════════════════════════════
def _perf_stats(port_vals: np.ndarray, bh_vals: np.ndarray, daily_rets: np.ndarray,
                initial: float) -> dict:
    total_return = (port_vals[-1] - initial) / initial
    bh_return = (bh_vals[-1] - initial) / initial
    std = daily_rets.std()
    sharpe = ((daily_rets.mean() - RISK_FREE_RATE) / std * math.sqrt(252)) if std > 1e-10 else 0.0
    peak = np.maximum.accumulate(port_vals)
    max_dd = float(((port_vals - peak) / peak).min())
    years = max(len(daily_rets) / 252.0, 1e-9)
    ann_ret = (port_vals[-1] / initial) ** (1 / years) - 1 if port_vals[-1] > 0 else -1.0
    calmar = ann_ret / abs(max_dd) if max_dd else 0.0
    return dict(total_return=total_return, bh_return=bh_return, sharpe=sharpe,
                max_drawdown=max_dd, calmar=calmar, ann_return=ann_ret)


def backtest_edge(prices: np.ndarray, exp_ret: np.ndarray, mode: str = 'binary',
                  threshold: float = None, exit_threshold: float = 0.0,
                  vol: np.ndarray = None, size_k: float = 0.25,
                  rebalance_band: float = 0.15, settlement_lag: int = None,
                  initial: float = None, label: str = '', verbose: bool = True) -> dict:
    """
    Edge-based long-only backtest that fixes the weaknesses of the legacy rule
    (which compared two consecutive *predictions* and ignored costs).

      prices   : array of length n+1; prices[t] is the close at which the
                 decision of day t is executed. prices[t+1] is the next close.
      exp_ret  : array of length n; model forecast of the simple return
                 prices[t+1]/prices[t]-1, known at the close of day t.

    Rules
      * Forecasts are clipped to the HOSE +/-7% daily band.
      * 'binary': enter when exp_ret > threshold (default = round-trip cost,
        0.40%), exit when exp_ret < exit_threshold.
      * 'sized' : target exposure w = clip(exp_ret / (size_k * vol), 0, 1) —
        full size once the forecast edge reaches `size_k` daily volatilities;
        trades smaller than `rebalance_band` of equity are skipped.
      * T+2 settlement: shares bought on day t cannot be sold before day t+2.
      * Costs: 0.15% on buys, 0.25% on sells (fee + tax).
    """
    prices = np.asarray(prices, dtype=float)
    exp_ret = np.clip(np.asarray(exp_ret, dtype=float), -HOSE_LIMIT, HOSE_LIMIT)
    n = len(exp_ret)
    if len(prices) != n + 1:
        raise ValueError(f"prices must have len(exp_ret)+1 points, got {len(prices)} vs {n}")
    if mode not in ('binary', 'sized'):
        raise ValueError("mode must be 'binary' or 'sized'")
    if mode == 'sized' and vol is None:
        raise ValueError("mode='sized' needs the vol array")
    threshold = ROUND_TRIP_COST if threshold is None else threshold
    lag = SETTLEMENT_LAG if settlement_lag is None else settlement_lag
    initial = float(INITIAL_CAPITAL if initial is None else initial)

    cash = initial
    lots = []                                    # [buy_day, shares]
    port_vals, trades, exposure = [initial], 0, []
    for t in range(n):
        p = prices[t]
        shares = sum(s for _, s in lots)
        equity = cash + shares * p
        holding = shares > 0

        if mode == 'binary':
            if exp_ret[t] > threshold:
                target_w = 1.0
            elif holding and exp_ret[t] >= exit_threshold:
                target_w = shares * p / equity          # hold
            else:
                target_w = 0.0
        else:
            sigma = max(float(vol[t]), 1e-6)
            target_w = float(np.clip(exp_ret[t] / (size_k * sigma), 0.0, 1.0)) \
                if exp_ret[t] > threshold else 0.0

        cur_w = shares * p / equity if equity > 0 else 0.0
        delta_w = target_w - cur_w
        band = 0.02 if mode == 'binary' else rebalance_band
        full_exit = target_w == 0.0 and holding
        if abs(delta_w) > 1e-9 and (abs(delta_w) >= band or full_exit):
            if delta_w > 0:                                         # buy
                spend = min(delta_w * equity, cash)
                buy_shares = spend / (p * (1 + TAX_BUY))
                if buy_shares > 0:
                    cash -= buy_shares * p * (1 + TAX_BUY)
                    lots.append([t, buy_shares]); trades += 1
            else:                                                   # sell sellable lots (FIFO)
                to_sell = min(-delta_w * equity / p, sum(s for d, s in lots if t - d >= lag))
                sold = 0.0
                for lot in lots:
                    if sold >= to_sell - 1e-12:
                        break
                    if t - lot[0] >= lag:
                        q = min(lot[1], to_sell - sold)
                        lot[1] -= q; sold += q
                if sold > 0:
                    cash += sold * p * (1 - TAX_SELL - BROKERAGE_SELL); trades += 1
                lots = [l for l in lots if l[1] > 1e-12]

        shares = sum(s for _, s in lots)
        port_vals.append(cash + shares * prices[t + 1])
        exposure.append(shares * prices[t + 1] / port_vals[-1] if port_vals[-1] > 0 else 0.0)

    port_vals = np.array(port_vals)
    daily_rets = np.diff(port_vals) / port_vals[:-1]
    bh_vals = initial / (prices[0] * (1 + TAX_BUY)) * prices
    stats = _perf_stats(port_vals, bh_vals, daily_rets, initial)
    stats.update(port_vals=port_vals, bh_vals=bh_vals, daily_rets=daily_rets,
                 n_trades=trades, exposure=float(np.mean(exposure)) if exposure else 0.0)
    if verbose:
        print(f"  [{label}]  Return={stats['total_return']*100:.1f}%  "
              f"B&H={stats['bh_return']*100:.1f}%  Sharpe={stats['sharpe']:.2f}  "
              f"MaxDD={stats['max_drawdown']*100:.1f}%  trades={trades}  "
              f"exposure={stats['exposure']*100:.0f}%")
    return stats
