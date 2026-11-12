# rl_trading.py — Offline RL trading agent (Fitted Q-Iteration)
"""
Replaces the rule "buy when the model predicts a rise" with a policy LEARNED from
data. This is a compact, dependency-free reinforcement-learning baseline —
fitted Q-iteration (batch RL) — rather than stable-baselines3 / DQN:

  State   : standardised stationary features at close of day t (+ current position)
  Action  : 0 = be flat, 1 = be long
  Reward  : position_after_action * next-day simple return - switching cost
            (cost per switch = half the round-trip fee; T+2 is ignored in training
             but enforced by the evaluation backtest)
  Q       : gradient-boosted regressor on (state, position, action); target
            r + gamma * max_a' Q(s', a, a'); a fixed number of FQI sweeps.

Because actions do not influence the market, every (state, position, action)
transition is observable from historical data, so all four (position, action)
combinations are used at each step. The greedy policy is trained on the training
years only and executed on each walk-forward test block via backtest_edge.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import ROUND_TRIP_COST
from evaluate import backtest_edge
from research_data import prepare_fold, walk_forward_folds


class FQIAgent:
    def __init__(self, gamma=0.9, sweeps=8, switch_cost=ROUND_TRIP_COST / 2, seed=0):
        self.gamma, self.sweeps, self.cost, self.seed = gamma, sweeps, switch_cost, seed
        self.q = None

    @staticmethod
    def _inp(S, pos, act):
        n = len(S)
        return np.column_stack([S, np.full(n, pos), np.full(n, act)])

    def _predict(self, S, pos, act):
        if self.q is None:
            return np.zeros(len(S))
        return self.q.predict(self._inp(S, pos, act))

    def fit(self, S: np.ndarray, ret: np.ndarray):
        """S: states (n, d); ret[t]: simple return from close t to close t+1."""
        n = len(S) - 1                       # transitions t -> t+1
        X_all, rewards = [], {}
        for pos in (0, 1):
            for act in (0, 1):
                rewards[(pos, act)] = act * ret[:n] - self.cost * abs(act - pos)
                X_all.append(self._inp(S[:n], pos, act))
        X_all = np.vstack(X_all)
        for _ in range(self.sweeps):
            # value of the next state if we stay in position `act`
            v_next = {act: np.maximum(self._predict(S[1:n + 1], act, 0),
                                      self._predict(S[1:n + 1], act, 1)) for act in (0, 1)}
            y = np.concatenate([rewards[(pos, act)] + self.gamma * v_next[act]
                                for pos in (0, 1) for act in (0, 1)])
            self.q = HistGradientBoostingRegressor(
                max_iter=80, learning_rate=0.08, max_depth=3, min_samples_leaf=60,
                l2_regularization=2.0, early_stopping=False, random_state=self.seed
            ).fit(X_all, y)
        return self

    def act(self, S: np.ndarray, pos: int) -> int:
        q = [self._predict(S, pos, a)[0] for a in (0, 1)]
        return int(q[1] > q[0])


def policy_signals(agent: FQIAgent, S: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """Run the greedy policy forward; returns +1% (long) / -1% (flat) pseudo-forecasts
    that backtest_edge's binary mode interprets as enter / exit."""
    pos, sig = start_pos, []
    for s in S:
        pos = agent.act(s[None, :], pos)
        sig.append(0.01 if pos else -0.01)
    return np.array(sig)


def run_rl_symbol(symbol: str, test_years, out_dir: str, seeds=(0, 1, 2)) -> list:
    from research_pipeline import default_groups, load_feature_table
    feat = load_feature_table(symbol)
    groups = default_groups(feat)
    records = []
    print(f"\n  [RL] {symbol}: fitted-Q agent, {len(seeds)} seeds")
    for fold in walk_forward_folds(feat['date'], test_years):
        fd = prepare_fold(feat, groups, fold, window=60)
        # training states / returns use the last row of each window
        S_tr = fd.X_train[:, -1, :]
        ret_tr = np.expm1(fd.y_train * fd.y_scale)
        S_te = fd.X_test[:, -1, :]
        prices = np.concatenate([[fd.prev_close[0]], fd.real_close])
        for seed in seeds:
            agent = FQIAgent(seed=seed).fit(S_tr, ret_tr)
            sig = policy_signals(agent, S_te)
            bt = backtest_edge(prices, sig, mode='binary', threshold=0.005, verbose=False)
            records.append(dict(symbol=symbol, experiment='rl', fold=fold.name, seed=seed,
                                model='RL-FQI', n=len(sig),
                                **{f'binary_{k}': float(bt[k]) for k in
                                   ('total_return', 'bh_return', 'sharpe', 'max_drawdown',
                                    'calmar', 'n_trades', 'exposure')}))
        sel = [r for r in records if r['fold'] == fold.name]
        print(f"    fold {fold.name}: Sharpe={np.mean([r['binary_sharpe'] for r in sel]):5.2f} "
              f"Return={np.mean([r['binary_total_return'] for r in sel])*100:6.1f}% "
              f"B&H={sel[0]['binary_bh_return']*100:6.1f}%")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(records, open(os.path.join(out_dir, f'metrics_rl_{symbol}.json'), 'w'), indent=2)
    return records
