#!/usr/bin/env python
"""
snapshot.py — Record end-of-day Vietnam market snapshots that have NO history API.

Runs after the HOSE close (via launchd, see job.sh) and appends one row per
(date, symbol) to CSV files so that, months from now, they can be used as
features in the walk-forward research (research.py).

  vn_foreign_daily.csv   per symbol: price, volume, value, foreign buy/sell/net
                         (foreign net VALUE is an estimate: net volume x avg price)
  vn_breadth_daily.csv   HOSE breadth: advancers/decliners, ceiling/floor, turnover,
                         foreign totals
  macro_daily.csv        SJC gold buy/sell, USD/VND  (best effort — may be empty)

Safety: skips weekends, skips before 14:50 (market still open) unless --force,
and skips days with no trades (holidays / pre-open board shows reference prices).
Re-running on the same day overwrites that day's rows (last good run wins).

Needs the finhub-mcp package (+ vnstock 4.x) in the interpreter that runs it.
"""
import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo('Asia/Ho_Chi_Minh')
DEFAULT_SYMBOLS = ['FPT', 'VCB', 'VIC', 'HPG', 'VNM', 'MWG', 'PNJ', 'SSI']
CLOSE_HOUR, CLOSE_MIN = 14, 50


def log(msg: str):
    print(f"{datetime.now(TZ):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def upsert(path: str, new: pd.DataFrame, keys: list):
    """Merge new rows into a CSV by key, atomically."""
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={'date': str})
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(keys).reset_index(drop=True)
    tmp = path + '.tmp'
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)


def snapshot_board(symbols, today: str, out: str) -> bool:
    from finhub_mcp.vnboard import breadth, fetch_board_for, load_board
    board = fetch_board_for(tuple(symbols), 'KBS')
    if not bool(board['traded'].any()) or float(board['volume'].sum()) <= 0:
        log("no trades on the board (holiday or pre-open) — nothing recorded")
        return False
    rows = board[['symbol', 'price', 'reference', 'volume', 'value_vnd', 'foreign_buy_volume',
                  'foreign_sell_volume', 'foreign_net_volume', 'foreign_net_value_vnd']].copy()
    rows.insert(0, 'date', today)
    upsert(os.path.join(out, 'vn_foreign_daily.csv'), rows, ['date', 'symbol'])
    log(f"foreign: {len(rows)} symbols recorded")

    _, hose = load_board(['KBS'], 'HOSE')
    b = breadth(hose)
    f = b.pop('foreign')
    row = {'date': today, 'group': 'HOSE', **b, **{f'foreign_{k}': v for k, v in f.items()}}
    upsert(os.path.join(out, 'vn_breadth_daily.csv'), pd.DataFrame([row]), ['date', 'group'])
    log(f"breadth: A/D={b['advancers']}/{b['decliners']} foreign net est "
        f"{f['net_value_bn_vnd_est']} bn VND")
    return True


def snapshot_macro(today: str, out: str):
    """SJC gold and USD/VND — external endpoints are flaky, so every item is optional."""
    row = {'date': today}
    try:
        from finhub_mcp.tools.market import _gold_prices
        gold = _gold_prices()
        rec = next((r for r in gold if 'SJC' in str(r.get('name', r.get('type', 'SJC'))).upper()), gold[0])
        for k in ('buy', 'sell'):
            for key in rec:
                if k in str(key).lower():
                    row[f'sjc_{k}'] = rec[key]
                    break
    except Exception as exc:
        log(f"sjc gold unavailable: {str(exc)[:80]}")
    try:
        from finhub_mcp.worldmarkets import forex_rate
        fx = forex_rate('USDVND')
        row['usdvnd'] = fx.get('rate', fx.get('price'))
    except Exception as exc:
        log(f"usdvnd unavailable: {str(exc)[:80]}")
    if len(row) > 1:
        upsert(os.path.join(out, 'macro_daily.csv'), pd.DataFrame([row]), ['date'])
        log(f"macro: {row}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=os.path.expanduser('~/Developer/finhub-snapshot/data'))
    p.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS)
    p.add_argument('--force', action='store_true', help='ignore the time-of-day / weekend guards')
    args = p.parse_args()

    now = datetime.now(TZ)
    today = now.strftime('%Y-%m-%d')
    if not args.force:
        if now.weekday() >= 5:
            log("weekend — skipped"); return 0
        if (now.hour, now.minute) < (CLOSE_HOUR, CLOSE_MIN):
            log("market still open — skipped"); return 0
        done = os.path.join(args.out, '.done', today)
        if os.path.exists(done):
            log("already recorded today — skipped"); return 0
    os.makedirs(os.path.join(args.out, '.done'), exist_ok=True)
    try:
        ok = snapshot_board(args.symbols, today, args.out)
    except Exception as exc:
        log(f"board snapshot FAILED: {exc!r}")
        return 1
    if ok:
        snapshot_macro(today, args.out)
        open(os.path.join(args.out, '.done', today), 'w').close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
