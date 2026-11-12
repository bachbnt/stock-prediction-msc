# research.py — Thesis experiments: walk-forward, multi-seed, ablation, significance
"""
Examples
  python research.py --fast --symbols FPT --models LSTM GRU Naive      # smoke test
  python research.py                                                    # all symbols, full grid
  python research.py --ablation --symbols FPT VCB                       # feature-group ablation
  python research.py --pooled                                           # one model for all tickers
  python research.py --tune --symbols FPT                               # Optuna tuning -> results/tuning
  python research.py --rl --symbols FPT                                 # fitted-Q RL trading agent
  python research.py --portfolio                                        # BL / risk-parity backtest
  python research.py --report                                           # tables + figures + REPORT.md
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import json

from config import RESEARCH, SYMBOLS


def parse_args():
    p = argparse.ArgumentParser(description='Research experiments (MSc thesis)',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--symbols', nargs='+', default=SYMBOLS)
    p.add_argument('--models', nargs='+', default=None,
                   help='models to run (default: all, see models_research.DEFAULT_MODELS)')
    p.add_argument('--seeds', nargs='+', type=int, default=RESEARCH['seeds'])
    p.add_argument('--test-years', nargs='+', type=int, default=RESEARCH['test_years'])
    p.add_argument('--groups', nargs='+', default=None,
                   help="feature groups (default: every group available + 'regime')")
    p.add_argument('--out', default=RESEARCH['results_dir'])
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--fast', action='store_true',
                   help='smoke-test preset: 1 seed, 2 test years, 15 epochs')
    p.add_argument('--no-resume', action='store_true', help='recompute cached results')
    p.add_argument('--hparams', default=None,
                   help='JSON file with tuned hyper-parameters (from --tune)')
    # experiments
    p.add_argument('--ablation', action='store_true', help='feature-group ablation study')
    p.add_argument('--ablation-models', nargs='+', default=['LSTM', 'GRU', 'CNN-LSTM'])
    p.add_argument('--pooled', action='store_true', help='train one model across all symbols')
    p.add_argument('--tune', action='store_true', help='hyper-parameter search (Optuna)')
    p.add_argument('--trials', type=int, default=25)
    p.add_argument('--rl', action='store_true', help='fitted-Q-iteration RL trading agent')
    p.add_argument('--portfolio', action='store_true',
                   help='portfolio backtest (Markowitz, Black-Litterman, risk parity)')
    p.add_argument('--portfolio-model', default='Ensemble')
    p.add_argument('--report', action='store_true', help='build tables, figures and REPORT.md')
    p.add_argument('--forecast', action='store_true',
                   help='next-trading-day forecast trained on all data (live_forecast.py)')
    p.add_argument('--sentiment', action='store_true')
    p.add_argument('--sentiment-model', default=None)
    p.add_argument('--refresh-macro', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(RESEARCH)
    seeds, test_years = args.seeds, args.test_years
    if args.fast:
        cfg.update(max_epochs=15, patience_es=4, patience_lr=2, mc_samples=10)
        seeds, test_years = seeds[:1], test_years[-2:]
    if args.epochs:
        cfg['max_epochs'] = args.epochs

    from models_research import DEFAULT_MODELS
    from research_pipeline import (ablation_variants, default_groups, load_feature_table,
                                   run_pooled, run_symbol)
    models = args.models or DEFAULT_MODELS
    def hparams_for(symbol):
        """--hparams may be a JSON file or a directory holding <SYMBOL>.json."""
        if not args.hparams:
            return None
        path = os.path.join(args.hparams, f'{symbol}.json') if os.path.isdir(args.hparams) \
            else args.hparams
        return json.load(open(path)) if os.path.exists(path) else None
    resume = not args.no_resume
    only_extra = args.report or args.portfolio or args.tune or args.rl or args.forecast

    if args.refresh_macro:
        from macro_features import load_macro, load_market
        load_market(True); load_macro(True)

    if args.tune:
        from tuning import tune_symbol
        for s in args.symbols:
            tune_symbol(s, cfg, args.out, n_trials=args.trials)

    main_run = not (args.ablation or args.pooled or only_extra)
    if main_run:
        for s in args.symbols:
            feat = load_feature_table(s, use_sentiment=args.sentiment,
                                      sentiment_model=args.sentiment_model,
                                      refresh_macro=False)
            groups = args.groups or default_groups(feat)
            run_symbol(s, models, seeds, test_years, groups, cfg, args.out, 'main',
                       hparams=hparams_for(s), feat=feat, resume=resume)

    if args.ablation:
        for s in args.symbols:
            feat = load_feature_table(s, use_sentiment=args.sentiment,
                                      sentiment_model=args.sentiment_model)
            groups = args.groups or default_groups(feat)
            for vname, vgroups in ablation_variants(groups).items():
                run_symbol(s, args.ablation_models, seeds, test_years, vgroups, cfg,
                           args.out, f'ablation_{vname}', hparams=hparams_for(s), feat=feat,
                           resume=resume)

    if args.pooled:
        feat = load_feature_table(args.symbols[0])
        groups = args.groups or default_groups(feat)
        run_pooled(args.symbols, models, seeds, test_years, groups, cfg, args.out,
                   'pooled', hparams=hparams_for(args.symbols[0]), resume=resume)

    if args.rl:
        from rl_trading import run_rl_symbol
        for s in args.symbols:
            run_rl_symbol(s, test_years, args.out)

    if args.portfolio:
        from portfolio_backtest import run_portfolio_study
        run_portfolio_study(args.symbols, args.out, model=args.portfolio_model)

    if args.forecast:
        from live_forecast import live_forecast
        for s in args.symbols:
            r = live_forecast(s)
            print(f"\n  {r['symbol']} after close {r['last_date']}: last={r['last_close']:.2f}  "
                  f"naive={r['naive']:.2f}  model={r['model_median']:.2f} "
                  f"({r['model_return_pct']:+.2f}%)  80% [{r['lo80']:.2f}, {r['hi80']:.2f}]  "
                  f"limits [{r['floor']:.2f}, {r['ceiling']:.2f}]")

    if args.report:
        from reporting import build_report
        build_report(args.out)


if __name__ == '__main__':
    main()
