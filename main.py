# main.py — Entry point: Vietnamese Stock Price Prediction (MSc Thesis)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (SYMBOLS, DATE_MILESTONE, WINDOW_SIZE, BATCH_SIZE,
                    FEATURE_COLS, TARGET_COL, SEED, N_PREDICT_STEPS)
from artifacts   import (artifact_directory, build_artifact_config,
                         load_artifact_bundle, save_artifact_bundle)
from data_loader import load_data, update_data, prepare_data, inverse_close
from features    import add_features
from models      import build_lstm, build_bilstm, build_cnnlstm, train_model, ensemble_predict
from evaluate    import compute_error_metrics, compute_trading_metrics
from visualize   import plot_symbol

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Full pipeline for a single stock symbol
# ═══════════════════════════════════════════════════════════════════════════════

def run_symbol(symbol: str, save_dir: str = 'plot',
               refresh: bool = False,
               n_steps: int = None,
               use_sentiment: bool = False,
               sentiment_model: str = None,
               plot_only: bool = False) -> dict:
    """
    Run the complete prediction pipeline for one stock symbol.

    Args:
        symbol:          Stock ticker (e.g. 'FPT')
        save_dir:        Directory to save charts
        refresh:         True → update CSV from vnstock before running
        n_steps:         Number of forecast steps (None → use N_PREDICT_STEPS from config)
        use_sentiment:   True → add sentiment feature from news articles
        sentiment_model: Path to a fine-tuned PhoBERT model (None → keyword fallback)
        plot_only:       True → load saved models and plot only, skip retraining
    """
    if n_steps is None:
        n_steps = N_PREDICT_STEPS

    print(f"\n{'#'*60}\n  {symbol}  (n_steps={n_steps})\n{'#'*60}")

    # 1. Load historical price data
    df = update_data(symbol) if refresh else load_data(symbol)
    df = add_features(df)

    # 2. Optionally append sentiment feature column
    if use_sentiment:
        _add_sentiment(df, symbol, sentiment_model)

    feature_cols = list(FEATURE_COLS)
    if use_sentiment and 'sentiment' in df.columns:
        if 'sentiment' not in feature_cols:
            feature_cols.append('sentiment')
        print(f"  Features: {len(feature_cols)} (incl. sentiment) | Rows: {len(df)}")
    else:
        print(f"  Features: {len(feature_cols)} | Rows: {len(df)}")

    artifact_config = build_artifact_config(
        symbol=symbol,
        date_milestone=DATE_MILESTONE,
        window=WINDOW_SIZE,
        n_steps=n_steps,
        feature_cols=feature_cols,
        sentiment_model=(sentiment_model if 'sentiment' in feature_cols else None),
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
    artifact_dir = artifact_directory(artifact_config)
    artifact_metadata = artifact_scaler = None
    if plot_only:
        artifact_metadata, artifact_scaler, artifact_error = \
            load_artifact_bundle(artifact_dir, artifact_config)
        if artifact_error:
            print(f"  [Artifact] {artifact_error}; models will be trained.")
        else:
            print(f"  [Artifact] Loaded config from {artifact_dir}")

    # 3. Build sliding-window sequences for train/test split. A plot-only run
    # reuses the exact scaler stored with the matching model configuration.
    X_train, y_train, X_test, y_test, scaler, df_train, df_test = \
        prepare_data(df, DATE_MILESTONE, WINDOW_SIZE, n_steps,
                     feature_cols=feature_cols, scaler=artifact_scaler)
    input_shape = (WINDOW_SIZE, len(feature_cols))

    # Use step-1 predictions as ground truth for error metrics and visualization
    if n_steps == 1:
        real_price = inverse_close(y_test, scaler, feature_cols)
    else:
        # For multi-step output, use the first predicted step as the reference target
        real_price = inverse_close(y_test[:, 0], scaler, feature_cols)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    predictions, error_metrics, trading, val_losses = {}, {}, {}, {}
    trained_models = {}

    # 4. Train or load each deep learning model
    artifact_changed = False
    for name, builder in [
        ('LSTM',     lambda: build_lstm(input_shape, n_steps=n_steps)),
        ('BiLSTM',   lambda: build_bilstm(input_shape, n_steps=n_steps)),
        ('CNN-LSTM', lambda: build_cnnlstm(input_shape, n_steps=n_steps)),
    ]:
        model_path = os.path.join(artifact_dir, f'{name}.keras')
        loaded = False
        saved_losses = (artifact_metadata or {}).get('val_losses', {})
        if plot_only and name in saved_losses and os.path.exists(model_path):
            try:
                print(f"\n  Loading {name} from {model_path}...")
                from keras.models import load_model
                model = load_model(model_path)
                if tuple(model.input_shape[1:]) != input_shape:
                    raise ValueError(
                        f"input shape {model.input_shape[1:]} != {input_shape}"
                    )
                if model.output_shape[-1] != n_steps:
                    raise ValueError(
                        f"output steps {model.output_shape[-1]} != {n_steps}"
                    )
                vl = float(saved_losses[name])
                loaded = True
            except Exception as exc:
                print(f"  [Artifact] Cannot load {name}: {exc}; retraining.")

        if not loaded:
            model = builder()
            print(f"\n  Training {name}...")
            vl = train_model(model, X_train, y_train, BATCH_SIZE)
            os.makedirs(artifact_dir, exist_ok=True)
            model.save(model_path)
            artifact_changed = True
            print(f"  → Saved: {model_path}")

        raw_pred = model.predict(X_test, verbose=0)
        # Use step 1 for the common evaluation chart when the model is multi-step.
        if raw_pred.ndim == 2 and raw_pred.shape[1] > 1:
            pred = inverse_close(raw_pred[:, 0], scaler, feature_cols)
        else:
            pred = inverse_close(raw_pred, scaler, feature_cols)

        predictions[name]    = pred
        val_losses[name]     = vl
        trained_models[name] = model
        error_metrics[name]  = compute_error_metrics(real_price, pred, name)
        trading[name]        = compute_trading_metrics(real_price, pred, name)

    if artifact_changed or artifact_metadata is None:
        training_data = {
            'rows': len(df_train),
            'start': df_train['date'].min().date().isoformat(),
            'end': df_train['date'].max().date().isoformat(),
        }
        save_artifact_bundle(
            artifact_dir, artifact_config, scaler, val_losses, training_data
        )
        print(f"  [Artifact] Saved scaler and metadata to {artifact_dir}")

    # 5. ARIMA baseline (walk-forward validation — re-fits on each new observation)
    from statsmodels.tsa.arima.model import ARIMA
    print(f"\n  ARIMA(5,1,0) walk-forward...")
    history     = list(df_train['close'].values)
    arima_preds = []
    for actual in df_test['close'].values[:len(real_price)]:
        try:
            yhat = float(ARIMA(history, order=(5, 1, 0)).fit().forecast(1).iloc[0])
        except Exception:
            yhat = history[-1]  # fallback: carry last known value
        arima_preds.append(yhat)
        history.append(actual)  # expand history window for next iteration
    pred_arima         = np.array(arima_preds)
    # Proxy validation loss for ARIMA: MSE of recent price changes (variance of returns)
    arima_proxy_loss   = float(np.mean(np.diff(df_train['close'].values[-50:]) ** 2))
    predictions['ARIMA']   = pred_arima
    val_losses['ARIMA']    = arima_proxy_loss
    error_metrics['ARIMA'] = compute_error_metrics(real_price, pred_arima, 'ARIMA')
    trading['ARIMA']       = compute_trading_metrics(real_price, pred_arima, 'ARIMA')

    # 6. Ensemble: weighted average of the 3 deep learning models
    dl_preds  = {k: predictions[k] for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    dl_losses = {k: val_losses[k]  for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    pred_ens, ens_weights = ensemble_predict(dl_preds, dl_losses)
    predictions['Ensemble']   = pred_ens
    error_metrics['Ensemble'] = compute_error_metrics(real_price, pred_ens, 'Ensemble')
    trading['Ensemble']       = compute_trading_metrics(real_price, pred_ens, 'Ensemble')

    # 7. Forecast future prices using the last available window.
    # horizon = n_steps so --steps also controls the forecast length.
    future_prices = _forecast_future(
        trained_models, ens_weights, df, scaler, n_steps, WINDOW_SIZE,
        horizon=n_steps, feature_cols=feature_cols
    )

    # 8. Generate and save visualization
    plot_symbol(symbol, df_test, real_price,
                predictions, trading, error_metrics, save_dir,
                future_prices=future_prices)

    # 9. Print summary for this symbol
    bh = list(trading.values())[0]['bh_return']
    print(f"\n  Buy & Hold: {bh*100:.1f}%")
    if future_prices:
        last_actual = real_price[-1]
        for name, fp in future_prices.items():
            chg   = (fp[0] - last_actual) / last_actual * 100
            arrow = '↑' if chg > 0 else '↓'
            print(f"  Forecast [{name}]: {fp[0]:,.0f} VND  {arrow}{abs(chg):.2f}%")

    return dict(
        symbol=symbol,
        real_price=real_price,
        error_metrics=error_metrics,
        trading=trading,
        predictions=predictions,
        future_prices=future_prices,
    )


def _forecast_future(trained_models: dict, ens_weights: dict,
                     df: pd.DataFrame, scaler, n_steps: int,
                     window: int, horizon: int = 1,
                     feature_cols=None) -> dict:
    """
    Forecast the next `horizon` trading days using an iterative / auto-regressive
    approach for single-step models (n_steps=1), or a single forward pass for
    multi-step models (n_steps>1).

    Iterative approach (n_steps=1):
      - Predict 1 step → append predicted close to the window (other features
        held constant at their last known values) → shift window forward → repeat.
      - This propagates uncertainty, so predictions drift toward the mean over
        longer horizons — which is expected behaviour.

    Args:
        horizon: number of future trading days to forecast (default 1).
    """
    try:
        feature_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
        target_idx = feature_cols.index(TARGET_COL)
        future = {}

        for name, model in trained_models.items():
            if n_steps > 1:
                # Multi-step: single forward pass returns `n_steps` values
                last_window = df[feature_cols].values[-window:]
                last_scaled = scaler.transform(last_window)
                X = last_scaled.reshape(1, window, len(feature_cols))
                raw = model.predict(X, verbose=0)
                future[name] = inverse_close(
                    raw, scaler, feature_cols
                ).flatten()[:horizon]
            else:
                # Single-step: iterative / auto-regressive forecast
                # Start with a copy of the last `window` rows (scaled)
                window_scaled = scaler.transform(
                    df[feature_cols].values[-window:]
                ).copy()   # shape: (window, n_features)

                preds = []
                for _ in range(horizon):
                    X = window_scaled.reshape(1, window, len(feature_cols))
                    raw = model.predict(X, verbose=0)          # (1, 1)
                    next_price_scaled = float(raw[0, 0])
                    preds.append(next_price_scaled)

                    # Slide the window forward: drop oldest row, append new row.
                    # Copy the last row's features and update only the close column.
                    new_row = window_scaled[-1].copy()
                    new_row[target_idx] = next_price_scaled
                    window_scaled = np.vstack([window_scaled[1:], new_row])

                # Convert scaled predictions back to price domain all at once
                future[name] = inverse_close(
                    np.array(preds), scaler, feature_cols
                )

        # Ensemble: weighted average of individual model forecasts
        ens = sum(ens_weights[k] * future[k]
                  for k in ens_weights if k in future)
        future['Ensemble'] = np.array(ens).flatten()

        return future
    except Exception as e:
        print(f"  [Forecast] Error: {e}")
        return {}


def _add_sentiment(df: pd.DataFrame, symbol: str,
                   model_path: str = None):
    """Add a 'sentiment' column to df in-place."""
    try:
        from sentiment.pipeline import run_sentiment_pipeline, add_sentiment_feature
        start = str(df['date'].min().date())
        end   = str(df['date'].max().date())
        sent_df = run_sentiment_pipeline(symbol, start, end,
                                          price_df=df,
                                          model_path=model_path)
        merged = add_sentiment_feature(df, sent_df)
        df['sentiment'] = merged['sentiment'].values
        print(f"  [Sentiment] Added: mean={df['sentiment'].mean():.3f}")
    except Exception as e:
        print(f"  [Sentiment] Error: {e} — skipping.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Print consolidated results summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list):
    print(f"\n\n{'═'*75}")
    print("  SUMMARY OF RESULTS")
    print(f"{'═'*75}")
    print(f"  {'Symbol':<5} {'Model':<10} {'MAPE%':>7} {'DA%':>7} "
          f"{'Ret%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'B&H%':>7}")
    print(f"  {'-'*65}")
    for r in all_results:
        for name in ['LSTM', 'BiLSTM', 'CNN-LSTM', 'ARIMA', 'Ensemble']:
            em = r['error_metrics'][name]
            td = r['trading'][name]
            print(f"  {r['symbol']:<5} {name:<10} "
                  f"{em['mape']*100:>7.2f} {em['da']*100:>7.1f} "
                  f"{td['total_return']*100:>8.1f} {td['sharpe']:>7.2f} "
                  f"{td['max_drawdown']*100:>8.1f} {td['bh_return']*100:>7.1f}")
    print(f"{'═'*75}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Portfolio Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def run_portfolio(all_results: list, save_dir: str = 'plot'):
    """Run Markowitz Mean-Variance Optimization across all symbol results."""
    if len(all_results) < 2:
        print("\n  Portfolio: at least 2 symbols are required for optimization.")
        return

    try:
        from portfolio import (compute_expected_returns, compute_cov_matrix,
                                optimize_portfolio, monte_carlo_frontier,
                                plot_portfolio, print_portfolio_summary)

        print(f"\n{'#'*60}\n  PORTFOLIO OPTIMIZATION\n{'#'*60}")
        exp_rets   = compute_expected_returns(all_results)
        cov        = compute_cov_matrix(all_results)
        opt_result = optimize_portfolio(exp_rets, cov)
        mc         = monte_carlo_frontier(exp_rets, cov)

        print_portfolio_summary(opt_result, all_results)
        plot_portfolio(all_results, opt_result, mc, save_dir)
    except Exception as e:
        import traceback
        print(f"\n  Portfolio error: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Stock Prediction — Vietnamese Market (MSc)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                             # run all 5 symbols, single-step
  python main.py --symbols FPT VCB           # run 2 symbols
  python main.py --refresh                   # update data from vnstock
  python main.py --steps 5                   # forecast next 5 days
  python main.py --portfolio                 # add Markowitz MVO
  python main.py --sentiment                 # add sentiment feature (keyword-based)
  python main.py --sentiment --sentiment-model models/phobert_sentiment
        """,
    )
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                        help='Stock ticker symbols (default: all)')
    parser.add_argument('--refresh', action='store_true',
                        help='Update data from vnstock before training')
    parser.add_argument('--data-only', action='store_true',
                        help='Only update data, skip model training')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot using saved models, skip retraining')
    parser.add_argument('--portfolio-only', action='store_true',
                        help='Re-run portfolio optimization only, skip per-symbol plots')
    parser.add_argument('--save-dir', default='plot',
                        help='Directory to save charts (default: plot/)')
    parser.add_argument('--steps', type=int, default=None,
                        help=f'Number of forecast steps (default: {N_PREDICT_STEPS})')
    parser.add_argument('--portfolio', action='store_true',
                        help='Run Markowitz Portfolio Optimization after training')
    parser.add_argument('--sentiment', action='store_true',
                        help='Add sentiment feature from financial news')
    parser.add_argument('--sentiment-model', default=None,
                        help='Path to fine-tuned PhoBERT model '
                             '(default: keyword scoring)')
    args = parser.parse_args()

    # ── --portfolio-only: re-run portfolio plot using saved per-symbol results ──
    if args.portfolio_only:
        print("==> Re-running portfolio optimization using saved models...")
        all_results = []
        for symbol in args.symbols:
            try:
                result = run_symbol(symbol, save_dir=args.save_dir,
                                    refresh=False, n_steps=args.steps,
                                    use_sentiment=args.sentiment,
                                    sentiment_model=args.sentiment_model,
                                    plot_only=True)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR {symbol}: {e}")
        if all_results:
            run_portfolio(all_results, args.save_dir)
            print(f"\nDone! Portfolio plot → ./{args.save_dir}/portfolio_optimization.png")
        exit(0)

    # ── --data-only: refresh data then exit ──────────────────────────────────
    if args.data_only:
        print("==> Updating data from vnstock...")
        for symbol in args.symbols:
            try:
                update_data(symbol)
            except Exception as e:
                print(f"  ERROR {symbol}: {e}")
        print("\nDone! Data updated.")
        exit(0)

    # ── train / plot-only ─────────────────────────────────────────────────────
    all_results = []
    for symbol in args.symbols:
        try:
            result = run_symbol(
                symbol,
                save_dir=args.save_dir,
                refresh=args.refresh,
                n_steps=args.steps,
                use_sentiment=args.sentiment,
                sentiment_model=args.sentiment_model,
                plot_only=args.plot_only,
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n  ERROR {symbol}: {e}")
            traceback.print_exc()

    if all_results:
        print_summary(all_results)

        if args.portfolio:
            run_portfolio(all_results, args.save_dir)

        print(f"\nDone! Plots → ./{args.save_dir}/")
