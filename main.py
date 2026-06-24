# main.py — Entry point: Vietnamese Stock Price Prediction (MSc Thesis)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from config import (SYMBOLS, DATE_MILESTONE, WINDOW_SIZE, BATCH_SIZE,
                    FEATURE_COLS, SEED, N_PREDICT_STEPS)
from data_loader import load_data, update_data, prepare_data, inverse_close
from features    import add_features
from models      import build_lstm, build_bilstm, build_cnnlstm, train_model, ensemble_predict
from evaluate    import compute_error_metrics, compute_trading_metrics
from visualize   import plot_symbol

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pipeline cho 1 mã
# ═══════════════════════════════════════════════════════════════════════════════

def run_symbol(symbol: str, save_dir: str = 'plot',
               refresh: bool = False,
               n_steps: int = None,
               use_sentiment: bool = False,
               sentiment_model: str = None) -> dict:
    """
    Pipeline đầy đủ cho 1 mã cổ phiếu.

    Args:
        symbol:          Mã cổ phiếu (e.g. 'FPT')
        save_dir:        Thư mục lưu chart
        refresh:         True → cập nhật CSV từ vnstock trước khi chạy
        n_steps:         Số bước dự đoán (None → dùng N_PREDICT_STEPS từ config)
        use_sentiment:   True → thêm sentiment feature từ tin tức
        sentiment_model: Path đến fine-tuned PhoBERT model (None → keyword fallback)
    """
    if n_steps is None:
        n_steps = N_PREDICT_STEPS

    print(f"\n{'#'*60}\n  {symbol}  (n_steps={n_steps})\n{'#'*60}")

    # 1. Load data
    df = update_data(symbol) if refresh else load_data(symbol)
    df = add_features(df)

    # 2. Sentiment feature (optional)
    if use_sentiment:
        _add_sentiment(df, symbol, sentiment_model)

    feature_cols = list(FEATURE_COLS)
    if use_sentiment and 'sentiment' in df.columns:
        if 'sentiment' not in feature_cols:
            feature_cols.append('sentiment')
        print(f"  Features: {len(feature_cols)} (incl. sentiment) | Rows: {len(df)}")
    else:
        print(f"  Features: {len(feature_cols)} | Rows: {len(df)}")

    # 3. Prepare sequences
    X_train, y_train, X_test, y_test, scaler, df_train, df_test = \
        prepare_data(df, DATE_MILESTONE, WINDOW_SIZE, n_steps)
    input_shape = (WINDOW_SIZE, len(feature_cols))

    # real_price: dùng step-1 để evaluate và visualize
    if n_steps == 1:
        real_price = inverse_close(y_test, scaler)
    else:
        # Dùng step-1 prediction của mỗi window làm ground truth
        real_price = inverse_close(y_test[:, 0], scaler)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    predictions, error_metrics, trading, val_losses = {}, {}, {}, {}

    # 4. Train models
    for name, model in [
        ('LSTM',     build_lstm(input_shape, n_steps=n_steps)),
        ('BiLSTM',   build_bilstm(input_shape, n_steps=n_steps)),
        ('CNN-LSTM', build_cnnlstm(input_shape, n_steps=n_steps)),
    ]:
        print(f"\n  Training {name}...")
        vl   = train_model(model, X_train, y_train, BATCH_SIZE)

        raw_pred = model.predict(X_test, verbose=0)
        if n_steps == 1:
            pred = inverse_close(raw_pred, scaler)
        else:
            pred = inverse_close(raw_pred[:, 0], scaler)   # step-1 for evaluation

        predictions[name]   = pred
        val_losses[name]    = vl
        error_metrics[name] = compute_error_metrics(real_price, pred, name)
        trading[name]       = compute_trading_metrics(real_price, pred, name)

    # 5. ARIMA baseline
    from statsmodels.tsa.arima.model import ARIMA
    print(f"\n  ARIMA(5,1,0) walk-forward...")
    history     = list(df_train['close'].values)
    arima_preds = []
    for actual in df_test['close'].values[:len(real_price)]:
        try:
            yhat = float(ARIMA(history, order=(5, 1, 0)).fit().forecast(1).iloc[0])
        except Exception:
            yhat = history[-1]
        arima_preds.append(yhat)
        history.append(actual)
    pred_arima         = np.array(arima_preds)
    arima_proxy_loss   = float(np.mean(np.diff(df_train['close'].values[-50:]) ** 2))
    predictions['ARIMA']   = pred_arima
    val_losses['ARIMA']    = arima_proxy_loss
    error_metrics['ARIMA'] = compute_error_metrics(real_price, pred_arima, 'ARIMA')
    trading['ARIMA']       = compute_trading_metrics(real_price, pred_arima, 'ARIMA')

    # 6. Ensemble (3 deep learning models)
    dl_preds  = {k: predictions[k] for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    dl_losses = {k: val_losses[k]  for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    pred_ens, _ = ensemble_predict(dl_preds, dl_losses)
    predictions['Ensemble']   = pred_ens
    error_metrics['Ensemble'] = compute_error_metrics(real_price, pred_ens, 'Ensemble')
    trading['Ensemble']       = compute_trading_metrics(real_price, pred_ens, 'Ensemble')

    # 7. Plot
    plot_symbol(symbol, df_test, real_price,
                predictions, trading, error_metrics, save_dir)

    # 8. Summary
    bh = list(trading.values())[0]['bh_return']
    print(f"\n  Buy & Hold: {bh*100:.1f}%")

    return dict(
        symbol=symbol,
        real_price=real_price,
        error_metrics=error_metrics,
        trading=trading,
        predictions=predictions,
    )


def _add_sentiment(df: pd.DataFrame, symbol: str,
                   model_path: str = None):
    """Thêm cột sentiment vào df (in-place)."""
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
        print(f"  [Sentiment] Lỗi: {e} — bỏ qua.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Print summary bảng tổng hợp
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list):
    print(f"\n\n{'═'*75}")
    print("  KẾT QUẢ TỔNG HỢP")
    print(f"{'═'*75}")
    print(f"  {'Mã':<5} {'Model':<10} {'MAPE%':>7} {'DA%':>7} "
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
    """Chạy Markowitz MVO trên kết quả của tất cả mã."""
    if len(all_results) < 2:
        print("\n  Portfolio: cần ít nhất 2 mã để tối ưu.")
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
        print(f"\n  Portfolio lỗi: {e}")
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
Ví dụ:
  python main.py                             # chạy tất cả 5 mã, single-step
  python main.py --symbols FPT VCB           # chạy 2 mã
  python main.py --refresh                   # cập nhật data từ vnstock
  python main.py --steps 5                   # dự đoán 5 ngày tới
  python main.py --portfolio                 # thêm Markowitz MVO
  python main.py --sentiment                 # thêm sentiment feature (keyword)
  python main.py --sentiment --sentiment-model models/phobert_sentiment
        """,
    )
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                        help='Mã cổ phiếu (mặc định: tất cả)')
    parser.add_argument('--refresh', action='store_true',
                        help='Cập nhật data từ vnstock trước khi train')
    parser.add_argument('--save-dir', default='plot',
                        help='Thư mục lưu chart (mặc định: plot/)')
    parser.add_argument('--steps', type=int, default=None,
                        help=f'Số bước dự đoán (mặc định: {N_PREDICT_STEPS})')
    parser.add_argument('--portfolio', action='store_true',
                        help='Chạy Markowitz Portfolio Optimization sau khi train')
    parser.add_argument('--sentiment', action='store_true',
                        help='Thêm sentiment feature từ tin tức tài chính')
    parser.add_argument('--sentiment-model', default=None,
                        help='Path đến fine-tuned PhoBERT model '
                             '(mặc định: keyword scoring)')
    args = parser.parse_args()

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
