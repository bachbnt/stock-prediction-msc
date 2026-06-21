# thesis.py — Entry point
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from config import SYMBOLS, DATE_MILESTONE, WINDOW_SIZE, BATCH_SIZE, FEATURE_COLS, SEED
from data_loader import load_data, update_data, prepare_data, inverse_close
from features import add_features
from models import build_lstm, build_bilstm, build_cnnlstm, train_model, ensemble_predict
from evaluate import compute_error_metrics, compute_trading_metrics
from visualize import plot_symbol

np.random.seed(SEED)
tf.random.set_seed(SEED)


def run_symbol(symbol: str, save_dir='plot', refresh=False) -> dict:
    """
    Pipeline đầy đủ cho 1 mã:
      refresh=True  → cập nhật CSV từ vnstock trước khi chạy
      refresh=False → dùng CSV local có sẵn
    """
    print(f"\n{'#'*60}\n  {symbol}\n{'#'*60}")

    # 1. Load data
    df = update_data(symbol) if refresh else load_data(symbol)
    df = add_features(df)
    print(f"  Features: {len(FEATURE_COLS)} | Rows: {len(df)}")

    # 2. Prepare sequences
    X_train, y_train, X_test, y_test, scaler, df_train, df_test = \
        prepare_data(df, DATE_MILESTONE, WINDOW_SIZE)
    input_shape = (WINDOW_SIZE, len(FEATURE_COLS))
    real_price  = inverse_close(y_test, scaler)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    predictions, error_metrics, trading, val_losses = {}, {}, {}, {}

    # 3. Train models
    for name, model in [
        ('LSTM',     build_lstm(input_shape)),
        ('BiLSTM',   build_bilstm(input_shape)),
        ('CNN-LSTM', build_cnnlstm(input_shape)),
    ]:
        print(f"\n  Training {name}...")
        vl   = train_model(model, X_train, y_train, BATCH_SIZE)
        pred = inverse_close(model.predict(X_test, verbose=0), scaler)
        predictions[name]   = pred
        val_losses[name]    = vl
        error_metrics[name] = compute_error_metrics(real_price, pred, name)
        trading[name]       = compute_trading_metrics(real_price, pred, name)

    # 4. ARIMA baseline
    from statsmodels.tsa.arima.model import ARIMA
    print(f"\n  ARIMA(5,1,0) walk-forward...")
    history = list(df_train['close'].values)
    arima_preds = []
    for actual in df_test['close'].values[:len(real_price)]:
        try:
            yhat = float(ARIMA(history, order=(5,1,0)).fit().forecast(1).iloc[0])
        except Exception:
            yhat = history[-1]
        arima_preds.append(yhat)
        history.append(actual)
    pred_arima = np.array(arima_preds)
    arima_proxy_loss = float(np.mean(np.diff(df_train['close'].values[-50:]) ** 2))
    predictions['ARIMA']   = pred_arima
    val_losses['ARIMA']    = arima_proxy_loss
    error_metrics['ARIMA'] = compute_error_metrics(real_price, pred_arima, 'ARIMA')
    trading['ARIMA']       = compute_trading_metrics(real_price, pred_arima, 'ARIMA')

    # 5. Ensemble (3 deep learning models)
    dl_preds  = {k: predictions[k] for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    dl_losses = {k: val_losses[k]  for k in ['LSTM', 'BiLSTM', 'CNN-LSTM']}
    pred_ens, _ = ensemble_predict(dl_preds, dl_losses)
    predictions['Ensemble']   = pred_ens
    error_metrics['Ensemble'] = compute_error_metrics(real_price, pred_ens, 'Ensemble')
    trading['Ensemble']       = compute_trading_metrics(real_price, pred_ens, 'Ensemble')

    # 6. Plot
    plot_symbol(symbol, df_test, real_price,
                predictions, trading, error_metrics, save_dir)

    # 7. Summary
    bh = list(trading.values())[0]['bh_return']
    print(f"\n  Buy & Hold: {bh*100:.1f}%")

    return dict(symbol=symbol, error_metrics=error_metrics,
                trading=trading, predictions=predictions)


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stock Prediction — MSc Thesis')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                        help='Mã cổ phiếu cần chạy (mặc định: tất cả)')
    parser.add_argument('--refresh', action='store_true',
                        help='Cập nhật dữ liệu từ vnstock trước khi train')
    parser.add_argument('--save-dir', default='plot',
                        help='Thư mục lưu chart (mặc định: plot/)')
    args = parser.parse_args()

    all_results = []
    for symbol in args.symbols:
        try:
            all_results.append(run_symbol(symbol, args.save_dir, args.refresh))
        except Exception as e:
            import traceback
            print(f"\n  ERROR {symbol}: {e}")
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
        print(f"\nDone! Plots → ./{args.save_dir}/")
