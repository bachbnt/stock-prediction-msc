# models_research.py — Models for the next-day LOG-RETURN research pipeline
"""
Every model consumes standardised windows (n, window, n_features) and returns a
forecast of the next-day log return in RAW units (the scaled network output is
multiplied back by fd.y_scale).

Families
  Point forecasters (MSE)  : LSTM, BiLSTM, CNN-LSTM, GRU, TCN, Transformer
  Probabilistic            : LSTM-MC  (Monte-Carlo dropout)
                             LSTM-Q   (quantile / pinball loss)
                             TFT-lite (variable selection + LSTM + causal attention,
                                       quantile outputs — a Keras approximation of
                                       the Temporal Fusion Transformer, NOT the
                                       pytorch-forecasting implementation)
  Direction                : LSTM-Dir (sigmoid head, binary cross-entropy)
  Baselines                : Naive (zero return / random walk), Drift, ARIMA(5,1,0),
                             HistGB (gradient boosting on lagged tabular features)
"""
import numpy as np

import keras
from keras import layers, regularizers, ops
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models import build_lstm, build_bilstm, build_cnnlstm
from research_data import tabular_features

DL_POINT = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'GRU', 'TCN', 'Transformer']
DL_PROB = ['LSTM-MC', 'LSTM-Q', 'TFT-lite']
DL_DIR = ['LSTM-Dir']
BASELINES = ['Naive', 'Drift', 'ARIMA', 'HistGB']
ALL_MODELS = DL_POINT + DL_PROB + DL_DIR + BASELINES
DEFAULT_MODELS = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'GRU', 'TCN', 'Transformer',
                  'LSTM-MC', 'LSTM-Q', 'TFT-lite', 'LSTM-Dir',
                  'Naive', 'Drift', 'ARIMA', 'HistGB']


# ═══════════════════════════════════════════════════════════════════════════════
# Builders
# ═══════════════════════════════════════════════════════════════════════════════
def _compile(m, loss='mse', lr=1e-3):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss=loss)
    return m


def build_gru(input_shape, units=64, dropout=0.2, l2=1e-4, lr=1e-3):
    m = keras.Sequential(name='GRU')
    m.add(layers.Input(shape=input_shape))
    m.add(layers.GRU(units, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.BatchNormalization()); m.add(layers.Dropout(dropout))
    m.add(layers.GRU(units // 2, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.BatchNormalization()); m.add(layers.Dropout(dropout))
    m.add(layers.Dense(32, activation='relu'))
    m.add(layers.Dense(1))
    return _compile(m, lr=lr)


def build_tcn(input_shape, filters=32, kernel_size=3, dilations=(1, 2, 4, 8, 16),
              dropout=0.2, l2=1e-4, lr=1e-3):
    """Temporal convolutional network: stacked causal dilated residual blocks."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters, 1)(inp)
    for d in dilations:
        h = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=d,
                          activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        h = layers.SpatialDropout1D(dropout)(h)
        h = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=d,
                          activation='relu', kernel_regularizer=regularizers.l2(l2))(h)
        x = layers.LayerNormalization()(layers.Add()([x, h]))
    last = layers.Flatten()(layers.Cropping1D((input_shape[0] - 1, 0))(x))
    x = layers.Concatenate()([last, layers.GlobalAveragePooling1D()(x)])
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    return _compile(keras.Model(inp, out, name='TCN'), lr=lr)


class PositionalEmbedding(layers.Layer):
    def __init__(self, length, dim, **kw):
        super().__init__(**kw)
        self.length, self.dim = length, dim
        self.emb = layers.Embedding(length, dim)

    def call(self, x):
        pos = ops.arange(self.length)
        return x + self.emb(pos)[None, :, :]


def build_transformer(input_shape, d_model=32, heads=4, blocks=2, ff=64,
                      dropout=0.2, l2=1e-4, lr=1e-3):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inp)
    x = PositionalEmbedding(input_shape[0], d_model)(x)
    for _ in range(blocks):
        att = layers.MultiHeadAttention(heads, d_model // heads, dropout=dropout)(x, x, use_causal_mask=True)
        x = layers.LayerNormalization()(layers.Add()([x, att]))
        f = layers.Dense(ff, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        f = layers.Dense(d_model)(layers.Dropout(dropout)(f))
        x = layers.LayerNormalization()(layers.Add()([x, f]))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(layers.Dropout(dropout)(x))
    out = layers.Dense(1)(x)
    return _compile(keras.Model(inp, out, name='Transformer'), lr=lr)


def build_lstm_mc(input_shape, units=64, dropout=0.3, l2=1e-4, lr=1e-3):
    """LSTM WITHOUT BatchNorm so that dropout can stay active at inference (MC dropout)."""
    m = keras.Sequential(name='LSTM-MC')
    m.add(layers.Input(shape=input_shape))
    m.add(layers.LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.LSTM(units // 2, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(32, activation='relu'))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1))
    return _compile(m, lr=lr)


def pinball_loss(quantiles):
    q = np.asarray(quantiles, dtype='float32')

    def loss(y_true, y_pred):
        y = ops.reshape(y_true, (-1, 1))
        err = y - y_pred
        return ops.mean(ops.maximum(q * err, (q - 1.0) * err))
    return loss


def build_lstm_quantile(input_shape, quantiles, units=64, dropout=0.2, l2=1e-4, lr=1e-3):
    m = keras.Sequential(name='LSTM-Q')
    m.add(layers.Input(shape=input_shape))
    m.add(layers.LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.LSTM(units // 2, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(32, activation='relu'))
    m.add(layers.Dense(len(quantiles)))
    return _compile(m, loss=pinball_loss(quantiles), lr=lr)


def build_tft_lite(input_shape, quantiles, d_model=32, heads=4, dropout=0.2, lr=1e-3):
    """
    TFT-style network: (1) soft variable selection per time step,
    (2) LSTM encoder for local processing, (3) causal multi-head self-attention with
    a gated residual, (4) quantile heads. The softmax weights of layer 'var_select'
    are exposed through `variable_importance` for interpretation.
    """
    T, F = input_shape
    inp = layers.Input(shape=input_shape)
    g = layers.Dense(d_model, activation='elu')(inp)
    w = layers.Dense(F, activation='softmax', name='var_select')(g)
    sel = layers.Multiply()([inp, layers.Rescaling(float(F))(w)])
    h = layers.Dense(d_model)(sel)
    h = layers.LSTM(d_model, return_sequences=True)(h)
    att = layers.MultiHeadAttention(heads, d_model // heads, dropout=dropout)(h, h, use_causal_mask=True)
    gate = layers.Dense(d_model, activation='sigmoid')(att)
    h = layers.LayerNormalization()(layers.Add()([h, layers.Multiply()([gate, att])]))
    last = layers.Flatten()(layers.Cropping1D((T - 1, 0))(h))
    x = layers.Dense(32, activation='relu')(layers.Dropout(dropout)(last))
    out = layers.Dense(len(quantiles))(x)
    return _compile(keras.Model(inp, out, name='TFT-lite'), loss=pinball_loss(quantiles), lr=lr)


def build_lstm_direction(input_shape, units=64, dropout=0.3, l2=1e-4, lr=1e-3):
    m = keras.Sequential(name='LSTM-Dir')
    m.add(layers.Input(shape=input_shape))
    m.add(layers.LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.LSTM(units // 2, kernel_regularizer=regularizers.l2(l2)))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(16, activation='relu'))
    m.add(layers.Dense(1, activation='sigmoid'))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy')
    return m


def variable_importance(model, X, feature_cols) -> dict:
    """Mean soft-selection weight per feature (TFT-lite only)."""
    sub = keras.Model(model.input, model.get_layer('var_select').output)
    w = sub.predict(X, verbose=0, batch_size=256)           # (n, T, F)
    imp = w.mean(axis=(0, 1))
    return dict(zip(feature_cols, imp.tolist()))


# ═══════════════════════════════════════════════════════════════════════════════
# Training / prediction helpers
# ═══════════════════════════════════════════════════════════════════════════════
def fit_keras(model, X, y, cfg, val_fraction=0.1) -> float:
    """Chronological validation (Keras validation_split takes the LAST fraction)."""
    hist = model.fit(
        X, y, epochs=cfg['max_epochs'], batch_size=cfg['batch_size'],
        validation_split=val_fraction, verbose=0, shuffle=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=cfg['patience_es'],
                                 restore_best_weights=True, verbose=0),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=cfg['patience_lr'], min_lr=1e-6, verbose=0)],
    )
    return float(min(hist.history['val_loss']))


def mc_predict(model, X, n_samples=30, batch_size=256):
    """Monte-Carlo dropout: n stochastic forward passes -> (mean, std)."""
    draws = []
    for _ in range(n_samples):
        outs = [np.asarray(model(X[i:i + batch_size], training=True)).reshape(-1)
                for i in range(0, len(X), batch_size)]
        draws.append(np.concatenate(outs))
    draws = np.stack(draws)
    return draws.mean(axis=0), draws.std(axis=0)


def fit_predict(name: str, fd, cfg: dict, seed: int, hparams: dict = None,
                train_close: np.ndarray = None) -> dict:
    """
    Train `name` on fold data and return
        dict(pred=raw log-return forecast, val_loss, [lo, hi, std, p_up, importance])
    """
    hparams = hparams or {}
    n_train, T, F = fd.X_train.shape
    shape = (T, F)
    out = {}

    if name in ('Naive', 'Drift'):
        pred = np.zeros(len(fd.y_test)) if name == 'Naive' else \
            np.full(len(fd.y_test), float(np.mean(fd.y_train) * fd.y_scale))
        return dict(pred=pred, val_loss=float('nan'))

    if name == 'ARIMA':
        if train_close is None:
            raise ValueError("ARIMA needs train_close (closes up to the last training day)")
        if abs(train_close[-1] - fd.prev_close[0]) > 1e-6 * abs(fd.prev_close[0]):
            raise ValueError("train_close must end at the close preceding the first test target")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            res = ARIMA(np.log(train_close), order=(5, 1, 0)).fit()
            preds = []
            for prev, real in zip(fd.prev_close, fd.real_close):
                preds.append(float(np.asarray(res.forecast(1))[0]) - np.log(prev))
                # observe the realised close, update state without re-estimating
                res = res.append([np.log(real)], refit=False)
            return dict(pred=np.array(preds), val_loss=float('nan'))
        except Exception as exc:                                    # convergence problems
            print(f"    [ARIMA] failed ({str(exc)[:60]}) -> naive")
            return dict(pred=np.zeros(len(fd.y_test)), val_loss=float('nan'))

    if name == 'HistGB':
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xtr = tabular_features(fd.flat_train, fd.X_train)
        Xte = tabular_features(fd.flat_test, fd.X_test)
        gb = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=hparams.get('max_iter', 150), learning_rate=0.05,
            max_depth=3, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
        gb.fit(Xtr, fd.y_train)
        return dict(pred=gb.predict(Xte) * fd.y_scale, val_loss=float('nan'))

    # ── deep models ───────────────────────────────────────────────────────────
    keras.utils.set_random_seed(seed)
    keras.backend.clear_session()
    lr = hparams.get('lr', 1e-3)
    units = hparams.get('units', 64)
    dropout = hparams.get('dropout', 0.2)
    l2 = hparams.get('l2', 1e-4)
    qs = cfg['quantiles']

    if name == 'LSTM':
        m = build_lstm(shape, units=units, dropout=dropout, l2=l2, n_steps=1)
    elif name == 'BiLSTM':
        m = build_bilstm(shape, units=units, dropout=dropout, l2=l2, n_steps=1)
    elif name == 'CNN-LSTM':
        m = build_cnnlstm(shape, lstm_units=units, dropout=dropout, l2=l2, n_steps=1)
    elif name == 'GRU':
        m = build_gru(shape, units=units, dropout=dropout, l2=l2, lr=lr)
    elif name == 'TCN':
        m = build_tcn(shape, dropout=dropout, l2=l2, lr=lr)
    elif name == 'Transformer':
        m = build_transformer(shape, dropout=dropout, l2=l2, lr=lr)
    elif name == 'LSTM-MC':
        m = build_lstm_mc(shape, units=units, dropout=max(dropout, 0.2), l2=l2, lr=lr)
    elif name == 'LSTM-Q':
        m = build_lstm_quantile(shape, qs, units=units, dropout=dropout, l2=l2, lr=lr)
    elif name == 'TFT-lite':
        m = build_tft_lite(shape, qs, dropout=dropout, lr=lr)
    elif name == 'LSTM-Dir':
        m = build_lstm_direction(shape, units=units, dropout=max(dropout, 0.2), l2=l2, lr=lr)
    else:
        raise ValueError(f"unknown model {name!r}")

    y_fit = (fd.y_train > 0).astype('float32') if name == 'LSTM-Dir' else fd.y_train
    val_loss = fit_keras(m, fd.X_train, y_fit, cfg)

    if name == 'LSTM-MC':
        mean, std = mc_predict(m, fd.X_test, cfg['mc_samples'])
        z = 1.2816                                              # 80% interval
        pred = mean * fd.y_scale
        out.update(std=std * fd.y_scale, lo=pred - z * std * fd.y_scale,
                   hi=pred + z * std * fd.y_scale)
    elif name in ('LSTM-Q', 'TFT-lite'):
        q = np.sort(m.predict(fd.X_test, verbose=0, batch_size=256), axis=1) * fd.y_scale
        mid = int(np.argmin(np.abs(np.array(qs) - 0.5)))
        pred = q[:, mid]
        out.update(lo=q[:, 0], hi=q[:, -1])
        if name == 'TFT-lite':
            out['importance'] = variable_importance(m, fd.X_test, fd.feature_cols)
    elif name == 'LSTM-Dir':
        p_up = m.predict(fd.X_test, verbose=0, batch_size=256).reshape(-1)
        out['p_up'] = p_up
        # pseudo expected return: signed confidence x current volatility
        pred = (2 * p_up - 1) * fd.vol
    else:
        pred = m.predict(fd.X_test, verbose=0, batch_size=256).reshape(-1) * fd.y_scale

    out.update(pred=pred, val_loss=val_loss)
    return out


def ensemble_returns(preds: dict, val_losses: dict):
    """Inverse-validation-loss weighted average of return forecasts."""
    w = {k: 1.0 / max(val_losses[k], 1e-9) for k in preds}
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    n = len(next(iter(preds.values())))
    return sum(w[k] * preds[k][:n] for k in preds), w
