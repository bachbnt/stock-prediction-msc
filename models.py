# models.py — LSTM, BiLSTM, CNN-LSTM, Ensemble
import numpy as np

from keras.models import Sequential
from keras.layers import (LSTM, Bidirectional, Conv1D, MaxPooling1D,
                           Dropout, Dense, BatchNormalization, Input)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2 as l2_reg

from config import MAX_EPOCHS, PATIENCE_ES, PATIENCE_LR, N_PREDICT_STEPS


def _callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_ES,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    ]


def build_lstm(input_shape, units=64, dropout=0.2, l2=1e-4,
               n_steps=None) -> Sequential:
    """2 lớp LSTM + BatchNorm + L2 regularization.
    n_steps: số bước dự đoán (None → dùng N_PREDICT_STEPS từ config).
    """
    n_steps = n_steps or N_PREDICT_STEPS
    m = Sequential(name='LSTM')
    m.add(Input(shape=input_shape))
    m.add(LSTM(units, return_sequences=True, kernel_regularizer=l2_reg(l2)))
    m.add(BatchNormalization()); m.add(Dropout(dropout))
    m.add(LSTM(units // 2, kernel_regularizer=l2_reg(l2)))
    m.add(BatchNormalization()); m.add(Dropout(dropout))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(n_steps))
    m.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return m


def build_bilstm(input_shape, units=64, dropout=0.2, l2=1e-4,
                 n_steps=None) -> Sequential:
    """Bidirectional LSTM — học pattern 2 chiều."""
    n_steps = n_steps or N_PREDICT_STEPS
    m = Sequential(name='BiLSTM')
    m.add(Input(shape=input_shape))
    m.add(Bidirectional(LSTM(units, return_sequences=True,
                              kernel_regularizer=l2_reg(l2))))
    m.add(BatchNormalization()); m.add(Dropout(dropout))
    m.add(Bidirectional(LSTM(units // 2, kernel_regularizer=l2_reg(l2))))
    m.add(BatchNormalization()); m.add(Dropout(dropout))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(n_steps))
    m.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return m


def build_cnnlstm(input_shape, filters=64, kernel_size=3,
                   lstm_units=64, dropout=0.2, l2=1e-4,
                   n_steps=None) -> Sequential:
    """
    CNN-LSTM Hybrid:
      Conv1D (x2) → trích xuất pattern cục bộ ngắn hạn
      MaxPooling   → giữ đặc trưng nổi bật
      LSTM         → học xu hướng dài hạn
    """
    n_steps = n_steps or N_PREDICT_STEPS
    m = Sequential(name='CNN-LSTM')
    m.add(Input(shape=input_shape))
    m.add(Conv1D(filters=filters, kernel_size=kernel_size,
                 activation='relu', padding='same', kernel_regularizer=l2_reg(l2)))
    m.add(Conv1D(filters=filters // 2, kernel_size=kernel_size,
                 activation='relu', padding='same', kernel_regularizer=l2_reg(l2)))
    m.add(MaxPooling1D(pool_size=2))
    m.add(Dropout(dropout))
    m.add(LSTM(lstm_units, kernel_regularizer=l2_reg(l2)))
    m.add(BatchNormalization()); m.add(Dropout(dropout))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(n_steps))
    m.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return m


def train_model(model: Sequential, X_train, y_train,
                batch_size=32, val_split=0.1) -> float:
    """Train model với EarlyStopping. Trả về best val_loss."""
    hist = model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=_callbacks(),
        verbose=0,
    )
    best_val = min(hist.history['val_loss'])
    n_ep     = len(hist.history['loss'])
    print(f"  [{model.name}] ep={n_ep} | best_val_loss={best_val:.5f}")
    return best_val


def ensemble_predict(predictions: dict, val_losses: dict):
    """
    Dynamic weighted ensemble: trọng số tỉ lệ nghịch với val_loss.
    Model val_loss thấp hơn → đóng góp nhiều hơn.
    Hỗ trợ cả single-step (1D) và multi-step (2D) predictions.
    """
    weights = {k: 1.0 / max(v, 1e-9) for k, v in val_losses.items()}
    total   = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    n    = len(next(iter(predictions.values())))
    pred = sum(weights[k] * predictions[k][:n] for k in predictions)

    print(f"  [Ensemble] Trọng số: " +
          ", ".join(f"{k}={v*100:.1f}%" for k, v in weights.items()))
    return pred, weights
