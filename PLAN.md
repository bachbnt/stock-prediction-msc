# Stock Prediction MSc — Kế hoạch mở rộng

## Trạng thái hiện tại ✅

| Module | Mô tả |
|--------|-------|
| `src/config.py` | Constants tập trung |
| `src/data_loader.py` | Load CSV, update vnstock, prepare sequences (single + multi-step) |
| `src/features.py` | 19 technical indicators |
| `src/models.py` | LSTM, BiLSTM, CNN-LSTM, dynamic ensemble |
| `src/evaluate.py` | RMSE, MAE, MAPE, DA, Sharpe, Drawdown, Calmar |
| `src/visualize.py` | 6-panel chart (incl. future forecast) |
| `src/portfolio.py` | Markowitz MVO, Monte Carlo frontier |
| `src/sentiment/` | News crawler + PhoBERT scorer + daily pipeline |
| `main.py` | CLI: `--refresh`, `--steps`, `--portfolio`, `--sentiment` |
| `scripts/` | `start.command`, `setup.command`, `refresh-data.command` |

### Chạy project
```bash
python main.py                          # 5 mã, 10 ngày forecast
python main.py --symbols FPT --steps 5  # 1 mã, 5 ngày
python main.py --refresh --portfolio    # cập nhật data + portfolio
python main.py --sentiment              # thêm sentiment feature
```

---

## Hướng mở rộng tiếp theo

---

### 1. Sentiment Analysis — hoàn thiện PhoBERT ⏳
*Đã có crawler + scorer, cần thêm data có nhãn*

**Bước cần làm thủ công:**
1. Chạy `python main.py --sentiment` để crawl và cache tin tức vào `data/news/`
2. Gán nhãn ~500–1000 câu bằng [Label Studio](https://labelstud.io) (free, local)
   - `1` = tích cực, `0` = trung tính, `-1` = tiêu cực
   - Lưu vào `data/labels/<symbol>_labeled.csv` (2 cột: `text`, `label`)
3. Fine-tune PhoBERT (cần GPU hoặc Apple Silicon MPS, ~30 phút):
   ```python
   from src.sentiment.scorer import fine_tune_phobert
   fine_tune_phobert('data/labels/FPT_labeled.csv', 'models/phobert_sentiment')
   ```
4. Chạy so sánh: 19 features vs 20 features (+ sentiment) → đo MAPE, DA, Sharpe

**Giá trị thesis:** Đây là điểm novelty cao nhất — ít paper Việt Nam kết hợp NLP + LSTM cho HOSE.

---

### 2. Transformer / TFT 🔲
*Kiến trúc hiện đại nhất cho time series, thường vượt LSTM 10–20%*

**Temporal Fusion Transformer (TFT)** — Google DeepMind 2020:
- Attention mechanism học được feature nào quan trọng theo thời gian
- Xử lý tốt missing data và multi-horizon forecast
- Output có **khoảng tin cậy** (quantile prediction) — rất phù hợp thesis

```bash
pip install pytorch-forecasting pytorch-lightning
```

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Thêm vào src/models.py hoặc tạo src/models_tft.py riêng
# So sánh TFT vs LSTM Ensemble trong visualize.py
```

**Thứ tự ưu tiên:** Cao — output có CI bands trông rất academic.

---

### 3. Hyperparameter Tuning 🔲
*Tự động tìm units, dropout, learning rate tối ưu*

```bash
pip install optuna
```

```python
import optuna

def objective(trial):
    units   = trial.suggest_int('units', 32, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    l2      = trial.suggest_float('l2', 1e-5, 1e-3, log=True)
    model   = build_lstm(input_shape, units=units, dropout=dropout, l2=l2)
    return train_model(model, X_train, y_train)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

Tạo file `src/tuning.py`, thêm flag `--tune` vào `main.py`.

---

### 4. Walk-Forward Validation 🔲
*Đánh giá thực tế hơn single train/test split*

Hiện tại: train 2015–2020, test 2021–2022 (1 lần duy nhất).

Walk-forward: train trên 3 năm, test 3 tháng, trượt dần → nhiều fold → kết quả tin cậy hơn cho thesis.

```python
# src/evaluate.py — thêm hàm walk_forward_backtest()
folds = [
    ('2015-2018', '2019'),
    ('2015-2019', '2020'),
    ('2015-2020', '2021'),
    ('2015-2021', '2022'),
]
# Average MAPE/Sharpe qua các fold
```

---

### 5. Macro Features 🔲
*Thêm dữ liệu vĩ mô vào feature set*

| Feature | Nguồn | Ý nghĩa |
|---------|-------|---------|
| Tỷ giá USD/VND | NHNN / vnstock | Ảnh hưởng cổ phiếu xuất khẩu |
| Giá vàng SJC | SJC API | Tương quan ngược với cổ phiếu |
| VN-Index | vnstock | Sentiment thị trường tổng thể |
| Lãi suất liên ngân hàng | NHNN | Ảnh hưởng cổ phiếu ngân hàng |
| Giá dầu Brent | Yahoo Finance | Ảnh hưởng HPG, ngành năng lượng |

```python
# src/macro_features.py
import yfinance as yf
oil = yf.download('BZ=F', start='2015-01-01')['Close']
```

---

### 6. Reinforcement Learning Trading 🔲
*Agent tự học chiến lược giao dịch, không cần rule-based*

```bash
pip install stable-baselines3 gym
```

Thay chiến lược long-only hiện tại (mua khi predict tăng) bằng RL agent:
- **State**: window 60 ngày features + portfolio state
- **Action**: BUY / HOLD / SELL
- **Reward**: daily return - transaction cost

So sánh RL vs rule-based strategy trong `evaluate.py`.

---

### 7. Statistical Significance Tests 🔲
*Bắt buộc nếu publish paper*

**Diebold-Mariano test**: kiểm định 2 model có khác biệt thống kê không.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Null hypothesis: 2 models có accuracy như nhau
# p < 0.05 → model A tốt hơn model B có ý nghĩa thống kê
```

**Thêm vào `src/evaluate.py`**: hàm `dm_test(pred1, pred2, real)`.

---

### 8. Ablation Study 🔲
*Chứng minh đóng góp của từng thành phần — quan trọng cho thesis*

Chạy 6 lần, mỗi lần bỏ 1 thứ:

| Experiment | Bỏ gì | MAPE | DA |
|-----------|-------|------|-----|
| Full model | — | ? | ? |
| No sentiment | bỏ feature 20 | ? | ? |
| No MACD | bỏ macd/signal/hist | ? | ? |
| No Bollinger | bỏ bb_width/bb_pct | ? | ? |
| No ensemble | chỉ LSTM | ? | ? |
| ARIMA only | baseline | ? | ? |

---

## Thứ tự ưu tiên đề xuất

| # | Việc | Độ khó | Giá trị thesis |
|---|------|--------|---------------|
| 1 | Sentiment fine-tune PhoBERT | Trung bình | ⭐⭐⭐⭐⭐ |
| 2 | Transformer / TFT | Cao | ⭐⭐⭐⭐⭐ |
| 3 | Ablation study | Thấp | ⭐⭐⭐⭐ |
| 4 | Walk-forward validation | Thấp | ⭐⭐⭐⭐ |
| 5 | Statistical significance tests | Thấp | ⭐⭐⭐⭐ |
| 6 | Hyperparameter tuning | Trung bình | ⭐⭐⭐ |
| 7 | Macro features | Trung bình | ⭐⭐⭐ |
| 8 | Reinforcement Learning | Cao | ⭐⭐⭐ |

---

## Ghi chú kỹ thuật

- vnstock 4.x: dùng `Quote(symbol, source).history()` — không dùng `stock_historical_data()`
- PhoBERT fine-tune: dùng Google Colab (A100 free) nếu máy không có GPU
- TFT cần PyTorch — tách riêng khỏi TensorFlow environment hoặc dùng conda
- Apple Silicon: `tensorflow-metal` cho TF, `torch` tự dùng MPS không cần cài thêm
- Mỗi ablation experiment lưu kết quả ra `results/<experiment_name>.json` để so sánh
