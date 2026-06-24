# Stock Prediction MSc — Kế hoạch mở rộng

## Đã hoàn thành

### Codebase
- `config.py` — constants tập trung (symbols, features, hyperparams, tax, portfolio)
- `data_loader.py` — load CSV local / GitHub, update từ vnstock, prepare sequences (single + multi-step)
- `features.py` — 19 technical indicators (RSI, MACD, Bollinger, SMA/EMA, Volume)
- `models.py` — LSTM, BiLSTM, CNN-LSTM, dynamic ensemble (hỗ trợ n_steps)
- `evaluate.py` — RMSE, MAE, MAPE, DA, Sharpe, Max Drawdown, Calmar, B&H
- `visualize.py` — chart 5 panel: price prediction, portfolio, drawdown, MAPE/DA, Return/Sharpe
- `portfolio.py` — Markowitz MVO: Min-Variance, Max-Sharpe, Equal-Weight, Monte Carlo frontier
- `sentiment/` — News crawler (Google News + VnExpress) + PhoBERT scorer + daily pipeline
- `main.py` — entry point với CLI args (`--symbols`, `--refresh`, `--steps`, `--portfolio`, `--sentiment`)

### Fixes so với code gốc
- Bug: train dùng `Open`, test dùng `Close` → đã thống nhất `Close`
- Thêm `validation_split=0.1` + `EarlyStopping` + `ReduceLROnPlateau`
- Giảm từ 4 lớp LSTM xuống 2 + `BatchNormalization` + L2 → giảm overfit
- Scaler fit chỉ trên train data → tránh data leakage
- Random seed 42 → tái lập kết quả

### Chạy project
```bash
bash setup.sh                          # setup venv lần đầu
source venv/bin/activate

python main.py                         # chạy tất cả 5 mã, single-step
python main.py --symbols FPT VCB       # chạy 2 mã
python main.py --refresh               # cập nhật data từ vnstock trước khi train
python main.py --steps 5               # multi-step: dự đoán 5 ngày tới
python main.py --portfolio             # thêm Markowitz Portfolio Optimization
python main.py --sentiment             # thêm sentiment feature (keyword scoring)
python main.py --sentiment --sentiment-model models/phobert_sentiment
```

---

## 1. Cập nhật dữ liệu ✅

CSV hiện tại kết thúc 31/05/2022, thiếu ~4 năm. `update_data()` trong `data_loader.py` đã viết sẵn.

```bash
pip install vnstock>=4.0.0
python main.py --refresh   # tự động fetch và nối vào CSV
```

**Cú pháp vnstock 4.x:**
```python
from vnstock import Quote
quote = Quote(symbol='FPT', source='VCI')
df = quote.history(start='2022-06-01', end='2026-06-01', interval='1D')
# cột trả về: time, open, high, low, close, volume (lowercase)
```

---

## 2. Multi-step Prediction ✅

Đã implement: `--steps N` để dự đoán N ngày tới.

```bash
python main.py --steps 5    # dự đoán 5 ngày
python main.py --steps 10   # dự đoán 10 ngày
```

Thay đổi đã thực hiện:
- `config.py`: thêm `N_PREDICT_STEPS = 1` (default single-step)
- `data_loader.py`: `_make_sequences()` hỗ trợ `n_steps`, y shape là `(N, n_steps)` khi >1
- `models.py`: `Dense(n_steps)` thay vì `Dense(1)`
- `main.py`: nhận `--steps`, lấy step-1 để evaluate/visualize

---

## 3. Sentiment Analysis ✅ (cần chạy thêm bước thủ công)

Module `sentiment/` đã viết hoàn chỉnh. Cần thêm 2 bước thủ công để đạt chất lượng cao nhất:

### Bước 1 — Thu thập tin tức (tự động)
```bash
python main.py --sentiment   # tự crawl và cache vào data/news/<symbol>/
```

### Bước 2 — Gán nhãn (thủ công, để fine-tune PhoBERT)
Cần ~500–1000 câu được gán nhãn: `1` (tích cực), `0` (trung tính), `-1` (tiêu cực)

Tool gán nhãn: [Label Studio](https://labelstud.io) (free, local)

Lưu vào: `data/labels/<symbol>_labeled.csv` với 2 cột: `text`, `label`

### Bước 3 — Fine-tune PhoBERT (optional nhưng nên làm)
```python
from sentiment.scorer import fine_tune_phobert
fine_tune_phobert(
    train_csv='data/labels/FPT_labeled.csv',
    output_dir='models/phobert_sentiment',
    num_epochs=3
)
```
Cần GPU hoặc Apple Silicon (MPS) → ~30 phút với 1000 samples.

### Bước 4 — Chạy với fine-tuned model
```bash
python main.py --sentiment --sentiment-model models/phobert_sentiment
```

### So sánh kết quả
Chạy pipeline 2 lần:
- Không có sentiment (19 features): `python main.py`
- Có sentiment (20 features): `python main.py --sentiment`

So sánh MAPE, DA, Sharpe → đánh giá mức độ cải thiện.

---

## 4. Portfolio Optimization ✅

Đã implement `portfolio.py` với Markowitz MVO.

```bash
python main.py --portfolio   # chạy sau khi train tất cả mã
```

Output:
- `plot/portfolio_optimization.png` — Efficient Frontier + Equity curves
- Bảng tóm tắt: Ann. Return, Volatility, Sharpe cho từng strategy
- Weights tối ưu cho từng mã

---

## Thứ tự ưu tiên còn lại

| # | Việc | Trạng thái |
|---|------|-----------|
| 1 | Cập nhật data từ vnstock | ✅ Code xong, cần `--refresh` |
| 2 | Multi-step prediction | ✅ Hoàn chỉnh |
| 3 | Portfolio Optimization | ✅ Hoàn chỉnh |
| 4 | Sentiment — crawl + keyword scoring | ✅ Code xong |
| 5 | Sentiment — gán nhãn thủ công | ⏳ Cần làm thủ công (~500 samples) |
| 6 | Sentiment — fine-tune PhoBERT | ⏳ Sau khi có nhãn |

---

## Ghi chú kỹ thuật

- vnstock 4.x đổi hoàn toàn cú pháp so với v1/v2 — không dùng `stock_historical_data()` nữa
- PhoBERT cần GPU để fine-tune nhanh — dùng Google Colab nếu máy không đủ mạnh
- Khi thêm feature `sentiment`, nhớ bật `--sentiment` flag khi chạy
- Apple Silicon (M-series): cài thêm `tensorflow-metal` để tăng tốc train
- Keyword scoring (không cần GPU) đã hoạt động ngay — đủ để thử nghiệm ban đầu
