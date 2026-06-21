# Stock Prediction MSc — Kế hoạch mở rộng

## Đã hoàn thành

### Codebase
- `config.py` — constants tập trung (symbols, features, hyperparams, tax)
- `data_loader.py` — load CSV local / GitHub, update từ vnstock, prepare sequences
- `features.py` — 19 technical indicators (RSI, MACD, Bollinger, SMA/EMA, Volume)
- `models.py` — LSTM, BiLSTM, CNN-LSTM, dynamic ensemble
- `evaluate.py` — RMSE, MAE, MAPE, DA, Sharpe, Max Drawdown, Calmar, B&H
- `visualize.py` — chart 5 panel: price prediction, portfolio, drawdown, MAPE/DA, Return/Sharpe
- `thesis.py` — entry point với CLI args (`--symbols`, `--refresh`, `--save-dir`)

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

python thesis.py                       # chạy tất cả 5 mã
python thesis.py --symbols FPT VCB    # chạy 2 mã
python thesis.py --refresh            # cập nhật data từ vnstock trước khi train
```

---

## Việc cần làm tiếp

### 1. Cập nhật dữ liệu (data_loader.py)

CSV hiện tại kết thúc **31/05/2022**, thiếu ~4 năm. Cần fetch thêm từ vnstock.

```bash
pip install vnstock>=4.0.0
python thesis.py --refresh   # tự động fetch và nối vào CSV
```

Kiểm tra `update_data()` trong `data_loader.py` — hàm đã viết sẵn, chỉ cần vnstock được cài.

**Cú pháp vnstock 4.x (quan trọng — API đã đổi so với v1/v2):**
```python
from vnstock import Quote
quote = Quote(symbol='FPT', source='VCI')  # hoặc source='KBS'
df = quote.history(start='2022-06-01', end='2026-06-01', interval='1D')
# cột trả về: time, open, high, low, close, volume (lowercase)
```

---

### 2. Sentiment Analysis — Feature thứ 20

Thêm sentiment score từ tin tức vào FEATURE_COLS, tăng từ 19 → 20 features.

#### Bước 1 — Thu thập tin tức

**Cách A: Google News (dễ nhất, có lịch sử)**
```bash
pip install pygooglenews
```
```python
from pygooglenews import GoogleNews
gn = GoogleNews(lang='vi', country='VN')
results = gn.search('FPT cổ phiếu', from_='2021-01-01', to_='2021-01-31')
# → tiêu đề + link, tổng hợp từ CafeF, VnExpress, Thanh Niên...
```

**Cách B: Scrape VnExpress trực tiếp (có nội dung đầy đủ)**
```bash
pip install requests beautifulsoup4
```
```python
import requests
from bs4 import BeautifulSoup

url = "https://timkiem.vnexpress.net/"
params = {"q": "FPT", "fromDate": "2021-01-01", "toDate": "2021-01-31"}
resp = requests.get(url, params=params)
soup = BeautifulSoup(resp.text, 'html.parser')
# parse tiêu đề + nội dung bài
```

**Cách C: RSS Feed (chỉ tin hiện tại, không có lịch sử)**
```python
import feedparser
feed = feedparser.parse('https://vnexpress.net/rss/kinh-doanh.rss')
```

**Khuyến nghị:** Dùng Cách A để lấy danh sách link theo ngày → Cách B để đọc nội dung từng bài.

Lưu output vào: `data/news/<symbol>/<YYYY-MM-DD>.json`

#### Bước 2 — Gán nhãn (bắt buộc để fine-tune PhoBERT)

Cần ~500–1000 câu được gán nhãn thủ công:
- `1` = tích cực (tăng trưởng, lợi nhuận tốt, ký hợp đồng lớn...)
- `0` = trung tính
- `-1` = tiêu cực (lỗ, điều tra, thị trường giảm...)

Tool gán nhãn đơn giản: [Label Studio](https://labelstud.io) (cài local, miễn phí)

#### Bước 3 — Fine-tune PhoBERT

```bash
pip install transformers torch datasets
```
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base", num_labels=3
)
# Fine-tune trên tập đã gán nhãn
# Output: sentiment score [-1, +1] cho mỗi bài báo
```

#### Bước 4 — Tổng hợp theo ngày & tích hợp

```python
# Tính sentiment score trung bình cho mỗi ngày giao dịch
daily_sentiment = news_df.groupby('date')['score'].mean()

# Thêm vào DataFrame chính
df['sentiment'] = df['date'].map(daily_sentiment).fillna(0)

# Cập nhật config.py
FEATURE_COLS = [...19 features hiện tại..., 'sentiment']  # → 20 features
```

#### Bước 5 — So sánh kết quả

Chạy pipeline 2 lần:
- Không có sentiment (19 features)
- Có sentiment (20 features)

So sánh MAPE, DA, Sharpe Ratio → đánh giá mức độ cải thiện.

---

### 3. Multi-step Prediction

Hiện tại: dự đoán 1 ngày tới → mở rộng ra 5, 10, 20 ngày.

Thay đổi cần làm trong `models.py`:
```python
# Thay Dense(1) → Dense(n_steps)
N_STEPS = 5   # dự đoán 5 ngày tới
m.add(Dense(N_STEPS))

# Thêm vào config.py
N_PREDICT_STEPS = 5
```

Thay đổi trong `data_loader.py`:
```python
# y_train giờ là vector [t+1, t+2, ..., t+n] thay vì scalar
y.append(data[i: i + n_steps, target_idx])
```

Metric đánh giá cần tính riêng cho từng bước (step-1, step-5, step-10).

---

### 4. Portfolio Optimization

Thay vì giao dịch từng mã riêng lẻ, phân bổ vốn tối ưu cho cả 5 mã.

```bash
pip install scipy
```
```python
from scipy.optimize import minimize

# Dùng dự đoán LSTM để ước lượng expected return
# Kết hợp với Markowitz Mean-Variance Optimization
# Output: weights = [w_FPT, w_VCB, w_VIC, w_HPG, w_VNM]
# Sum(weights) = 1, weights >= 0
```

So sánh:
- Strategy hiện tại: giao dịch từng mã độc lập
- Portfolio strategy: phân bổ vốn theo tỷ trọng tối ưu

---

## Thứ tự ưu tiên

| # | Việc | Độ phức tạp | Giá trị học thuật |
|---|------|------------|------------------|
| 1 | Cập nhật data từ vnstock | Thấp | — |
| 2 | Multi-step prediction | Trung bình | Trung bình |
| 3 | Sentiment Analysis | Cao | Cao |
| 4 | Portfolio Optimization | Trung bình | Trung bình |

---

## Ghi chú kỹ thuật

- vnstock 4.x đổi hoàn toàn cú pháp so với v1/v2 — không dùng `stock_historical_data()` nữa
- PhoBERT cần GPU để fine-tune nhanh — dùng Google Colab nếu máy không đủ mạnh
- Khi thêm feature `sentiment`, nhớ cập nhật `FEATURE_COLS` trong `config.py` và chạy lại toàn bộ pipeline
- Apple Silicon (M-series): cài thêm `tensorflow-metal` để tăng tốc train
