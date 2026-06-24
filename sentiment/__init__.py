# sentiment/ — Module phân tích cảm xúc tin tức tài chính
#
# Sử dụng:
#   from sentiment.pipeline import run_sentiment_pipeline
#   df = run_sentiment_pipeline(symbol='FPT', start='2021-01-01', end='2023-12-31')
#   # Trả về DataFrame với cột: date, sentiment ([-1, +1])
#
# Yêu cầu:
#   pip install pygooglenews requests beautifulsoup4 transformers torch
#
# Quy trình:
#   1. crawler.py  — Thu thập tin tức từ Google News + VnExpress
#   2. scorer.py   — Tính sentiment score bằng PhoBERT
#   3. pipeline.py — Tổng hợp theo ngày, merge vào DataFrame giá

from sentiment.pipeline import run_sentiment_pipeline, load_sentiment_cache

__all__ = ['run_sentiment_pipeline', 'load_sentiment_cache']
