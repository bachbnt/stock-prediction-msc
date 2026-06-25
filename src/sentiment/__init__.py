# sentiment/ — Module for financial news sentiment analysis
#
# Usage:
#   from sentiment.pipeline import run_sentiment_pipeline
#   df = run_sentiment_pipeline(symbol='FPT', start='2021-01-01', end='2023-12-31')
#   # Returns a DataFrame with columns: date, sentiment ([-1, +1])
#
# Requirements:
#   pip install pygooglenews requests beautifulsoup4 transformers torch
#
# Workflow:
#   1. crawler.py  — Collect news from Google News + VnExpress
#   2. scorer.py   — Compute sentiment scores using PhoBERT
#   3. pipeline.py — Aggregate by day, merge into price DataFrame

from sentiment.pipeline import run_sentiment_pipeline, load_sentiment_cache

__all__ = ['run_sentiment_pipeline', 'load_sentiment_cache']
