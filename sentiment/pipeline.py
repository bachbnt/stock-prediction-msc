# sentiment/pipeline.py — Tổng hợp sentiment theo ngày, tích hợp vào features
"""
Quy trình đầy đủ:
  1. Crawl tin tức (hoặc dùng cache)
  2. Score từng article
  3. Aggregate theo ngày giao dịch
  4. Fill missing days (forward-fill rồi fill 0)
  5. Lưu ra data/sentiment/<symbol>.csv

Kết quả: DataFrame với cột ['date', 'sentiment'] — có thể merge vào price df.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

from sentiment.crawler import crawl_news
from sentiment.scorer  import PhoBERTScorer

logger = logging.getLogger(__name__)

SENTIMENT_DIR = 'data/sentiment'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Score articles
# ═══════════════════════════════════════════════════════════════════════════════

def score_articles(articles: list,
                   scorer: Optional[PhoBERTScorer] = None) -> pd.DataFrame:
    """
    Tính sentiment score cho danh sách articles.
    Dùng title + content (nếu có) làm input.

    Returns: DataFrame với ['date', 'title', 'score']
    """
    if scorer is None:
        scorer = PhoBERTScorer()   # keyword fallback nếu không có model

    texts = [
        f"{a.get('title', '')} {a.get('content', '')}".strip()
        for a in articles
    ]

    print(f"  Scoring {len(texts)} articles...")
    scores = scorer.score_batch(texts)

    records = []
    for art, score in zip(articles, scores):
        records.append({
            'date':  art['date'],
            'title': art.get('title', ''),
            'score': score,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Aggregate theo ngày
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_daily(scored_df: pd.DataFrame,
                    trading_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Tính sentiment score trung bình cho mỗi ngày giao dịch.
    Ngày không có tin → 0 (neutral).
    Ngày có nhiều tin → mean score.

    Args:
        scored_df:     DataFrame với ['date', 'score']
        trading_dates: DatetimeIndex của tất cả ngày giao dịch

    Returns:
        pd.Series với index=trading_dates, values=sentiment [-1, +1]
    """
    scored_df['date'] = pd.to_datetime(scored_df['date'])
    daily = (scored_df
             .groupby('date')['score']
             .agg(['mean', 'count'])
             .rename(columns={'mean': 'sentiment', 'count': 'n_articles'}))

    # Reindex theo trading dates, fill với 0 (neutral)
    sentiment = daily['sentiment'].reindex(trading_dates, fill_value=np.nan)

    # Forward-fill tối đa 3 ngày (tin hôm nay ảnh hưởng 3 ngày tới)
    sentiment = sentiment.fillna(method='ffill', limit=3)
    sentiment = sentiment.fillna(0.0)

    return sentiment


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cache sentiment scores
# ═══════════════════════════════════════════════════════════════════════════════

def save_sentiment_cache(symbol: str, sentiment: pd.Series,
                         cache_dir: str = SENTIMENT_DIR):
    """Lưu daily sentiment scores ra CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{symbol}.csv')
    df   = sentiment.reset_index()
    df.columns = ['date', 'sentiment']
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.to_csv(path, index=False)
    logger.info(f"[{symbol}] Sentiment saved → {path}")


def load_sentiment_cache(symbol: str,
                         cache_dir: str = SENTIMENT_DIR) -> Optional[pd.DataFrame]:
    """
    Load cached sentiment scores nếu có.
    Returns DataFrame với ['date', 'sentiment'] hoặc None.
    """
    path = os.path.join(cache_dir, f'{symbol}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    logger.info(f"[{symbol}] Loaded sentiment cache: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_sentiment_pipeline(symbol: str,
                            start: str,
                            end: str,
                            price_df: Optional[pd.DataFrame] = None,
                            model_path: Optional[str] = None,
                            force_refresh: bool = False,
                            cache_dir: str = SENTIMENT_DIR) -> pd.DataFrame:
    """
    Pipeline đầy đủ: crawl → score → aggregate → cache → return.

    Args:
        symbol:        Mã cổ phiếu
        start/end:     'YYYY-MM-DD'
        price_df:      DataFrame giá (có cột 'date') để lấy trading dates.
                       None → dùng tất cả ngày trong [start, end].
        model_path:    Path đến fine-tuned PhoBERT model.
                       None → dùng keyword scoring (không cần cài transformers).
        force_refresh: Bỏ qua cache, crawl lại.

    Returns:
        DataFrame với ['date', 'sentiment'] — indexed by date.
        Có thể merge vào price_df: price_df.merge(result, on='date', how='left')
    """
    print(f"\n  [Sentiment] {symbol}: {start} → {end}")

    # Load từ cache nếu có và không force refresh
    if not force_refresh:
        cached = load_sentiment_cache(symbol, cache_dir)
        if cached is not None:
            # Check đủ coverage
            cached_start = str(cached['date'].min().date())
            cached_end   = str(cached['date'].max().date())
            if cached_start <= start and cached_end >= end:
                print(f"  [Sentiment] Cache hit: {len(cached)} rows")
                mask = (cached['date'] >= start) & (cached['date'] <= end)
                return cached[mask].reset_index(drop=True)

    # Xác định trading dates
    if price_df is not None:
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])
        mask    = (price_df['date'] >= start) & (price_df['date'] <= end)
        trading_dates = pd.DatetimeIndex(price_df[mask]['date'].values)
    else:
        trading_dates = pd.date_range(start=start, end=end, freq='B')

    # 1. Crawl tin tức
    articles = crawl_news(symbol, start, end, force_refresh=force_refresh)
    if not articles:
        print(f"  [Sentiment] Không có tin tức, trả về 0 (neutral) cho tất cả ngày.")
        sentiment_s = pd.Series(0.0, index=trading_dates, name='sentiment')
        result = sentiment_s.reset_index()
        result.columns = ['date', 'sentiment']
        return result

    # 2. Score articles
    scorer     = PhoBERTScorer(model_path=model_path)
    scored_df  = score_articles(articles, scorer)

    # 3. Aggregate theo ngày
    sentiment_s = aggregate_daily(scored_df, trading_dates)

    # 4. Save cache
    save_sentiment_cache(symbol, sentiment_s, cache_dir)

    # 5. Return
    result = sentiment_s.reset_index()
    result.columns = ['date', 'sentiment']
    print(f"  [Sentiment] Done: {len(result)} ngày, "
          f"mean={result['sentiment'].mean():.3f}, "
          f"std={result['sentiment'].std():.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Merge sentiment vào price DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def add_sentiment_feature(price_df: pd.DataFrame,
                           sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment scores vào DataFrame giá.
    Ngày không có sentiment → fill 0.

    Args:
        price_df:     DataFrame với cột 'date'
        sentiment_df: DataFrame với ['date', 'sentiment']

    Returns:
        price_df với cột 'sentiment' được thêm vào.
    """
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    sent = sentiment_df.copy()
    sent['date'] = pd.to_datetime(sent['date'])

    df = df.merge(sent[['date', 'sentiment']], on='date', how='left')
    df['sentiment'] = df['sentiment'].fillna(0.0)
    return df
