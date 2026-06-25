# sentiment/pipeline.py — Aggregate daily sentiment scores and integrate into features
"""
Full workflow:
  1. Crawl news articles (or use cache)
  2. Score each article
  3. Aggregate by trading day
  4. Fill missing days (forward-fill then fill with 0)
  5. Save to data/sentiment/<symbol>.csv

Result: DataFrame with columns ['date', 'sentiment'] — can be merged into price df.
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
    Compute sentiment scores for a list of articles.
    Uses title + content (if available) as input text.

    Returns: DataFrame with ['date', 'title', 'score']
    """
    if scorer is None:
        scorer = PhoBERTScorer()   # use keyword fallback if no model is provided

    # Concatenate title and content for richer signal
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
# 2. Aggregate by day
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_daily(scored_df: pd.DataFrame,
                    trading_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Compute the average sentiment score for each trading day.
    Days with no news → 0 (neutral).
    Days with multiple articles → mean score.

    Args:
        scored_df:     DataFrame with ['date', 'score']
        trading_dates: DatetimeIndex of all trading days in the period

    Returns:
        pd.Series with index=trading_dates, values=sentiment [-1, +1]
    """
    scored_df['date'] = pd.to_datetime(scored_df['date'])
    daily = (scored_df
             .groupby('date')['score']
             .agg(['mean', 'count'])
             .rename(columns={'mean': 'sentiment', 'count': 'n_articles'}))

    # Reindex to all trading dates; days with no articles become NaN initially
    sentiment = daily['sentiment'].reindex(trading_dates, fill_value=np.nan)

    # Forward-fill up to 3 days: today's news is assumed to influence the next 3 trading days
    sentiment = sentiment.fillna(method='ffill', limit=3)
    # Remaining NaN (no recent news) → 0 (neutral)
    sentiment = sentiment.fillna(0.0)

    return sentiment


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cache sentiment scores
# ═══════════════════════════════════════════════════════════════════════════════

def save_sentiment_cache(symbol: str, sentiment: pd.Series,
                         cache_dir: str = SENTIMENT_DIR):
    """Save daily sentiment scores to a CSV file."""
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
    Load cached sentiment scores if available.
    Returns DataFrame with ['date', 'sentiment'] or None if not found.
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
    Full pipeline: crawl → score → aggregate → cache → return.

    Args:
        symbol:        Stock ticker symbol
        start/end:     'YYYY-MM-DD'
        price_df:      Price DataFrame (with 'date' column) used to extract trading dates.
                       None → use all business days in [start, end].
        model_path:    Path to a fine-tuned PhoBERT model.
                       None → use keyword scoring (no transformers installation needed).
        force_refresh: Ignore cache and re-crawl from scratch.

    Returns:
        DataFrame with ['date', 'sentiment'] — indexed by date.
        Can be merged into price_df: price_df.merge(result, on='date', how='left')
    """
    print(f"\n  [Sentiment] {symbol}: {start} → {end}")

    # Return from cache if available and coverage is sufficient
    if not force_refresh:
        cached = load_sentiment_cache(symbol, cache_dir)
        if cached is not None:
            # Verify that the cache covers the full requested date range
            cached_start = str(cached['date'].min().date())
            cached_end   = str(cached['date'].max().date())
            if cached_start <= start and cached_end >= end:
                print(f"  [Sentiment] Cache hit: {len(cached)} rows")
                mask = (cached['date'] >= start) & (cached['date'] <= end)
                return cached[mask].reset_index(drop=True)

    # Determine trading dates from price data or fall back to business day range
    if price_df is not None:
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])
        mask    = (price_df['date'] >= start) & (price_df['date'] <= end)
        trading_dates = pd.DatetimeIndex(price_df[mask]['date'].values)
    else:
        # freq='B' generates Monday–Friday business days
        trading_dates = pd.date_range(start=start, end=end, freq='B')

    # 1. Crawl news articles
    articles = crawl_news(symbol, start, end, force_refresh=force_refresh)
    if not articles:
        print(f"  [Sentiment] No news found, returning 0 (neutral) for all trading days.")
        sentiment_s = pd.Series(0.0, index=trading_dates, name='sentiment')
        result = sentiment_s.reset_index()
        result.columns = ['date', 'sentiment']
        return result

    # 2. Score each article
    scorer     = PhoBERTScorer(model_path=model_path)
    scored_df  = score_articles(articles, scorer)

    # 3. Aggregate scores by trading day
    sentiment_s = aggregate_daily(scored_df, trading_dates)

    # 4. Persist to cache for future runs
    save_sentiment_cache(symbol, sentiment_s, cache_dir)

    # 5. Return as a flat DataFrame
    result = sentiment_s.reset_index()
    result.columns = ['date', 'sentiment']
    print(f"  [Sentiment] Done: {len(result)} days, "
          f"mean={result['sentiment'].mean():.3f}, "
          f"std={result['sentiment'].std():.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Merge sentiment into price DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def add_sentiment_feature(price_df: pd.DataFrame,
                           sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment scores into a price DataFrame.
    Trading days without a matching sentiment score → filled with 0.

    Args:
        price_df:     DataFrame with a 'date' column
        sentiment_df: DataFrame with ['date', 'sentiment']

    Returns:
        price_df with an additional 'sentiment' column.
    """
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    sent = sentiment_df.copy()
    sent['date'] = pd.to_datetime(sent['date'])

    # Left join to preserve all price rows even when sentiment data is missing
    df = df.merge(sent[['date', 'sentiment']], on='date', how='left')
    df['sentiment'] = df['sentiment'].fillna(0.0)
    return df
