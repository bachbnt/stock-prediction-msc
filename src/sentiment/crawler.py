# sentiment/crawler.py — Collect Vietnamese financial news articles
"""
Two sources:
  A. Google News (pygooglenews) — easy to use, historical data, title + link
  B. VnExpress search API      — full content, requires HTML parsing

Output: list of dicts [{'date': str, 'title': str, 'content': str, 'source': str}]
Cache saved to: data/news/<symbol>/<YYYY-MM>.json
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Google News via pygooglenews
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_google_news(symbol: str, start: str, end: str,
                      query_extra: str = 'cổ phiếu') -> list:
    """
    Fetch news articles from Google News RSS feed.

    Args:
        symbol:      Stock ticker symbol (e.g. 'FPT')
        start:       'YYYY-MM-DD'
        end:         'YYYY-MM-DD'
        query_extra: Additional keyword appended to query (default: 'cổ phiếu' — Vietnamese for 'stock')

    Returns:
        list of {'date', 'title', 'url', 'source'}
    """
    try:
        from pygooglenews import GoogleNews
    except ImportError:
        logger.error("pygooglenews is not installed. Run: pip install pygooglenews")
        return []

    gn      = GoogleNews(lang='vi', country='VN')
    query   = f'{symbol} {query_extra}'
    results = []

    # pygooglenews accepts dates in YYYY-MM-DD format
    try:
        search = gn.search(query, from_=start, to_=end)
        for entry in search.get('entries', []):
            pub = entry.get('published', '')
            try:
                date = datetime.strptime(pub[:16], '%a, %d %b %Y').strftime('%Y-%m-%d')
            except ValueError:
                date = start  # fallback to start date if parsing fails

            results.append({
                'date':    date,
                'title':   entry.get('title', '').strip(),
                'url':     entry.get('link', ''),
                'content': '',           # will be fetched later if needed
                'source':  'google_news',
            })
        logger.info(f"[{symbol}] Google News: {len(results)} articles ({start} → {end})")
    except Exception as e:
        logger.error(f"[{symbol}] Google News error: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B. VnExpress search (scraping)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_vnexpress(symbol: str, start: str, end: str,
                    delay: float = 1.0) -> list:
    """
    Scrape VnExpress search results for a given stock ticker.
    Rate-limited (delay seconds between requests to avoid being blocked).

    Returns:
        list of {'date', 'title', 'content', 'url', 'source'}
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests/beautifulsoup4 are not installed.")
        return []

    headers = {'User-Agent': 'Mozilla/5.0 (compatible; StockSentimentBot/1.0)'}
    results = []
    page    = 1

    # Convert date strings to Unix timestamps required by VnExpress search API
    start_ts = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    end_ts   = int(datetime.strptime(end, '%Y-%m-%d').timestamp())

    while True:
        url = (f'https://timkiem.vnexpress.net/?q={symbol}+cổ+phiếu'
               f'&media_type=1&fromdate={start_ts}&todate={end_ts}'
               f'&page={page}')
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            articles = soup.select('article.item-news')
            if not articles:
                break  # no more pages

            for art in articles:
                title_el = art.select_one('h3.title-news a, h2.title-news a')
                desc_el  = art.select_one('p.description')
                date_el  = art.select_one('span.time-count')

                title   = title_el.get_text(strip=True) if title_el else ''
                content = desc_el.get_text(strip=True)  if desc_el  else ''
                art_url = title_el['href'] if title_el and title_el.get('href') else ''

                # Parse date from data attribute or fall back to start date
                date = start
                if date_el and date_el.get('data-publish'):
                    try:
                        ts   = int(date_el['data-publish'])
                        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    except (ValueError, OSError):
                        pass

                if title:
                    results.append({
                        'date':    date,
                        'title':   title,
                        'content': content,
                        'url':     art_url,
                        'source':  'vnexpress',
                    })

            page += 1
            time.sleep(delay)

        except Exception as e:
            logger.error(f"[{symbol}] VnExpress page {page} error: {e}")
            break

    logger.info(f"[{symbol}] VnExpress: {len(results)} articles ({start} → {end})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# C. Cache management
# ═══════════════════════════════════════════════════════════════════════════════

def save_news_cache(symbol: str, articles: list,
                    cache_dir: str = 'data/news'):
    """Save articles to JSON files, grouped by month."""
    by_month: dict = {}
    for art in articles:
        month = art['date'][:7]   # extract YYYY-MM from YYYY-MM-DD
        by_month.setdefault(month, []).append(art)

    for month, arts in by_month.items():
        path = os.path.join(cache_dir, symbol, f'{month}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Merge with existing cache to avoid overwriting previously fetched data
        existing = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        # Deduplicate by title to avoid storing the same article twice
        all_arts   = existing + arts
        seen_titles = set()
        deduped    = []
        for a in all_arts:
            if a['title'] not in seen_titles:
                seen_titles.add(a['title'])
                deduped.append(a)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)
    logger.info(f"[{symbol}] Saved {len(articles)} articles to {cache_dir}/{symbol}/")


def load_news_cache(symbol: str, start: str, end: str,
                    cache_dir: str = 'data/news') -> list:
    """Load all cached articles within the [start, end] date range."""
    articles = []
    start_m  = start[:7]; end_m = end[:7]
    sym_dir  = os.path.join(cache_dir, symbol)

    if not os.path.isdir(sym_dir):
        return articles

    for fname in sorted(os.listdir(sym_dir)):
        if not fname.endswith('.json'):
            continue
        month = fname[:-5]   # strip .json to get YYYY-MM
        if month < start_m or month > end_m:
            continue  # skip months outside the requested range
        with open(os.path.join(sym_dir, fname), 'r', encoding='utf-8') as f:
            arts = json.load(f)
        # Filter to exact date range (monthly files may contain out-of-range dates)
        articles.extend([a for a in arts if start <= a['date'] <= end])

    return articles


# ═══════════════════════════════════════════════════════════════════════════════
# D. Main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def crawl_news(symbol: str, start: str, end: str,
               use_google: bool = True, use_vnexpress: bool = True,
               cache_dir: str = 'data/news',
               force_refresh: bool = False) -> list:
    """
    Collect news articles, using cache if available.

    Args:
        symbol:        Stock ticker symbol
        start/end:     'YYYY-MM-DD'
        use_google:    Enable/disable Google News source
        use_vnexpress: Enable/disable VnExpress source
        force_refresh: Ignore cache and re-fetch from sources

    Returns:
        list of deduplicated article dicts
    """
    if not force_refresh:
        cached = load_news_cache(symbol, start, end, cache_dir)
        if cached:
            logger.info(f"[{symbol}] Cache: {len(cached)} articles")
            return cached

    articles = []
    if use_google:
        articles.extend(fetch_google_news(symbol, start, end))
    if use_vnexpress:
        articles.extend(fetch_vnexpress(symbol, start, end))

    # Deduplicate across sources by title
    seen   = set()
    unique = []
    for a in articles:
        if a['title'] not in seen:
            seen.add(a['title'])
            unique.append(a)

    save_news_cache(symbol, unique, cache_dir)
    return unique
