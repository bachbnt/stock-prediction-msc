# sentiment/labeling.py — Tools for the manual PhoBERT labeling step
"""
Labels must be written by a human (price-reaction "weak labels" are deliberately
NOT offered: they would leak the prediction target into the sentiment feature).

Workflow
  1. python main.py --sentiment                       # crawls + caches data/news/<SYM>/
  2. python -m sentiment.labeling export FPT 600      # -> data/labels/FPT_to_label.csv
  3. fill the `label` column (1 positive, 0 neutral, -1 negative) -> save as
     data/labels/FPT_labeled.csv
  4. python -m sentiment.labeling check data/labels/FPT_labeled.csv
  5. fine_tune_phobert('data/labels/FPT_labeled.csv', 'models/phobert_sentiment')
  6. python -m sentiment.labeling eval data/labels/FPT_labeled.csv   # keyword baseline
  7. python research.py --ablation --sentiment --sentiment-model models/phobert_sentiment

The export is stratified by month so the sample spans the whole period.
"""
import os
import sys

import numpy as np
import pandas as pd


def export_label_template(symbol: str, n: int = 600, start: str = '2015-01-01',
                          end: str = '2100-01-01', out_dir: str = 'data/labels',
                          seed: int = 0) -> str:
    from sentiment.crawler import load_news_cache
    arts = load_news_cache(symbol, start, end)
    if not arts:
        raise SystemExit(f"No cached news for {symbol}. Run: python main.py --sentiment")
    df = pd.DataFrame([{'date': a['date'], 'text': f"{a.get('title', '')} {a.get('content', '')}".strip()}
                       for a in arts]).drop_duplicates('text')
    df['month'] = df['date'].str[:7]
    per_month = max(1, n // max(df['month'].nunique(), 1))
    rng = np.random.default_rng(seed)
    parts = [g.sample(min(len(g), per_month), random_state=int(rng.integers(1 << 31)))
             for _, g in df.groupby('month')]
    out = pd.concat(parts).head(n)[['date', 'text']].assign(label='')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{symbol}_to_label.csv')
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows -> {path}  (fill the 'label' column: 1 / 0 / -1)")
    return path


def check_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {'text', 'label'} <= set(df.columns):
        raise SystemExit("CSV needs columns: text,label")
    bad = ~df['label'].isin([-1, 0, 1])
    print(f"{len(df)} rows | class counts: {df['label'].value_counts().to_dict()} | invalid: {int(bad.sum())}")
    if bad.any():
        print(df[bad].head())
    return df


def eval_keyword_baseline(path: str) -> dict:
    """Accuracy / macro-F1 of the keyword fallback scorer on labeled data."""
    from sentiment.scorer import keyword_score
    from sklearn.metrics import accuracy_score, f1_score
    df = check_labels(path)
    df = df[df['label'].isin([-1, 0, 1])]
    pred = [1 if keyword_score(t) > 0.1 else -1 if keyword_score(t) < -0.1 else 0 for t in df['text']]
    res = dict(accuracy=accuracy_score(df['label'], pred),
               macro_f1=f1_score(df['label'], pred, average='macro'))
    print(res)
    return res


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'export':
        export_label_template(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    elif cmd == 'check':
        check_labels(sys.argv[2])
    elif cmd == 'eval':
        eval_keyword_baseline(sys.argv[2])
    else:
        print(__doc__)
