# sentiment/scorer.py — Compute sentiment scores using PhoBERT
"""
Two modes:
  1. Zero-shot (no fine-tuning required):
       Uses 'vinai/phobert-base' + MLM-based keyword scoring
       → fast, no labeled data needed, moderate accuracy

  2. Fine-tuned (requires ~500-1000 labeled samples):
       Uses AutoModelForSequenceClassification fine-tuned on domain data
       → slower, significantly higher accuracy

Labels: POSITIVE (+1), NEUTRAL (0), NEGATIVE (-1)

Installation:
  pip install transformers torch sentencepiece
  # Apple Silicon:
  pip install transformers torch sentencepiece  # torch will automatically use MPS
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Vietnamese keyword dictionary for zero-shot fallback ─────────────────────
# Positive financial keywords (growth, profit, contracts, dividends, etc.)
_POS_KEYWORDS = [
    'tăng', 'tăng trưởng', 'lợi nhuận', 'vượt', 'kỷ lục', 'kế hoạch', 'ký kết',
    'hợp đồng', 'thắng', 'triển vọng', 'tốt', 'khả quan', 'mua', 'upside',
    'tích cực', 'tăng mạnh', 'đột phá', 'cổ tức', 'thưởng', 'chia cổ phiếu',
]
# Negative financial keywords (losses, violations, debt, sell-offs, etc.)
_NEG_KEYWORDS = [
    'giảm', 'lỗ', 'thua', 'vi phạm', 'phạt', 'khởi tố', 'điều tra', 'nợ xấu',
    'bán tháo', 'giảm sâu', 'thấp nhất', 'rủi ro', 'khó khăn', 'áp lực',
    'tiêu cực', 'suy giảm', 'thu hẹp', 'đóng cửa', 'phá sản',
]


def keyword_score(text: str) -> float:
    """
    Simple keyword-based sentiment score.
    Returns: float in [-1, +1]
    """
    text_lower = text.lower()
    pos = sum(1 for kw in _POS_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in _NEG_KEYWORDS if kw in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    # Normalize by total keyword matches so score stays in [-1, +1]
    return (pos - neg) / total


# ═══════════════════════════════════════════════════════════════════════════════
# PhoBERT Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class PhoBERTScorer:
    """
    Wrapper around PhoBERT to compute sentiment scores for Vietnamese text.

    If a fine-tuned model exists at model_path → load and use it.
    Otherwise → fall back to keyword-based scoring.
    """

    MODEL_NAME   = 'vinai/phobert-base-v2'
    LABEL_MAP    = {0: -1.0, 1: 0.0, 2: 1.0}   # NEGATIVE, NEUTRAL, POSITIVE
    MAX_LENGTH   = 256

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_path: Path to the fine-tuned model directory.
                        None → use keyword fallback.
            device:     'cpu', 'cuda', 'mps' (Apple Silicon).
                        None → auto-detect.
        """
        self.model      = None
        self.tokenizer  = None
        self.device     = self._detect_device(device)
        self._loaded    = False

        if model_path:
            self._load_model(model_path)

    def _detect_device(self, device: Optional[str]) -> str:
        """Auto-detect the best available compute device."""
        if device:
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'   # Apple Silicon GPU
            if torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
        except ImportError:
            pass
        return 'cpu'

    def _load_model(self, model_path: str):
        """Load a fine-tuned PhoBERT model from a local directory."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading PhoBERT from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()  # set to inference mode (disables dropout, etc.)
            self._loaded = True
            logger.info(f"PhoBERT loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Falling back to keyword scoring.")

    def score(self, text: str) -> float:
        """
        Compute sentiment score for a single text snippet.
        Returns: float in [-1, +1]
        """
        if not text or not text.strip():
            return 0.0  # empty text is treated as neutral

        if not self._loaded:
            return keyword_score(text)

        try:
            import torch
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.MAX_LENGTH,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Weighted dot product: probs * label values (-1, 0, +1)
            # Gives a continuous score in [-1, +1] rather than a hard class label
            labels = np.array([-1.0, 0.0, 1.0])
            score  = float(np.dot(probs, labels))
            return np.clip(score, -1.0, 1.0)

        except Exception as e:
            logger.warning(f"PhoBERT inference error: {e}. Falling back to keyword scoring.")
            return keyword_score(text)

    def score_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Compute sentiment scores for a list of texts in batches.
        Returns: list of floats in [-1, +1]
        """
        if not self._loaded:
            return [keyword_score(t) for t in texts]

        try:
            import torch
            scores = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    max_length=self.MAX_LENGTH,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                labels = np.array([-1.0, 0.0, 1.0])
                # Compute weighted score for each sample in the batch
                batch_scores = [float(np.clip(np.dot(p, labels), -1.0, 1.0))
                                for p in probs]
                scores.extend(batch_scores)
            return scores
        except Exception as e:
            logger.warning(f"Batch inference error: {e}")
            return [keyword_score(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning helper
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune_phobert(train_csv: str, output_dir: str = 'models/phobert_sentiment',
                      num_epochs: int = 3, batch_size: int = 16):
    """
    Fine-tune PhoBERT on a labeled dataset.

    train_csv: CSV with 2 columns: 'text' (headline/content) and 'label' (-1, 0, 1)

    Run on a GPU or Apple Silicon (MPS) for faster training.
    Estimated time: ~30 minutes on M2 Pro with 1000 samples, 3 epochs.
    """
    try:
        import torch
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   AdamW, get_linear_schedule_with_warmup)
    except ImportError:
        logger.error("Required: pip install transformers torch pandas")
        return

    # Load labeled training data
    df = pd.read_csv(train_csv)
    assert 'text' in df.columns and 'label' in df.columns, \
        "CSV must have columns 'text' and 'label' (-1, 0, 1)"

    # Remap sentiment labels to class indices: -1→0, 0→1, 1→2
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label_id'] = df['label'].map(label_map)

    tokenizer = AutoTokenizer.from_pretrained(PhoBERTScorer.MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        PhoBERTScorer.MODEL_NAME, num_labels=3
    )

    # Simple Dataset wrapper that tokenizes all texts upfront
    class NewsDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc    = tokenizer(texts, truncation=True, padding=True,
                                    max_length=256, return_tensors='pt')
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):  return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    dataset    = NewsDataset(df['text'].tolist(), df['label_id'].tolist())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device    = PhoBERTScorer()._detect_device(None)
    model.to(device)

    optimizer  = AdamW(model.parameters(), lr=2e-5)
    total_steps= len(dataloader) * num_epochs
    # Linear warmup for the first 10% of steps, then linear decay
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, labels in dataloader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            labels  = labels.to(device)
            outputs = model(**batch, labels=labels)
            loss    = outputs.loss
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader):.4f}")

    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved → {output_dir}")
    print(f"Fine-tuning complete! Model → {output_dir}")
    print(f"Usage: scorer = PhoBERTScorer(model_path='{output_dir}')")
