# sentiment/scorer.py — Tính sentiment score bằng PhoBERT
"""
Hai chế độ:
  1. Zero-shot (không cần fine-tune):
       Dùng 'vinai/phobert-base' + MLM-based keyword scoring
       → nhanh, không cần data, độ chính xác trung bình

  2. Fine-tuned (cần ~500-1000 mẫu gán nhãn):
       Dùng AutoModelForSequenceClassification fine-tuned
       → chậm hơn, độ chính xác cao hơn đáng kể

Labels: POSITIVE (+1), NEUTRAL (0), NEGATIVE (-1)

Cài đặt:
  pip install transformers torch sentencepiece
  # Apple Silicon:
  pip install transformers torch sentencepiece  # torch tự dùng MPS
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Từ điển keyword-based cho zero-shot fallback ──────────────────────────────
_POS_KEYWORDS = [
    'tăng', 'tăng trưởng', 'lợi nhuận', 'vượt', 'kỷ lục', 'kế hoạch', 'ký kết',
    'hợp đồng', 'thắng', 'triển vọng', 'tốt', 'khả quan', 'mua', 'upside',
    'tích cực', 'tăng mạnh', 'đột phá', 'cổ tức', 'thưởng', 'chia cổ phiếu',
]
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
    return (pos - neg) / total


# ═══════════════════════════════════════════════════════════════════════════════
# PhoBERT Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class PhoBERTScorer:
    """
    Wrapper quanh PhoBERT để tính sentiment score cho văn bản tiếng Việt.

    Nếu fine-tuned model tồn tại tại model_path → load và dùng.
    Nếu không → dùng keyword fallback.
    """

    MODEL_NAME   = 'vinai/phobert-base-v2'
    LABEL_MAP    = {0: -1.0, 1: 0.0, 2: 1.0}   # NEGATIVE, NEUTRAL, POSITIVE
    MAX_LENGTH   = 256

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_path: Path đến fine-tuned model directory.
                        None → dùng keyword fallback.
            device:     'cpu', 'cuda', 'mps' (Apple Silicon).
                        None → tự detect.
        """
        self.model      = None
        self.tokenizer  = None
        self.device     = self._detect_device(device)
        self._loaded    = False

        if model_path:
            self._load_model(model_path)

    def _detect_device(self, device: Optional[str]) -> str:
        if device:
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'

    def _load_model(self, model_path: str):
        """Load fine-tuned PhoBERT từ local directory."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading PhoBERT từ {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"PhoBERT loaded on {self.device}")
        except Exception as e:
            logger.error(f"Không load được model: {e}. Dùng keyword fallback.")

    def score(self, text: str) -> float:
        """
        Tính sentiment score cho một đoạn văn bản.
        Returns: float in [-1, +1]
        """
        if not text or not text.strip():
            return 0.0

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

            # Weighted score: probs * labels (-1, 0, +1)
            labels = np.array([-1.0, 0.0, 1.0])
            score  = float(np.dot(probs, labels))
            return np.clip(score, -1.0, 1.0)

        except Exception as e:
            logger.warning(f"PhoBERT inference lỗi: {e}. Dùng keyword fallback.")
            return keyword_score(text)

    def score_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Tính sentiment scores cho nhiều văn bản.
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
                batch_scores = [float(np.clip(np.dot(p, labels), -1.0, 1.0))
                                for p in probs]
                scores.extend(batch_scores)
            return scores
        except Exception as e:
            logger.warning(f"Batch inference lỗi: {e}")
            return [keyword_score(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning helper
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune_phobert(train_csv: str, output_dir: str = 'models/phobert_sentiment',
                      num_epochs: int = 3, batch_size: int = 16):
    """
    Fine-tune PhoBERT trên tập dữ liệu gán nhãn.

    train_csv: CSV với 2 cột: 'text' (tiêu đề/nội dung) và 'label' (-1, 0, 1)

    Chạy trên máy có GPU hoặc Apple Silicon (MPS) để nhanh hơn.
    Ước tính: ~30 phút trên M2 Pro với 1000 samples, 3 epochs.
    """
    try:
        import torch
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   AdamW, get_linear_schedule_with_warmup)
    except ImportError:
        logger.error("Cần: pip install transformers torch pandas")
        return

    # Load data
    df = pd.read_csv(train_csv)
    assert 'text' in df.columns and 'label' in df.columns, \
        "CSV cần có cột 'text' và 'label' (-1, 0, 1)"

    # Map labels: -1→0, 0→1, 1→2
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label_id'] = df['label'].map(label_map)

    tokenizer = AutoTokenizer.from_pretrained(PhoBERTScorer.MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        PhoBERTScorer.MODEL_NAME, num_labels=3
    )

    # Simple Dataset
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
    print(f"Fine-tuning xong! Model → {output_dir}")
    print(f"Dùng: scorer = PhoBERTScorer(model_path='{output_dir}')")
