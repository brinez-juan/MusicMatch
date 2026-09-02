from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

MODEL_NAME = "MilaNLProc/xlm-emo-t"
MAX_TOKENS = 512


class EmotionScorer:
    """Multilingual emotion classifier (XLM-T fine-tuned on emotion data).

    Produces per-label probabilities in [0, 1] for the labels exposed by the model
    (anger, fear, joy, sadness for xlm-emo-t). Use ``get_scorer()`` to obtain a
    cached singleton — instantiating directly forces a fresh model download.
    """

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading emotion model: %s", MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(self.device)
        id2label = self.model.config.id2label
        self.labels: list[str] = [id2label[i].lower() for i in range(len(id2label))]

    def _zero_vector(self) -> dict[str, float]:
        return {label: 0.0 for label in self.labels}

    @torch.inference_mode()
    def score(self, text: str) -> dict[str, float]:
        """Return mean per-label probabilities for a single text."""
        clean = _clean(text)
        if not clean:
            return self._zero_vector()

        enc = self.tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
            return_overflowing_tokens=True,
            stride=0,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0).cpu().tolist()
        return {label: float(p) for label, p in zip(self.labels, mean_probs)}

    @torch.inference_mode()
    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Score many texts with a single tokenizer call and single forward pass.

        Chunks from every non-empty input share one GPU trip; per-sample results are
        recovered via ``overflow_to_sample_mapping`` and averaged. Empty inputs
        short-circuit to a zero vector without touching the model.
        """
        cleaned = [_clean(t) for t in texts]
        non_empty_indices = [i for i, c in enumerate(cleaned) if c]
        if not non_empty_indices:
            return [self._zero_vector() for _ in texts]

        non_empty_texts = [cleaned[i] for i in non_empty_indices]
        enc = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
            return_overflowing_tokens=True,
            stride=0,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        mapping = enc["overflow_to_sample_mapping"].tolist()

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1).cpu()

        num_samples = len(non_empty_texts)
        num_labels = probs.shape[1]
        sums = torch.zeros(num_samples, num_labels)
        counts = torch.zeros(num_samples)
        for chunk_idx, sample_idx in enumerate(mapping):
            sums[sample_idx] += probs[chunk_idx]
            counts[sample_idx] += 1
        averaged = (sums / counts.unsqueeze(1)).tolist()

        results: list[dict[str, float]] = [self._zero_vector() for _ in texts]
        for position, sample_idx in enumerate(non_empty_indices):
            vec = averaged[position]
            results[sample_idx] = {label: float(p) for label, p in zip(self.labels, vec)}
        return results


def _clean(text: str) -> str:
    if not text:
        return ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


@functools.lru_cache(maxsize=1)
def get_scorer() -> EmotionScorer:
    """Return the process-wide :class:`EmotionScorer`, loading it on first call."""
    return EmotionScorer()


def get_continuous_sentiment(text: str) -> float:
    """Back-compat shim: derive a single valence scalar from the emotion vector.

    Formula: ``joy - mean(anger, fear, sadness)``, clipped to [-1, +1]. Kept so the
    ``sentiment_score`` column (and the recommender weight that relies on it) keep
    working while callers migrate to the full emotion dict.
    """
    emotions = get_scorer().score(text)
    negative = (emotions["anger"] + emotions["fear"] + emotions["sadness"]) / 3.0
    valence = emotions["joy"] - negative
    return max(-1.0, min(1.0, valence))
