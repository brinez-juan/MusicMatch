from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from langdetect import LangDetectException, detect


logger = logging.getLogger(__name__)

EN_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
ES_MODEL = "pysentimiento/robertuito-sentiment-analysis"

# nlptown outputs 5 classes (1★..5★) -> map to [-1, +1] via (k-3)/2.
EN_CLASS_WEIGHTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
# pysentimiento outputs [NEG, NEU, POS] -> P(POS) - P(NEG).
ES_CLASS_WEIGHTS = np.array([-1.0, 0.0, 1.0])

_models: dict[str, dict[str, Any]] = {}


def _load_en() -> dict[str, Any]:
    if "en" not in _models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info("Loading English sentiment model: %s", EN_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(EN_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL)
        model.eval()
        _models["en"] = {"tokenizer": tokenizer, "model": model}
    return _models["en"]


def _load_es() -> dict[str, Any]:
    if "es" not in _models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info("Loading Spanish sentiment model: %s", ES_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(ES_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(ES_MODEL)
        model.eval()
        _models["es"] = {"tokenizer": tokenizer, "model": model}
    return _models["es"]


def _score_with(bundle: dict[str, Any], text: str, weights: np.ndarray) -> float:
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=0,
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return float((probs @ weights).mean())


def get_continuous_sentiment(text: str) -> float:
    """Return continuous sentiment in [-1.0, +1.0] for a full lyric text.

    Detects the dominant language once on the whole text (more reliable than per-line
    on short utterances) and scores the entire text, chunked to fit the model.
    """
    clean = "\n".join(l.strip() for l in text.splitlines() if l.strip())
    if not clean:
        return 0.0

    try:
        lang = detect(clean)
    except LangDetectException:
        lang = "en"

    if lang.startswith("es"):
        return _score_with(_load_es(), clean, ES_CLASS_WEIGHTS)
    return _score_with(_load_en(), clean, EN_CLASS_WEIGHTS)
