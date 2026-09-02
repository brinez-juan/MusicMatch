"""Unit tests for the EmotionScorer. These load the real XLM-T model once per
test session (via the module-scoped ``scorer`` fixture), so the first run on a
fresh machine will download ~500MB of weights. Skipped when
``TRANSFORMERS_OFFLINE=1`` is set so CI environments without network can opt out.
"""
from __future__ import annotations

import math
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("TRANSFORMERS_OFFLINE") == "1",
    reason="TRANSFORMERS_OFFLINE=1 — skipping tests that would download the model.",
)

EXPECTED_LABELS = {"anger", "fear", "joy", "sadness"}


@pytest.fixture(scope="module")
def scorer():
    from musicmatch.sentiment import get_scorer

    return get_scorer()


def test_empty_text_returns_zeros(scorer) -> None:
    result = scorer.score("")
    assert set(result.keys()) == EXPECTED_LABELS
    assert all(v == 0.0 for v in result.values())


def test_whitespace_only_returns_zeros(scorer) -> None:
    result = scorer.score("\n   \n\t  \n")
    assert all(v == 0.0 for v in result.values())


def test_labels_present(scorer) -> None:
    result = scorer.score("hello")
    assert set(result.keys()) == EXPECTED_LABELS
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_batch_matches_single(scorer) -> None:
    texts = [
        "I am so happy and full of joy today",
        "This is terrifying, I am afraid of what comes next",
        "I am so angry at what they did",
    ]
    batched = scorer.score_batch(texts)
    single = [scorer.score(t) for t in texts]

    assert len(batched) == len(single)
    for b, s in zip(batched, single):
        assert set(b.keys()) == set(s.keys())
        for label in b:
            assert math.isclose(b[label], s[label], abs_tol=1e-5), (
                f"{label}: batch={b[label]} single={s[label]}"
            )


def test_batch_handles_empty_entries(scorer) -> None:
    results = scorer.score_batch(["I feel great", "", "   "])
    assert len(results) == 3
    assert all(set(r.keys()) == EXPECTED_LABELS for r in results)
    assert all(v == 0.0 for v in results[1].values())
    assert all(v == 0.0 for v in results[2].values())
    assert any(v > 0.0 for v in results[0].values())


def test_singleton_identity() -> None:
    from musicmatch.sentiment import get_scorer

    assert get_scorer() is get_scorer()
