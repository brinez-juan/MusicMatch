"""End-to-end smoke test against the live APIs.

Run with: pytest -m smoke

Skipped automatically if any required environment variable is missing.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from musicmatch.config import ConfigError, Settings
from musicmatch.pipeline import process_track
from musicmatch.recommender import FEATURE_COLUMNS, find_similar_tracks
from musicmatch.storage import SongStore


BLINDING_LIGHTS = "0VjIjW4GlUZAMYd2vXMi3b"


@pytest.fixture(scope="module")
def settings() -> Settings:
    try:
        return Settings.load()
    except ConfigError as e:
        pytest.skip(str(e))


@pytest.mark.smoke
def test_pipeline_and_recommender_end_to_end(settings: Settings, tmp_path: Path) -> None:
    row = process_track(BLINDING_LIGHTS)

    expected_keys = {
        "spotify_track_id", "title", "artist", "album", "release_date",
        "popularity", "duration_ms", "language", "genres", *FEATURE_COLUMNS,
    }
    assert expected_keys.issubset(row.keys())
    assert row["spotify_track_id"] == BLINDING_LIGHTS
    for col in FEATURE_COLUMNS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"

    fixtures = [
        {**row, "spotify_track_id": "fixture_a", "title": "A", "artist": "X",
         "tempo": row["tempo"] + 50, "energy": 0.1, "danceability": 0.1, "happiness": 0.1,
         "acousticness": 0.9, "instrumentalness": 0.5, "liveness": 0.5, "speechiness": 0.5,
         "loudness": -30.0, "sentiment_score": -0.8,
         "emotion_anger": 0.02, "emotion_fear": 0.03, "emotion_joy": 0.05, "emotion_sadness": 0.9,
         "language": "", "genres": ""},
        {**row, "spotify_track_id": "fixture_b", "title": "B", "artist": "Y",
         "language": "", "genres": ""},
        {**row, "spotify_track_id": "fixture_c", "title": "C", "artist": "Z",
         "energy": 1.0 - row["energy"], "language": "", "genres": ""},
    ]

    store_path = tmp_path / "songs.csv"
    pd.DataFrame([row, *fixtures]).to_csv(store_path, index=False, encoding="utf-8-sig")
    store = SongStore(store_path)

    results = find_similar_tracks(BLINDING_LIGHTS, store, top_n=2)
    assert len(results) == 2
    assert all(r["spotify_track_id"] != BLINDING_LIGHTS for r in results)
    assert results[0]["similarity"] >= results[1]["similarity"]
