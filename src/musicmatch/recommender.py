from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from musicmatch.storage import SongStore


FEATURE_COLUMNS = [
    "tempo",
    "energy",
    "danceability",
    "happiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "sentiment_score",
]

# Perceptual weights: valence/energy/danceability dominate how "similar" two tracks
# feel; liveness and loudness matter less. Sentiment gets an emphasis boost.
FEATURE_WEIGHTS = np.array([
    1.0,  # tempo
    1.5,  # energy
    1.5,  # danceability
    1.5,  # happiness
    1.0,  # acousticness
    1.0,  # instrumentalness
    0.5,  # liveness
    1.0,  # speechiness
    0.5,  # loudness
    1.5,  # sentiment_score
])

LANGUAGE_BONUS = 0.25  # +bonus if match, -bonus if mismatch, 0 if unknown either side
GENRE_BONUS_MAX = 0.25  # scaled by Jaccard(target_genres, candidate_genres)


def _split_genres(value: Any) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    return {g.strip() for g in value.split("|") if g.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _language_bonus(target: str, other: str) -> float:
    if not target or not other:
        return 0.0
    return LANGUAGE_BONUS if target == other else -LANGUAGE_BONUS


def find_similar_tracks(target_id: str, store: SongStore, top_n: int = 5) -> list[dict[str, Any]]:
    df = store.dataframe()
    if df.empty:
        raise ValueError("Dataset is empty — ingest some tracks first.")
    if target_id not in df["spotify_track_id"].values:
        raise ValueError(f"Track {target_id} not in dataset. Ingest it first.")
    if len(df) < 2:
        raise ValueError("Need at least 2 tracks in the dataset to compute similarity.")

    # Legacy rows stored loudness as e.g. "-6 dB"; strip units before coercing.
    features = df[FEATURE_COLUMNS].copy()
    if features["loudness"].dtype == object:
        features["loudness"] = (
            features["loudness"].astype(str).str.replace("dB", "", regex=False).str.strip()
        )
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Standardize so loudness (~ -60..0) and tempo (~50..200) don't dominate the
    # [0, 100] features in the euclidean distance.
    scaled = StandardScaler().fit_transform(features.values)
    weighted = scaled * FEATURE_WEIGHTS

    target_idx = int(df.index[df["spotify_track_id"] == target_id][0])
    diffs = weighted - weighted[target_idx]
    dists = np.linalg.norm(diffs, axis=1)

    # Gaussian kernel keyed to the median non-zero distance — gives [0, 1] similarity
    # that stays comparable as the dataset grows.
    sigma = max(float(np.median(dists[dists > 0])), 1e-6)
    audio_sim = np.exp(-(dists ** 2) / (2 * sigma ** 2))

    target_language = str(df.iloc[target_idx].get("language") or "")
    target_genres = _split_genres(df.iloc[target_idx].get("genres"))

    languages = df.get("language", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
    genres_col = df.get("genres", pd.Series([""] * len(df))).fillna("").astype(str).tolist()

    combined = np.empty(len(df))
    for i in range(len(df)):
        lang_b = _language_bonus(target_language, languages[i])
        genre_b = _jaccard(target_genres, _split_genres(genres_col[i])) * GENRE_BONUS_MAX
        combined[i] = float(audio_sim[i]) + lang_b + genre_b

    ranked = combined.argsort()[::-1]

    out: list[dict[str, Any]] = []
    seen_artists: set[str] = set()
    target_artist = str(df.iloc[target_idx]["artist"])
    for i in ranked:
        idx = int(i)
        if idx == target_idx:
            continue
        artist = str(df.iloc[idx]["artist"])
        if artist == target_artist or artist in seen_artists:
            continue
        seen_artists.add(artist)
        out.append({
            "spotify_track_id": df.iloc[idx]["spotify_track_id"],
            "title": df.iloc[idx]["title"],
            "artist": artist,
            "similarity": float(combined[idx]),
        })
        if len(out) >= top_n:
            break
    return out
