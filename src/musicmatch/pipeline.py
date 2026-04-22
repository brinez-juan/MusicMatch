from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from musicmatch import musixmatch, sentiment, soundnet, spotify


logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEP_S = 7


def _parse_loudness(value: Any) -> float:
    # SoundNet returns loudness as a string like "-6 dB". Strip units so the
    # recommender can treat it as a numeric feature.
    if isinstance(value, str):
        return float(value.replace("dB", "").strip())
    return float(value)


def process_track(url_or_id: str) -> dict[str, Any]:
    """Fetch metadata, audio features, and lyric sentiment for a single track."""
    track_id = spotify.get_track_id_from_url(url_or_id)
    logger.info("Processing track %s", track_id)

    spotify_data = spotify.fetch_spotify_metadata(track_id)
    soundnet_data = soundnet.fetch_soundnet(track_id)

    lyrics, language = musixmatch.get_lyrics_and_language(
        spotify_data["title"], spotify_data["artist"]
    )
    sentiment_score = sentiment.get_continuous_sentiment(lyrics) if lyrics else 0.0

    try:
        genres = spotify.fetch_artist_genres(spotify_data["artist_id"])
    except Exception:
        logger.exception("Could not fetch genres for artist %s", spotify_data["artist_id"])
        genres = []

    return {
        "spotify_track_id": track_id,
        "title": spotify_data["title"],
        "artist": spotify_data["artist"],
        "album": spotify_data["album"],
        "release_date": spotify_data["release_date"],
        "popularity": spotify_data["popularity"],
        "duration_ms": spotify_data["duration_ms"],
        "tempo": soundnet_data["tempo"],
        "energy": soundnet_data["energy"],
        "danceability": soundnet_data["danceability"],
        "happiness": soundnet_data["happiness"],
        "acousticness": soundnet_data["acousticness"],
        "instrumentalness": soundnet_data["instrumentalness"],
        "liveness": soundnet_data["liveness"],
        "speechiness": soundnet_data["speechiness"],
        "loudness": _parse_loudness(soundnet_data["loudness"]),
        "sentiment_score": sentiment_score,
        "language": language or "",
        "genres": "|".join(genres),
    }


def process_collection(url: str) -> Iterator[dict[str, Any]]:
    """Yield processed rows for every track in an album or playlist URL."""
    if "album" in url or "playlist" in url:
        track_ids = spotify.get_all_tracks_from_url(url)
    else:
        track_ids = [spotify.get_track_id_from_url(url)]

    logger.info("Found %d tracks to process", len(track_ids))
    for i, track_id in enumerate(track_ids, start=1):
        logger.info("[%d/%d] %s", i, len(track_ids), track_id)
        try:
            yield process_track(track_id)
        except Exception:
            logger.exception("Failed to process track %s", track_id)
        if i < len(track_ids):
            time.sleep(RATE_LIMIT_SLEEP_S)
