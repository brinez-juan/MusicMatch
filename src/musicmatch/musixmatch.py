from __future__ import annotations

import logging

import requests

from musicmatch.config import Settings


logger = logging.getLogger(__name__)

BASE_URL = "https://api.musixmatch.com/ws/1.1"


def search_track(song_title: str, artist_name: str, settings: Settings | None = None) -> dict | None:
    """Return {'track_id': str, 'language': str} for best match, or None."""
    s = settings or Settings.load()
    params = {
        "q_track": song_title,
        "q_artist": artist_name,
        "apikey": s.musixmatch_api_key,
        "page_size": 1,
        "s_track_rating": "desc",
    }
    r = requests.get(f"{BASE_URL}/track.search", params=params, timeout=30).json()
    track_list = r.get("message", {}).get("body", {}).get("track_list", [])
    if not track_list:
        return None
    track = track_list[0]["track"]
    return {
        "track_id": str(track["track_id"]),
        "language": (track.get("track_language") or "").lower(),
    }


def fetch_lyrics_by_id(track_id: str, settings: Settings | None = None) -> str | None:
    s = settings or Settings.load()
    params = {"track_id": track_id, "apikey": s.musixmatch_api_key}
    r = requests.get(f"{BASE_URL}/track.lyrics.get", params=params, timeout=30).json()
    body = r.get("message", {}).get("body", {}).get("lyrics", {}).get("lyrics_body", "")
    if not body:
        return None
    return body.split("*******")[0].strip()


def get_lyrics(song_title: str, artist_name: str, settings: Settings | None = None) -> str | None:
    logger.info("Searching Musixmatch for: %s — %s", song_title, artist_name)
    hit = search_track(song_title, artist_name, settings)
    if not hit:
        logger.info("Track not found on Musixmatch")
        return None
    return fetch_lyrics_by_id(hit["track_id"], settings)


def get_lyrics_and_language(
    song_title: str, artist_name: str, settings: Settings | None = None
) -> tuple[str | None, str]:
    """Same single search call — returns (lyrics, language). Language is "" if unknown."""
    logger.info("Searching Musixmatch for: %s — %s", song_title, artist_name)
    hit = search_track(song_title, artist_name, settings)
    if not hit:
        logger.info("Track not found on Musixmatch")
        return None, ""
    lyrics = fetch_lyrics_by_id(hit["track_id"], settings)
    return lyrics, hit["language"]
