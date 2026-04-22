from __future__ import annotations

import logging
import re
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from musicmatch.config import REPO_ROOT, Settings


logger = logging.getLogger(__name__)

SCOPE = "user-read-private user-read-email user-top-read user-library-read"
CACHE_PATH = str(REPO_ROOT / ".spotipy_cache")
TRACK_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
URI_RE = re.compile(r"spotify:track:([A-Za-z0-9]+)")
URL_TRACK_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?track/([A-Za-z0-9]+)")
URL_ALBUM_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?album/([A-Za-z0-9]+)")
URL_PLAYLIST_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?playlist/([A-Za-z0-9]+)")

_client: spotipy.Spotify | None = None


def get_client(settings: Settings | None = None) -> spotipy.Spotify:
    global _client
    if _client is None:
        s = settings or Settings.load()
        _client = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=s.spotify_client_id,
            client_secret=s.spotify_client_secret,
            redirect_uri=s.spotify_redirect_uri,
            scope=SCOPE,
            cache_path=CACHE_PATH,
        ))
    return _client


def get_track_id_from_url(value: str) -> str:
    """Extract a Spotify track ID from a URI, URL, or bare ID."""
    m = URI_RE.match(value)
    if m:
        return m.group(1)

    m = URL_TRACK_RE.search(value)
    if m:
        return m.group(1)

    if TRACK_ID_RE.match(value):
        return value

    raise ValueError(f"Invalid Spotify track URL or ID: {value}")


def fetch_spotify_metadata(track_id: str) -> dict[str, Any]:
    metadata = get_client().track(track_id)
    return {
        "title": metadata["name"],
        "artist": metadata["artists"][0]["name"],
        "artist_id": metadata["artists"][0]["id"],
        "popularity": metadata["popularity"],
        "duration_ms": metadata["duration_ms"],
        "release_date": metadata["album"]["release_date"],
        "album": metadata["album"]["name"],
    }


def fetch_artist_genres(artist_id: str) -> list[str]:
    return list(get_client().artist(artist_id).get("genres") or [])


def get_all_tracks_from_url(url: str) -> list[str]:
    """Return all track IDs from a Spotify album or playlist URL, paginated."""
    sp = get_client()

    album_match = URL_ALBUM_RE.search(url)
    if album_match:
        return _paginate_album(sp, album_match.group(1))

    playlist_match = URL_PLAYLIST_RE.search(url)
    if playlist_match:
        return _paginate_playlist(sp, playlist_match.group(1))

    raise ValueError("URL must be an album or playlist.")


def _paginate_album(sp: spotipy.Spotify, album_id: str) -> list[str]:
    ids: list[str] = []
    page = sp.album_tracks(album_id)
    while page:
        ids.extend(t["id"] for t in page["items"] if t.get("id"))
        page = sp.next(page) if page.get("next") else None
    return ids


def _paginate_playlist(sp: spotipy.Spotify, playlist_id: str) -> list[str]:
    ids: list[str] = []
    page = sp.playlist_tracks(playlist_id)
    while page:
        ids.extend(
            item["track"]["id"]
            for item in page["items"]
            if item.get("track") and item["track"].get("id")
        )
        page = sp.next(page) if page.get("next") else None
    return ids
