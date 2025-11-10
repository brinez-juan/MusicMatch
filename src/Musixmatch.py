import requests
from config import MUSIXMATCH_API_KEY

BASE_URL = "https://api.musixmatch.com/ws/1.1"


def search_track_on_musixmatch(song_title, artist_name):
    """
    Search Musixmatch for the correct track, return a musixmatch track_id.
    """
    url = f"{BASE_URL}/track.search"
    params = {
        "q_track": song_title,
        "q_artist": artist_name,
        "apikey": MUSIXMATCH_API_KEY,
        "page_size": 1,
        "s_track_rating": "desc"
    }

    r = requests.get(url, params=params)
    data = r.json()

    track_list = data.get("message", {}).get("body", {}).get("track_list", [])
    if not track_list:
        return None

    return track_list[0]["track"]["track_id"]


def fetch_lyrics(track_id):
    """
    Given a Musixmatch track_id, fetch full lyrics.
    """
    url = f"{BASE_URL}/track.lyrics.get"
    params = {
        "track_id": track_id,
        "apikey": MUSIXMATCH_API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    lyrics_body = data.get("message", {}).get("body", {}).get("lyrics", {})
    lyrics = lyrics_body.get("lyrics_body", "")

    # Musixmatch attaches "******* This Lyrics…" footer — remove it
    cleaned = lyrics.split("*******")[0].strip()

    return cleaned


def get_lyrics(song_title, artist_name):
    """
    High-level function:
    - search track
    - fetch lyrics
    """
    print(f"Searching Musixmatch for: {song_title} — {artist_name}")
    track_id = search_track_on_musixmatch(song_title, artist_name)

    if not track_id:
        print("Track not found on Musixmatch.")
        return None

    print("Found track_id:", track_id)
    return fetch_lyrics(track_id)


if __name__ == "__main__":
    # Test example
    lyrics = get_lyrics("Blinding Lights", "The Weeknd")
    print("\n=== LYRICS ===\n")
    print(lyrics)
