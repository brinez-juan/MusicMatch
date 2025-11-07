# ✅ Create Spotify client ONCE
from spotipy.oauth2 import SpotifyOAuth
import spotipy

from config import ( 
    SPOTIPY_CLIENT_ID, 
    SPOTIPY_CLIENT_SECRET, 
    SPOTIPY_REDIRECT_URI
)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-private user-read-email",
    cache_path=".spotipy_cache"
))

########################
# 2. Fetch Spotify metadata
########################
def fetch_spotify_metadata(spotify_track_id):
    metadata = sp.track(spotify_track_id)
    return {
        "title": metadata["name"],
        "artist": metadata["artists"][0]["name"],
        "popularity": metadata["popularity"],
        "duration_ms": metadata["duration_ms"],
        "release_date": metadata["album"]["release_date"]
    }