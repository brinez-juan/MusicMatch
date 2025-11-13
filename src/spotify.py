from spotipy.oauth2 import SpotifyOAuth #spotify authentication library
import spotipy #spotify functions library
import re

# Configuration variables
from config import ( 
    SPOTIPY_CLIENT_ID, 
    SPOTIPY_CLIENT_SECRET, 
    SPOTIPY_REDIRECT_URI
)

""" 
Spotify authentication - alows access to Spotify API 

    Takes the client ID, client secret, and redirect URI from the config file wich are 
    obtained from the Spotify Developer Dashboard when creating an app.

    Defines the scope of access required, in this case to read user private data, email, 
    top tracks, and library.

"""

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-private user-read-email user-top-read user-library-read",
    cache_path=".spotipy_cache"
))

"""
Retrieve metadata for a Spotify track by its ID.

    The function recieves the Spotify track ID as input and uses the Spotipy library
    to return metadata such as title, artist, popularity, duration, release date, and 
    album name.

"""

def fetch_spotify_metadata(spotify_track_id):
    metadata = sp.track(spotify_track_id)
    return {
        "title": metadata["name"],
        "artist": metadata["artists"][0]["name"],
        "popularity": metadata["popularity"],
        "duration_ms": metadata["duration_ms"],
        "release_date": metadata["album"]["release_date"],
        "album": metadata["album"]["name"]  
    }

def get_all_tracks_from_url(url):
    """
    Returns a list of track IDs from a Spotify album or playlist URL.
    """
    album_match = re.search(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?album/([\w\d]+)", url)
    playlist_match = re.search(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?playlist/([\w\d]+)", url)

    if album_match:
        album_id = album_match.group(1)
        results = sp.album_tracks(album_id)
        return [t["id"] for t in results["items"]]

    elif playlist_match:
        playlist_id = playlist_match.group(1)
        results = sp.playlist_tracks(playlist_id)
        return [t["track"]["id"] for t in results["items"] if t["track"]]

    else:
        raise ValueError("URL must be an album or playlist.")


# TEST CALL
if __name__ == "__main__":
    test_id = "0VjIjW4GlUZAMYd2vXMi3b"  # Blinding Lights
    print(fetch_spotify_metadata(test_id))