from spotipy.oauth2 import SpotifyOAuth #spotify authentication library
import spotipy #spotify functions library

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


# TEST CALL
if __name__ == "__main__":
    test_id = "2NQhQlcVAcDRN3jf6FDTlm"  # Blinding Lights
    print(fetch_spotify_metadata(test_id))