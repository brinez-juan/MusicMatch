import numpy as np
import pandas as pd
from spotify import fetch_spotify_metadata
from soundNet import fetch_soundnet
from Musixmatch import get_lyrics, get_continuous_sentiment
import re

########################
# Integration: Build unified data vector from Spotify track URL or ID
########################

def get_track_id_from_url(spotify_url_or_id):
    """
    Extracts Spotify track ID from a URL or returns the ID if already provided.
    Supports URLs with or without locale segments (e.g. /intl-es/).
    """
    # Spotify URI: spotify:track:<id>
    match_uri = re.match(r"spotify:track:(\w+)", spotify_url_or_id)
    if match_uri:
        return match_uri.group(1)

    # Spotify URL: https://open.spotify.com/(intl-xx/)?track/<id>
    match_url = re.search(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?track/([\w\d]+)", spotify_url_or_id)
    if match_url:
        return match_url.group(1)

    # Assume it's already an ID
    if re.match(r"^[\w\d]{22}$", spotify_url_or_id):
        return spotify_url_or_id

    raise ValueError(f"Invalid Spotify track URL or ID: {spotify_url_or_id}")



def build_feature_vector(spotify_track_url_or_id):
    """
    Fetches metadata, audio features, and sentiment from all three sources
    and returns a unified dictionary with all extracted features.
    """
    spotify_track_id = get_track_id_from_url(spotify_track_url_or_id)
    print(f"\nProcessing track ID: {spotify_track_id}")
    
    # ===== 1. FETCH SPOTIFY METADATA =====
    print("[1/3] Fetching Spotify metadata...")
    spotify_data = fetch_spotify_metadata(spotify_track_id)

    # ===== 2. FETCH SOUNDNET AUDIO FEATURES =====
    print("\n[2/3] Fetching SoundNet audio features...")
    soundnet_data = fetch_soundnet(spotify_track_id)

    # ===== 3. FETCH LYRICS & SENTIMENT =====
    print("\n[3/3] Fetching lyrics and sentiment...")
    lyrics = get_lyrics(spotify_data['title'], spotify_data['artist'])
    if lyrics:
        sentiment_score = get_continuous_sentiment(lyrics)
    else:
        sentiment_score = 0.0
        lyrics = ""


    result = {
        "spotify_track_id": spotify_track_id,
        "title": spotify_data['title'],
        "artist": spotify_data['artist'],
        "album": spotify_data['album'],
        "release_date": spotify_data['release_date'],
        "popularity": spotify_data['popularity'],
        "duration_ms": spotify_data['duration_ms'],
        "tempo": soundnet_data['tempo'],
        "energy": soundnet_data['energy'],
        "danceability": soundnet_data['danceability'],
        "happiness": soundnet_data['happiness'],
        "acousticness": soundnet_data['acousticness'],
        "instrumentalness": soundnet_data['instrumentalness'],
        "liveness": soundnet_data['liveness'],
        "speechiness": soundnet_data['speechiness'],
        "loudness": soundnet_data['loudness'],
        "sentiment_score": sentiment_score,
    }

    print("\n✓ Song features successfully extracted!")

    return result


# TEST CALL
if __name__ == "__main__":
    song = input("Enter Spotify track URL or ID: ")
    features = build_feature_vector(song)
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
    

   

