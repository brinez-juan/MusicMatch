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
    """
    # Spotify URI: spotify:track:<id>
    match_uri = re.match(r"spotify:track:(\w+)", spotify_url_or_id)
    if match_uri:
        return match_uri.group(1)

    # Spotify URL: https://open.spotify.com/track/<id>
    match_url = re.search(r"open\.spotify\.com/track/(\w+)", spotify_url_or_id)
    if match_url:
        return match_url.group(1)

    # Assume it’s already an ID
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
    print(f"  ✓ Title: {spotify_data['title']}")
    print(f"  ✓ Artist: {spotify_data['artist']}")
    print(f"  ✓ Popularity: {spotify_data['popularity']}")

    # ===== 2. FETCH SOUNDNET AUDIO FEATURES =====
    print("\n[2/3] Fetching SoundNet audio features...")
    soundnet_data = fetch_soundnet(spotify_track_id)
    print(f"  ✓ Tempo: {soundnet_data.get('tempo', 'N/A')}")
    print(f"  ✓ Energy: {soundnet_data.get('energy', 'N/A')}")
    print(f"  ✓ Danceability: {soundnet_data.get('danceability', 'N/A')}")

    # ===== 3. FETCH LYRICS & SENTIMENT =====
    print("\n[3/3] Fetching lyrics and sentiment...")
    lyrics = get_lyrics(spotify_data['title'], spotify_data['artist'])
    if lyrics:
        sentiment_score = get_continuous_sentiment(lyrics)
        print(f"  ✓ Lyrics found ({len(lyrics)} characters)")
        print(f"  ✓ Sentiment score: {sentiment_score:.3f}")
    else:
        sentiment_score = 0.0
        lyrics = ""
        print(f"  ⚠ No lyrics found, setting sentiment to 0.0")

    # ===== 4. BUILD UNIFIED FEATURE VECTOR =====
    tempo = float(soundnet_data.get("tempo", 0))
    energy = float(soundnet_data.get("energy", 0)) / 100.0
    danceability = float(soundnet_data.get("danceability", 0)) / 100.0
    happiness = float(soundnet_data.get("happiness", 0)) / 100.0
    acousticness = float(soundnet_data.get("acousticness", 0)) / 100.0
    instrumentalness = float(soundnet_data.get("instrumentalness", 0)) / 100.0
    liveness = float(soundnet_data.get("liveness", 0)) / 100.0
    speechiness = float(soundnet_data.get("speechiness", 0)) / 100.0

    loudness_str = soundnet_data.get("loudness", "-60 dB")
    try:
        loudness = float(loudness_str.replace(" dB", ""))
    except:
        loudness = -60.0
    loudness_norm = (loudness + 60.0) / 60.0

    popularity = spotify_data['popularity'] / 100.0
    sentiment_norm = (sentiment_score + 1.0) / 2.0

    feature_vector = np.array([
        tempo,
        energy,
        danceability,
        happiness,
        acousticness,
        instrumentalness,
        liveness,
        speechiness,
        loudness_norm,
        popularity,
        sentiment_norm
    ], dtype=np.float32)

    result = {
        "spotify_track_id": spotify_track_id,
        "title": spotify_data['title'],
        "artist": spotify_data['artist'],
        "album": spotify_data['album'],
        "release_date": spotify_data['release_date'],
        "popularity": spotify_data['popularity'],
        "duration_ms": spotify_data['duration_ms'],
        "tempo": tempo,
        "energy": energy * 100,
        "danceability": danceability * 100,
        "happiness": happiness * 100,
        "acousticness": acousticness * 100,
        "instrumentalness": instrumentalness * 100,
        "liveness": liveness * 100,
        "speechiness": speechiness * 100,
        "loudness_db": loudness,
        "sentiment_score": sentiment_score,
        "sentiment_label": (
            "Very Negative" if sentiment_score < -0.6 else
            "Negative" if sentiment_score < -0.2 else
            "Neutral" if sentiment_score < 0.2 else
            "Positive" if sentiment_score < 0.6 else
            "Very Positive"
        ),
        "lyrics_available": bool(lyrics),
        "feature_vector": feature_vector,
        "feature_vector_labels": [
            "tempo_norm",
            "energy",
            "danceability",
            "happiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
            "loudness",
            "popularity",
            "sentiment"
        ]
    }

    print(f"\n✓ Feature vector created ({len(feature_vector)} dimensions)")
    return result


# TEST CALL
if __name__ == "__main__":
    test_url = "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"
    result = build_feature_vector(test_url)
    print("\nFINAL FEATURE VECTOR")
    print(f"{result['title']} by {result['artist']}, sentiment: {result['sentiment_score']:.3f}")

    urls = [
        "https://open.spotify.com/track/29bl4Sr23RrFR0o8mSvPJ2",
        "spotify:track:1U3A66OHQyTu4N2QTMsP86",
        "0VjIjW4GlUZAMYd2vXMi3b"
    ]
    df, results = build_feature_dataframe(urls)
    print(df)

