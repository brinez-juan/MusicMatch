import numpy as np
import pandas as pd
from spotify import fetch_spotify_metadata
from soundNet import fetch_soundnet
from Musixmatch import get_lyrics, get_continuous_sentiment

########################
# Integration: Build unified data vector from Spotify track ID
########################

def build_feature_vector(spotify_track_id):
    """
    Fetches metadata, audio features, and sentiment from all three sources
    and returns a unified dictionary with all extracted features.
    
    Args:
        spotify_track_id (str): Spotify track ID
        
    Returns:
        dict: Combined feature vector with metadata, audio features, and sentiment
        
    Raises:
        Exception: If any API call fails
    """
    
    print(f"\n{'='*60}")
    print(f"Building feature vector for track: {spotify_track_id}")
    print(f"{'='*60}\n")
    
    # ===== 1. FETCH SPOTIFY METADATA =====
    print("[1/3] Fetching Spotify metadata...")
    try:
        spotify_data = fetch_spotify_metadata(spotify_track_id)
        print(f"  ✓ Title: {spotify_data['title']}")
        print(f"  ✓ Artist: {spotify_data['artist']}")
        print(f"  ✓ Popularity: {spotify_data['popularity']}")
    except Exception as e:
        print(f"  ✗ Error fetching Spotify metadata: {e}")
        raise
    
    # ===== 2. FETCH SOUNDNET AUDIO FEATURES =====
    print("\n[2/3] Fetching SoundNet audio features...")
    try:
        soundnet_data = fetch_soundnet(spotify_track_id)
        print(f"  ✓ Tempo: {soundnet_data.get('tempo', 'N/A')}")
        print(f"  ✓ Energy: {soundnet_data.get('energy', 'N/A')}")
        print(f"  ✓ Danceability: {soundnet_data.get('danceability', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Error fetching SoundNet data: {e}")
        raise
    
    # ===== 3. FETCH LYRICS & SENTIMENT =====
    print("\n[3/3] Fetching lyrics and sentiment...")
    try:
        lyrics = get_lyrics(spotify_data['title'], spotify_data['artist'])
        if lyrics:
            sentiment_score = get_continuous_sentiment(lyrics)
            print(f"  ✓ Lyrics found ({len(lyrics)} characters)")
            print(f"  ✓ Sentiment score: {sentiment_score:.3f}")
        else:
            sentiment_score = 0.0
            lyrics = ""
            print(f"  ⚠ No lyrics found, setting sentiment to 0.0")
    except Exception as e:
        print(f"  ⚠ Error fetching lyrics/sentiment: {e}")
        sentiment_score = 0.0
        lyrics = ""
    
    # ===== 4. BUILD UNIFIED FEATURE VECTOR =====
    print("\n" + "="*60)
    print("Building unified feature vector...")
    print("="*60)
    
    # Extract numeric features from SoundNet
    tempo = float(soundnet_data.get("tempo", 0))
    energy = float(soundnet_data.get("energy", 0)) / 100.0  # normalize to 0-1
    danceability = float(soundnet_data.get("danceability", 0)) / 100.0
    happiness = float(soundnet_data.get("happiness", 0)) / 100.0
    acousticness = float(soundnet_data.get("acousticness", 0)) / 100.0
    instrumentalness = float(soundnet_data.get("instrumentalness", 0)) / 100.0
    liveness = float(soundnet_data.get("liveness", 0)) / 100.0
    speechiness = float(soundnet_data.get("speechiness", 0)) / 100.0
    
    # Extract loudness and normalize to 0-1 range (-60 dB to 0 dB)
    loudness_str = soundnet_data.get("loudness", "-60 dB")
    try:
        loudness = float(loudness_str.replace(" dB", ""))
    except:
        loudness = -60.0
    loudness_norm = (loudness + 60.0) / 60.0  # normalize to 0-1
    
    # Extract popularity from Spotify (already 0-100)
    popularity = spotify_data['popularity'] / 100.0
    
    # Sentiment score is already continuous -1 to +1, normalize to 0-1
    sentiment_norm = (sentiment_score + 1.0) / 2.0
    
    # Create feature vector (numeric features only)
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
    
    # Create comprehensive return dictionary
    result = {
        # Metadata
        "spotify_track_id": spotify_track_id,
        "title": spotify_data['title'],
        "artist": spotify_data['artist'],
        "album": spotify_data['album'],
        "release_date": spotify_data['release_date'],
        "popularity": spotify_data['popularity'],
        "duration_ms": spotify_data['duration_ms'],
        
        # Audio Features (raw values)
        "tempo": tempo,
        "energy": energy * 100,  # keep original scale for display
        "danceability": danceability * 100,
        "happiness": happiness * 100,
        "acousticness": acousticness * 100,
        "instrumentalness": instrumentalness * 100,
        "liveness": liveness * 100,
        "speechiness": speechiness * 100,
        "loudness_db": loudness,
        
        # Sentiment Analysis
        "sentiment_score": sentiment_score,  # -1 to +1
        "sentiment_label": (
            "Very Negative" if sentiment_score < -0.6 else
            "Negative" if sentiment_score < -0.2 else
            "Neutral" if sentiment_score < 0.2 else
            "Positive" if sentiment_score < 0.6 else
            "Very Positive"
        ),
        "lyrics_available": bool(lyrics),
        
        # Unified Feature Vector (normalized 0-1)
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
    
    # Print summary
    print(f"\n✓ Feature vector created ({len(feature_vector)} dimensions)")
    print(f"  Shape: {feature_vector.shape}")
    print(f"  Range: [{feature_vector.min():.3f}, {feature_vector.max():.3f}]")
    
    return result


def build_feature_dataframe(spotify_track_ids):
    """
    Builds feature vectors for multiple tracks and returns as DataFrame.
    
    Args:
        spotify_track_ids (list): List of Spotify track IDs
        
    Returns:
        pd.DataFrame: DataFrame with one row per track and feature columns
    """
    results = []
    
    for i, track_id in enumerate(spotify_track_ids):
        print(f"\n[{i+1}/{len(spotify_track_ids)}] Processing track...")
        try:
            feature_dict = build_feature_vector(track_id)
            results.append(feature_dict)
        except Exception as e:
            print(f"  ✗ Skipped track {track_id}: {e}")
            continue
    
    # Create DataFrame with main features (not the vector itself)
    df = pd.DataFrame([
        {
            "spotify_track_id": r["spotify_track_id"],
            "title": r["title"],
            "artist": r["artist"],
            "album": r["album"],
            "popularity": r["popularity"],
            "tempo": r["tempo"],
            "energy": r["energy"],
            "danceability": r["danceability"],
            "happiness": r["happiness"],
            "acousticness": r["acousticness"],
            "instrumentalness": r["instrumentalness"],
            "liveness": r["liveness"],
            "speechiness": r["speechiness"],
            "loudness_db": r["loudness_db"],
            "sentiment_score": r["sentiment_score"],
            "sentiment_label": r["sentiment_label"],
            "lyrics_available": r["lyrics_available"],
        }
        for r in results
    ])
    
    return df, results  # return both DataFrame and full result dicts with vectors


# TEST CALL
if __name__ == "__main__":
    # Test with a single track
    test_id = "0VjIjW4GlUZAMYd2vXMi3b"  # Blinding Lights - The Weeknd
    
    try:
        result = build_feature_vector(test_id)
        
        print("\n\n" + "="*60)
        print("FINAL FEATURE VECTOR")
        print("="*60)
        print(f"Track: {result['title']} by {result['artist']}")
        print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
        print(f"\nNumeric Features (normalized 0-1):")
        for label, value in zip(result['feature_vector_labels'], result['feature_vector']):
            print(f"  {label:20s}: {value:.3f}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

    track_ids = ["29bl4Sr23RrFR0o8mSvPJ2", "1U3A66OHQyTu4N2QTMsP86"]  # Example IDs
    df, results = build_feature_dataframe(track_ids)
    print(df)
