import requests
import time 
import numpy as np
from sentence_transformers import SentenceTransformer
from spotipy.oauth2 import SpotifyOAuth
import spotipy

from config import (
    SOUNDNET_API_KEY, 
    SPOTIPY_CLIENT_ID, 
    SPOTIPY_CLIENT_SECRET, 
    SPOTIPY_REDIRECT_URI
)

# ----- GLOBAL INITIALIZATION -----
sbert = SentenceTransformer("all-MiniLM-L6-v2")
LAST_REQUEST_TIME = 0

# ✅ Create Spotify client ONCE
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-private user-read-email",
    cache_path=".spotipy_cache"
))

########################
# 1. Fetch from SoundNet
########################
def fetch_soundnet(track_id):
    global LAST_REQUEST_TIME

    now = time.time()
    elapsed = now - LAST_REQUEST_TIME
    if elapsed < 1:
        time.sleep(1 - elapsed)

    url = f"https://track-analysis.p.rapidapi.com/pktx/spotify/{track_id}"
    headers = {
        "x-rapidapi-key": SOUNDNET_API_KEY,
        "x-rapidapi-host": "track-analysis.p.rapidapi.com",
        "connection": "close"
    }

    r = requests.get(url, headers=headers)
    LAST_REQUEST_TIME = time.time()

    if r.status_code == 200:
        return r.json()

    if r.status_code == 429:
        print("[429] Too many requests. Waiting 1.5s and retrying...")
        time.sleep(3)
        return fetch_soundnet(track_id)

    raise Exception(f"SoundNet error {r.status_code} — {r.text}")


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


########################
# 3. Lyrics → sentiment & embedding
########################
def embed_lyrics(lyrics_text):
    if not lyrics_text:
        return np.zeros(384)
    return sbert.encode(lyrics_text, normalize_embeddings=True)


########################
# 4. Audio numeric vector
########################
def build_numeric_vector(soundnet_json):
    tempo = float(soundnet_json.get("tempo", 0))
    energy = float(soundnet_json.get("energy", 0)) / 100
    dance = float(soundnet_json.get("danceability", 0)) / 100
    happy = float(soundnet_json.get("happiness", 0)) / 100
    acoustic = float(soundnet_json.get("acousticness", 0)) / 100
    inst = float(soundnet_json.get("instrumentalness", 0)) / 100
    live = float(soundnet_json.get("liveness", 0)) / 100
    speech = float(soundnet_json.get("speechiness", 0)) / 100

    loud_str = soundnet_json.get("loudness", "-60 dB")
    try:
        loud = float(loud_str.replace(" dB", ""))
    except:
        loud = -60.0

    loud_norm = (loud + 60) / 60.0

    return np.array([tempo, energy, dance, happy, acoustic, inst, live, speech, loud_norm], dtype=float)


########################
# 5. Final feature vector
########################
def build_feature_vector(soundnet_json, spotify_meta, lyrics_text):
    numeric = build_numeric_vector(soundnet_json)
    lyr_emb = embed_lyrics(lyrics_text)
    popularity = spotify_meta["popularity"] / 100.0

    return np.concatenate([numeric, [popularity], lyr_emb])

