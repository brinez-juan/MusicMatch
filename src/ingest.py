import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from spotipy.oauth2 import SpotifyOAuth
import spotipy
from config import SOUNDNET_API_KEY, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

# Load model once
sbert = SentenceTransformer("all-MiniLM-L6-v2")

########################
# 1. Fetch from SoundNet
########################
def fetch_soundnet(spotify_track_id):
    """
    Llama al endpoint de SoundNet en RapidAPI para obtener análisis de audio.
    """
    url = f"https://track-analysis.p.rapidapi.com/pktx/spotify/{spotify_track_id}"

    headers = {
        "x-rapidapi-key": SOUNDNET_API_KEY,
        "x-rapidapi-host": "track-analysis.p.rapidapi.com"
    }

    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        raise Exception(f"SoundNet API error: {r.status_code} — {r.text}")

    return r.json()

########################
# 2. Fetch Spotify metadata
########################
def fetch_spotify_metadata(spotify_track_id):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-read-private user-read-email"
    ))
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
        return np.zeros(384)  # same dimension as model
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

    # normalize loudness to 0..1
    loud_norm = (loud + 60) / 60.0

    return np.array([tempo, energy, dance, happy, acoustic, inst, live, speech, loud_norm], dtype=float)

########################
# 5. Make full feature vector
########################
def build_feature_vector(soundnet_json, spotify_meta, lyrics_text):
    numeric = build_numeric_vector(soundnet_json)
    lyr_emb = embed_lyrics(lyrics_text)
    popularity = spotify_meta["popularity"] / 100.0

    # concatenate: [numeric, popularity, lyrics_embedding]
    final_vector = np.concatenate([numeric, [popularity], lyr_emb])
    return final_vector
