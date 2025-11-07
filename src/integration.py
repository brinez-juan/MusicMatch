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

###
# 5. Final feature vector

#def build_feature_vector(soundnet_json, spotify_meta, lyrics_text):
    #numeric = build_numeric_vector(soundnet_json)
    #lyr_emb = embed_lyrics(lyrics_text)
    #popularity = spotify_meta["popularity"] / 100.0

    #return np.concatenate([numeric, [popularity], lyr_emb])
###
