import requests
import time 
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    SOUNDNET_API_KEY
)

########################
# 1. Fetch from SoundNet
########################
def fetch_soundnet(track_id):

    url = f"https://track-analysis.p.rapidapi.com/pktx/spotify/{track_id}"
    headers = {
        "x-rapidapi-key": SOUNDNET_API_KEY,
        "x-rapidapi-host": "track-analysis.p.rapidapi.com",
    }

    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        return r.json()

    if r.status_code == 429:
        print("[429] Too many requests. Waiting 1.5s and retrying...")
        time.sleep(3)
        return fetch_soundnet(track_id)

    raise Exception(f"SoundNet error {r.status_code} — {r.text}")


"""
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
"""
# TEST CALL
if __name__ == "__main__":
    test_id = "0VjIjW4GlUZAMYd2vXMi3b"  
    print(fetch_soundnet(test_id))

