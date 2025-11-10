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

# TEST CALL
if __name__ == "__main__":
    test_id = "0VjIjW4GlUZAMYd2vXMi3b"  
    print(fetch_soundnet(test_id))

