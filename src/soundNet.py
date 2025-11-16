import requests
import time 
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    SOUNDNET_API_KEY
)

""" 
SoundNet extraction - recieves the ID for a song and shows the audio features

    Takes the rapidapi key for Soundnet from the config file and makes a request to the API
    to fecth the audio analysis of a given track

    It has error handling. The API limits the number of requests someone can do at the same time. 
    If the request status code equals 200 then ir proceeds normally, but if it's 429 it recieved too
    many requests, therefore waits and trys again automatically

"""
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

