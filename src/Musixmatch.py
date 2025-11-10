import requests
from config import MUSIXMATCH_API_KEY
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langdetect import detect
import numpy as np
import torch
import torch.nn.functional as F

BASE_URL = "https://api.musixmatch.com/ws/1.1"


def search_track_on_musixmatch(song_title, artist_name):
    url = f"{BASE_URL}/track.search"
    params = {
        "q_track": song_title,
        "q_artist": artist_name,
        "apikey": MUSIXMATCH_API_KEY,
        "page_size": 1,
        "s_track_rating": "desc"
    }
    r = requests.get(url, params=params).json()
    track_list = r.get("message", {}).get("body", {}).get("track_list", [])
    return track_list[0]["track"]["track_id"] if track_list else None


def fetch_lyrics(track_id):
    url = f"{BASE_URL}/track.lyrics.get"
    params = {"track_id": track_id, "apikey": MUSIXMATCH_API_KEY}
    r = requests.get(url, params=params).json()
    lyrics = r.get("message", {}).get("body", {}).get("lyrics", {}).get("lyrics_body", "")
    return lyrics.split("*******")[0].strip()


def get_lyrics(song_title, artist_name):
    print(f"Searching Musixmatch for: {song_title} — {artist_name}")
    track_id = search_track_on_musixmatch(song_title, artist_name)
    if not track_id:
        print("Track not found.")
        return None

    print("Found track_id:", track_id)
    return fetch_lyrics(track_id)


# ---- English model ----
tokenizer_en = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_en = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
pipe_en = pipeline("sentiment-analysis", model=model_en, tokenizer=tokenizer_en)

# ---- Spanish model ----
tokenizer_es = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
model_es = AutoModelForSequenceClassification.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
pipe_es = pipeline("sentiment-analysis", model=model_es, tokenizer=tokenizer_es)

def get_continuous_sentiment(text):
    """
    Hybrid English/Spanish sentiment:
    Returns continuous sentiment between -1.0 (very negative) and +1.0 (very positive)
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    scores = []

    for line in lines:
        try:
            lang = detect(line)
        except:
            lang = "en"

        # ----- English -----
        if lang.startswith("en"):
            result = pipe_en(line)[0]
            stars = int(result["label"][0])  # "3 stars" → 3
            score = (stars - 3) / 2          # map to -1..+1
            scores.append(score * result["score"])  # weight by model confidence

        # ----- Spanish -----
        elif lang.startswith("es"):
            # Tokenize and get logits
            inputs = tokenizer_es(line, return_tensors="pt")
            with torch.no_grad():
                logits = model_es(**inputs).logits
                probs = F.softmax(logits, dim=-1)[0].numpy()  # probability vector

            # Labels: ['NEG', 'NEU', 'POS'] → map to continuous -1..0..+1
            score = probs[0]*-1 + probs[1]*0 + probs[2]*1
            scores.append(score)

        # ----- Unknown / other language -----
        else:
            scores.append(0.0)

    return float(np.mean(scores))

if __name__ == "__main__":
    lyrics = get_lyrics("Freak on a Leash", "Korn")
    print("\n=== LYRICS ===\n")
    print(lyrics)

    sentiment = get_continuous_sentiment(lyrics)
    print("\nSentiment Score:", sentiment)


