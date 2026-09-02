# MusicMatch

MusicMatch builds an emotional/audio fingerprint of a song and recommends similar
tracks. For any Spotify song it pulls **Spotify** metadata, **SoundNet** audio
features, and **Musixmatch** lyrics, runs multilingual emotion analysis over the
lyrics, stores the resulting feature row in a CSV, and ranks the most similar
tracks from everything it has ingested so far.

Requires **Python 3.10+**.

## How it works

The pipeline is a linear three-source enrichment, orchestrated by
`src/musicmatch/pipeline.py`:

1. **Spotify** (`spotify.py`) — metadata (title, artist, album, popularity,
   duration). Uses lazy OAuth (the browser consent flow only fires on first real
   call, not at import) and paginates album/playlist URLs.
2. **SoundNet** (`soundnet.py`) — audio features (tempo, energy, danceability,
   happiness, acousticness, etc.) from the RapidAPI track-analysis endpoint, with
   exponential backoff on rate limits.
3. **Musixmatch** (`musixmatch.py`) — lyrics lookup by title + artist, returning
   cleanly when there's no match.
4. **Sentiment** (`sentiment.py`) — a multilingual emotion classifier
   (`MilaNLProc/xlm-emo-t`) that scores lyrics into **anger / fear / joy / sadness**
   probabilities and a derived `[-1, +1]` valence score. It handles languages
   natively (no language detection step) and loads the model lazily on first use.

Each song becomes one feature row in `data/songs.csv` (written atomically and
deduplicated by Spotify track ID via `storage.py`). To recommend, `recommender.py`
standardizes the numeric features with `StandardScaler` — so wide-range features
like `loudness` and `tempo` don't dominate — and ranks candidates by **cosine
similarity**, returning the closest tracks to the target.

### Feature schema

`data/songs.csv` stores, per track:
`spotify_track_id, title, artist, album, release_date, popularity, duration_ms,
tempo, energy, danceability, happiness, acousticness, instrumentalness, liveness,
speechiness, loudness, sentiment_score, emotion_anger, emotion_fear, emotion_joy,
emotion_sadness, language, genres`.

Similarity is computed over the audio/valence features plus the four emotion
columns (see `recommender.FEATURE_COLUMNS`).

## Install

```bash
pip install -e .[test]
```

## Configure

Copy `.env.example` to `.env` and fill in the four API keys:

- Spotify (`SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI`)
  from https://developer.spotify.com/dashboard
- RapidAPI track-analysis (`SOUNDNET_API_KEY`)
  from https://rapidapi.com/Glavier/api/track-analysis
- Musixmatch (`MUSIXMATCH_API_KEY`)
  from https://developer.musixmatch.com/

The first Spotify call opens a browser for OAuth and caches the token in
`.spotipy_cache`.

## Use

```bash
musicmatch ingest <spotify-track-or-album-or-playlist-url>
musicmatch recommend <spotify-track-id-or-url> --top 5
musicmatch list --limit 20
```

Or via module: `python -m musicmatch ingest <url>`.

The dataset lives at `data/songs.csv` by default; override with
`MUSICMATCH_DATASET_PATH` in `.env`.

## Test

```bash
pytest                # default — skips the live-API smoke test
pytest -m smoke       # runs the live end-to-end test (needs a valid .env)
```

## Roadmap / related

The lyric-emotion step currently uses an off-the-shelf 4-emotion model
(`xlm-emo-t`). It's intended to be replaced by **[LyRemo](https://github.com/brinez-juan/LSA-model)**
(repo: `LSA-model`) — a custom multilingual model trained on ~3M song lyrics that
outputs a richer **28-emotion** distribution, which will let MusicMatch match
songs on far finer emotional detail.

## License

MIT — see `LICENSE`.
