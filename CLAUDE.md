# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout note

There are two nested `MusicMatch` directories. The outer one (`C:\Users\cprocesos\Documents\MusicMatch\`) is just a wrapper; everything lives in the inner `MusicMatch/`. The persisted dataset is `data/songs.csv` inside the inner repo (it used to live in the outer dir — this was moved during the cleanup).

## Install / run

The project is an installable package with a console script. After `pip install -e .[test]` from the inner repo:

```bash
musicmatch ingest <spotify-url>            # track / album / playlist
musicmatch recommend <track-id-or-url> [--top 5]
musicmatch list [--limit 20]
```

`python -m musicmatch …` works equivalently. There is no longer any need to `cd src/`; the package is importable from anywhere.

## Tests

```bash
pytest                # default — runs everything except the live-API smoke test
pytest -m smoke       # runs tests/test_smoke.py against live Spotify/SoundNet/Musixmatch
```

The smoke test auto-skips if any required env var is unset.

## Required environment

`src/musicmatch/config.py::Settings.load()` reads from a `.env` file at the inner repo root and validates all keys at startup, listing every missing one in a single `ConfigError`. Required keys (see `.env.example`):

- `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI` — Spotify Developer Dashboard
- `SOUNDNET_API_KEY` — RapidAPI key for `track-analysis.p.rapidapi.com`
- `MUSIXMATCH_API_KEY` — Musixmatch developer API
- `MUSICMATCH_DATASET_PATH` — optional override for the CSV location

Spotify uses OAuth (`SpotifyOAuth`) and caches the token in `.spotipy_cache` at the repo root. The first run opens a browser for user consent.

## Architecture

The pipeline is a linear three-source enrichment, orchestrated by `src/musicmatch/pipeline.py::process_track`:

1. **`spotify.py`** — lazy OAuth client (constructed on first `get_client()` call, *not* at import — so `import musicmatch.spotify` doesn't trigger a browser). `fetch_spotify_metadata(track_id)` returns title/artist/album/popularity/duration. `get_all_tracks_from_url` paginates album/playlist URLs (with optional `/intl-xx/` locale segments) using `sp.next()` until exhausted. `get_track_id_from_url` accepts URI, URL, or bare 22-char ID.
2. **`soundnet.py`** — calls the RapidAPI SoundNet endpoint with the Spotify track ID. Iterative exponential backoff on 429 (max 5 attempts, capped at 30s) — the old recursive retry was replaced because it could loop forever. Raises typed `SoundNetError` on non-200 / non-429 responses.
3. **`musixmatch.py`** — pure HTTP wrapper: `search_track` then `fetch_lyrics_by_id`, exposed via `get_lyrics(title, artist)`. Returns `None` cleanly when the API has no match.
4. **`sentiment.py`** — owns the HuggingFace models (split out of musixmatch.py). **Lazy-loaded**: `_load_en()` / `_load_es()` download weights on first call only, so importing this module is cheap. English lines (per `langdetect`) → `nlptown/bert-base-multilingual-uncased-sentiment` (5-star → [-1, +1] weighted by confidence). Spanish lines → `pysentimiento/robertuito-sentiment-analysis` (NEG/NEU/POS expected value). Other languages → 0.0.

`pipeline.py` flattens the three sources into one dict per track. `storage.py::SongStore` is a small class wrapping the CSV: loads once on construction, `add(row)` deduplicates by `spotify_track_id`, and writes atomically (write to `.tmp` + `os.replace`) so a crash mid-write can't corrupt the dataset.

`recommender.py::find_similar_tracks` reads the CSV via `SongStore`, standardizes `FEATURE_COLUMNS` with `StandardScaler` (critical — without it `loudness` ~ -60..0 and `tempo` ~50..200 dominate the [0,1] features), then ranks by cosine similarity. Returns `top_n` rows excluding the target.

`pipeline.process_collection` sleeps `RATE_LIMIT_SLEEP_S = 7` between tracks when ingesting albums/playlists.

The CLI (`cli.py`) does `Settings.load()` only inside subcommand handlers, so `--help` works without env vars. It catches `ConfigError` (exit 2) and `ValueError` (exit 1) for clean error messages.

## Output schema

`data/songs.csv` columns (from `pipeline.py::process_track`): `spotify_track_id, title, artist, album, release_date, popularity, duration_ms, tempo, energy, danceability, happiness, acousticness, instrumentalness, liveness, speechiness, loudness, sentiment_score`. Written with `utf-8-sig` encoding so Excel renders accents correctly. The numeric subset used for similarity is `recommender.FEATURE_COLUMNS`.
