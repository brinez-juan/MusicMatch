# MusicMatch

Pulls Spotify metadata, SoundNet audio features, and Musixmatch lyrics for a song,
runs hybrid English/Spanish sentiment over the lyrics, stores the resulting feature
row in a CSV, and recommends similar tracks from what's been ingested.

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

## License

See `LICENSE`.
