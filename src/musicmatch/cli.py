from __future__ import annotations

import argparse
import logging
import sys

from musicmatch.config import ConfigError, Settings
from musicmatch.logging_setup import setup_logging


logger = logging.getLogger(__name__)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from musicmatch.pipeline import process_collection
    from musicmatch.storage import SongStore
    from musicmatch.config import Settings

    store = SongStore(Settings.load().dataset_path)
    added = 0
    for row in process_collection(args.url):
        if store.add(row):
            added += 1
    logger.info("Ingest complete: %d new tracks added (dataset size: %d)", added, len(store))
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    from musicmatch.config import Settings
    from musicmatch.recommender import find_similar_tracks
    from musicmatch.spotify import get_track_id_from_url
    from musicmatch.storage import SongStore

    store = SongStore(Settings.load().dataset_path)
    target_id = get_track_id_from_url(args.track)
    results = find_similar_tracks(target_id, store, top_n=args.top)

    print(f"\nTop {len(results)} tracks similar to {target_id}:")
    for i, rec in enumerate(results, start=1):
        print(f"{i}. {rec['title']} — {rec['artist']}  (similarity: {rec['similarity']:.3f})")
    return 0


def _cmd_backfill(args: argparse.Namespace) -> int:
    import time

    from musicmatch.config import Settings
    from musicmatch.pipeline import RATE_LIMIT_SLEEP_S
    from musicmatch.storage import SongStore
    from musicmatch import musixmatch, spotify

    store = SongStore(Settings.load().dataset_path)
    df = store.dataframe()
    if df.empty:
        print("Dataset is empty — nothing to backfill.")
        return 0

    language_col = df.get("language")
    genres_col = df.get("genres")

    def missing(row_idx: int) -> bool:
        lang = "" if language_col is None else str(language_col.iloc[row_idx] or "")
        gen = "" if genres_col is None else str(genres_col.iloc[row_idx] or "")
        return not lang or not gen

    targets = [i for i in range(len(df)) if missing(i)]
    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        print("All rows already have language and genres — nothing to do.")
        return 0

    logger.info("Backfilling %d rows", len(targets))
    updated = 0
    for n, idx in enumerate(targets, start=1):
        row = df.iloc[idx]
        track_id = str(row["spotify_track_id"])
        title, artist = str(row["title"]), str(row["artist"])
        logger.info("[%d/%d] %s — %s", n, len(targets), title, artist)

        patch: dict[str, object] = {}
        try:
            sp_track = spotify.get_client().track(track_id)
            artist_id = sp_track["artists"][0]["id"]
            patch["genres"] = "|".join(spotify.fetch_artist_genres(artist_id))
        except Exception:
            logger.exception("Spotify lookup failed for %s", track_id)

        try:
            hit = musixmatch.search_track(title, artist)
            patch["language"] = (hit or {}).get("language", "") or ""
        except Exception:
            logger.exception("Musixmatch lookup failed for %s", track_id)

        if patch and store.update(track_id, patch):
            updated += 1

        if n < len(targets):
            time.sleep(RATE_LIMIT_SLEEP_S)

    logger.info("Backfill complete: %d rows updated", updated)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from musicmatch.config import Settings
    from musicmatch.storage import SongStore

    store = SongStore(Settings.load().dataset_path)
    df = store.dataframe()
    if df.empty:
        print("Dataset is empty.")
        return 0

    cols = ["spotify_track_id", "title", "artist", "album"]
    view = df[cols].head(args.limit) if args.limit else df[cols]
    print(view.to_string(index=False))
    print(f"\n({len(df)} tracks total)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musicmatch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch + store features for a track, album, or playlist URL")
    p_ingest.add_argument("url", help="Spotify track / album / playlist URL or track ID")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_rec = sub.add_parser("recommend", help="Find similar tracks for a stored track")
    p_rec.add_argument("track", help="Spotify track URL or ID (must already be in the dataset)")
    p_rec.add_argument("--top", type=int, default=5, help="How many similar tracks to return")
    p_rec.set_defaults(func=_cmd_recommend)

    p_bf = sub.add_parser("backfill", help="Fill missing language/genres columns for existing rows")
    p_bf.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    p_bf.set_defaults(func=_cmd_backfill)

    p_list = sub.add_parser("list", help="Show stored tracks")
    p_list.add_argument("--limit", type=int, default=20, help="Max rows to print (0 = all)")
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
