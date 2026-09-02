"""One-off migration: re-fetch lyrics and score emotions for rows in data/songs.csv
that predate the multi-label emotion schema.

Usage:
    python scripts/backfill_emotions.py           # backfill every row missing emotions
    python scripts/backfill_emotions.py --limit 5 # process only the first N missing rows

The script is idempotent: rows that already have a populated ``emotion_anger`` column
are skipped, so you can interrupt and re-run safely. Rate-limited at 7s per row to
stay within Musixmatch's quota — expect ~40 min for a 338-row dataset.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time

from musicmatch import musixmatch, sentiment
from musicmatch.config import ConfigError, Settings
from musicmatch.pipeline import RATE_LIMIT_SLEEP_S
from musicmatch.storage import SongStore


logger = logging.getLogger("backfill_emotions")

EMOTION_COLUMNS = ("emotion_anger", "emotion_fear", "emotion_joy", "emotion_sadness")


def _needs_backfill(row) -> bool:
    for col in EMOTION_COLUMNS:
        value = row.get(col)
        if value is None:
            return True
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return True
        if math.isnan(numeric):
            return True
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N missing rows.")
    args = parser.parse_args()

    try:
        settings = Settings.load()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    store = SongStore(settings.dataset_path)
    df = store.dataframe()
    if df.empty:
        logger.info("Dataset is empty — nothing to backfill.")
        return 0

    pending = [row for _, row in df.iterrows() if _needs_backfill(row)]
    if args.limit is not None:
        pending = pending[: args.limit]
    logger.info("Found %d rows missing emotion data (of %d total).", len(pending), len(df))
    if not pending:
        return 0

    scorer = sentiment.get_scorer()  # warm up once, not in the loop

    processed = 0
    failed = 0
    for i, row in enumerate(pending, start=1):
        track_id = str(row["spotify_track_id"])
        title = str(row.get("title") or "")
        artist = str(row.get("artist") or "")
        logger.info("[%d/%d] %s — %s (%s)", i, len(pending), title, artist, track_id)

        try:
            lyrics, language = musixmatch.get_lyrics_and_language(title, artist, settings)
        except Exception:
            logger.exception("Musixmatch lookup failed for %s", track_id)
            failed += 1
            if i < len(pending):
                time.sleep(RATE_LIMIT_SLEEP_S)
            continue

        if lyrics:
            emotions = scorer.score(lyrics)
        else:
            emotions = {label: 0.0 for label in ("anger", "fear", "joy", "sadness")}
            logger.info("No lyrics — zero-filling emotions.")

        patch = {
            "emotion_anger": emotions["anger"],
            "emotion_fear": emotions["fear"],
            "emotion_joy": emotions["joy"],
            "emotion_sadness": emotions["sadness"],
            "sentiment_score": emotions["joy"]
            - (emotions["anger"] + emotions["fear"] + emotions["sadness"]) / 3.0,
        }
        # Only overwrite language if the existing value is blank.
        if language and not str(row.get("language") or "").strip():
            patch["language"] = language

        if not store.update(track_id, patch):
            logger.warning("store.update returned False for %s (row vanished?)", track_id)
            failed += 1
        else:
            processed += 1

        if i < len(pending):
            time.sleep(RATE_LIMIT_SLEEP_S)

    logger.info("Backfill complete: %d updated, %d failed.", processed, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
