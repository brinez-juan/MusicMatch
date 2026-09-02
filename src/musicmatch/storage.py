from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class SongStore:
    """CSV-backed store of song feature rows, deduplicated by spotify_track_id."""

    def __init__(self, path: Path):
        self.path = path
        self._df: pd.DataFrame = self._load()

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            df = pd.read_csv(self.path)
            logger.info("Loaded %d songs from %s", len(df), self.path)
            return df
        logger.info("No existing dataset at %s — starting fresh", self.path)
        return pd.DataFrame()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def contains(self, track_id: str) -> bool:
        if self._df.empty:
            return False
        return track_id in self._df["spotify_track_id"].values

    def update(self, track_id: str, patch: dict[str, Any]) -> bool:
        """Apply `patch` to the row with this track_id and persist. Returns False if not found."""
        if self._df.empty:
            return False
        mask = self._df["spotify_track_id"] == track_id
        if not mask.any():
            return False
        for col, value in patch.items():
            if col not in self._df.columns:
                self._df[col] = ""
            self._df.loc[mask, col] = value
        self._write()
        return True

    def add(self, row: dict[str, Any]) -> bool:
        """Append a row and persist. Returns False if track was already present."""
        if self.contains(row["spotify_track_id"]):
            logger.warning("Skip duplicate: %s — %s", row["title"], row["artist"])
            return False

        new = pd.DataFrame([row])
        self._df = new if self._df.empty else pd.concat([self._df, new], ignore_index=True)
        self._write()
        logger.info("Saved: %s — %s", row["title"], row["artist"])
        return True

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, self.path)
