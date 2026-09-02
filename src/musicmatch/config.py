from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "songs.csv"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    soundnet_api_key: str
    musixmatch_api_key: str
    dataset_path: Path

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env")

        required = {
            "SPOTIPY_CLIENT_ID": os.getenv("SPOTIPY_CLIENT_ID"),
            "SPOTIPY_CLIENT_SECRET": os.getenv("SPOTIPY_CLIENT_SECRET"),
            "SPOTIPY_REDIRECT_URI": os.getenv("SPOTIPY_REDIRECT_URI"),
            "SOUNDNET_API_KEY": os.getenv("SOUNDNET_API_KEY"),
            "MUSIXMATCH_API_KEY": os.getenv("MUSIXMATCH_API_KEY"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ConfigError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )

        dataset_override = os.getenv("MUSICMATCH_DATASET_PATH")
        dataset_path = Path(dataset_override) if dataset_override else DEFAULT_DATASET_PATH

        return cls(
            spotify_client_id=required["SPOTIPY_CLIENT_ID"],
            spotify_client_secret=required["SPOTIPY_CLIENT_SECRET"],
            spotify_redirect_uri=required["SPOTIPY_REDIRECT_URI"],
            soundnet_api_key=required["SOUNDNET_API_KEY"],
            musixmatch_api_key=required["MUSIXMATCH_API_KEY"],
            dataset_path=dataset_path,
        )
