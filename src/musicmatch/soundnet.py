from __future__ import annotations

import logging
import time
from typing import Any

import requests

from musicmatch.config import Settings


logger = logging.getLogger(__name__)

API_HOST = "track-analysis.p.rapidapi.com"
MAX_ATTEMPTS = 5


class SoundNetError(RuntimeError):
    pass


def fetch_soundnet(track_id: str, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or Settings.load()
    url = f"https://{API_HOST}/pktx/spotify/{track_id}"
    headers = {
        "x-rapidapi-key": s.soundnet_api_key,
        "x-rapidapi-host": API_HOST,
    }

    for attempt in range(MAX_ATTEMPTS):
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            delay = min(2 ** attempt, 30)
            logger.warning("SoundNet 429, retrying in %ds (attempt %d/%d)",
                           delay, attempt + 1, MAX_ATTEMPTS)
            time.sleep(delay)
            continue
        raise SoundNetError(f"SoundNet error {r.status_code} — {r.text}")

    raise SoundNetError(f"SoundNet 429 after {MAX_ATTEMPTS} attempts")
