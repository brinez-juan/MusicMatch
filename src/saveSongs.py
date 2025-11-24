import os
import pandas as pd

# ===== Global cache =====
CACHE = None
FILENAME = "songs.csv"


def load_songs(filename=FILENAME):
    """
    Loads all saved songs from CSV into a DataFrame.
    Returns an empty DataFrame if file doesn't exist.
    """
    global CACHE
    if CACHE is not None:
        return CACHE

    if os.path.exists(filename):
        CACHE = pd.read_csv(filename)
        print(f"✓ Loaded {len(CACHE)} songs from {filename}.")
    else:
        CACHE = pd.DataFrame()
        print("⚠ No existing songs found, starting fresh.")

    return CACHE


def song_exists(track_id, filename=FILENAME):
    """
    Checks if a song with the given Spotify track ID already exists in the dataset.
    """
    df = load_songs(filename)
    return not df.empty and track_id in df["spotify_track_id"].values


def get_song_by_id(track_id, filename=FILENAME):
    """
    Returns a dictionary of the song data for a given Spotify track ID if it exists.
    """
    df = load_songs(filename)
    if df.empty:
        return None
    match = df[df["spotify_track_id"] == track_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def save_song_data(song_data, filename=FILENAME):
    """
    Saves a single song's data (dictionary) into a local CSV file.
    Automatically creates the file if it doesn't exist and prevents duplicates.
    """
    global CACHE
    new_row = pd.DataFrame([song_data])

    # Load or create DataFrame
    if os.path.exists(filename):
        df = load_songs(filename)
        if song_data["spotify_track_id"] in df["spotify_track_id"].values:
            print(f"⚠ Song {song_data['title']} by {song_data['artist']} already exists in {filename}.")
            return
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # Save to CSV
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    CACHE = None  # invalidate cache to force reload on next call
    print(f"✓ Song '{song_data['title']}' by {song_data['artist']} saved to {filename}.")

