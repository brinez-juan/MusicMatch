import os
import pandas as pd

def save_song_data(song_data, filename="songs.csv"):
    """
    Saves a single song's data (dictionary) into a local CSV file.
    Automatically creates the file if it doesn't exist and prevents duplicates.
    """
    # Convert dict to DataFrame
    new_row = pd.DataFrame([song_data])

    # If file exists, read it; otherwise create a new one
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        
        # Check for duplicates
        if song_data["spotify_track_id"] in df["spotify_track_id"].values:
            print(f"⚠ Song {song_data['title']} by {song_data['artist']} already exists in {filename}.")
            return
        
        # Append new row
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # Save to CSV
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✓ Song '{song_data['title']}' by {song_data['artist']} saved to {filename}.")


def load_songs(filename="songs.csv"):
    """
    Loads all saved songs from CSV.
    Returns a pandas DataFrame, or an empty one if file doesn't exist.
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print("⚠ No songs found. The file doesn't exist yet.")
        return pd.DataFrame()
