from integration import fetch_soundnet, fetch_spotify_metadata, build_feature_vector
import pandas as pd
import numpy as np

# ====== STEP 1: Provide IDs manually (replace these) ======
soundnet_track_id = "a396e5ef3e08c870041b67c0d0e7863d"
spotify_track_id = "1DzKS1j1XiKJ5HN9bnFGO0"

# If you have lyrics already:
lyrics = """Baby, yo quisiera quedarme contigo... (etc)"""

# ====== STEP 2: FETCH DATA ======
print("Fetching SoundNet data...")
sn = fetch_soundnet(soundnet_track_id)
print("SoundNet:", sn)   # verify

print("Fetching Spotify metadata...")
sp = fetch_spotify_metadata(spotify_track_id)
print("Spotify:", sp)

# ====== STEP 3: BUILD VECTOR ======
print("Building vector...")
vector = build_feature_vector(sn, sp, lyrics)

print("\nFeature vector created!")
print("Vector length:", len(vector))
print(vector[:20], "...")  # first 20 values

# ====== STEP 4: SAVE TO CSV ======
row = {
    "spotify_id": spotify_track_id,
    "title": sp["title"],
    "artist": sp["artist"],
    "vector": vector.tolist()
}

df = pd.DataFrame([row])
df.to_csv("one_track_features.csv", index=False)
print("\nSaved to one_track_features.csv")