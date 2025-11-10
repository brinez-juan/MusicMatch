import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from integration import build_feature_dataframe

########################
# Similarity / Recommendation Module
########################

def find_similar_tracks(target_track_id, all_tracks_results, top_n=5):
    """
    Given a target track ID and a list of track dictionaries (from build_feature_dataframe),
    returns the top N most similar tracks based on the unified feature vector.
    
    Args:
        target_track_id (str): Spotify track ID of the target
        all_tracks_results (list of dict): List of track data (from build_feature_dataframe)
        top_n (int): Number of similar tracks to return
        
    Returns:
        list of dict: Top N most similar tracks with similarity scores
    """
    # Find target vector
    target_track = None
    for track in all_tracks_results:
        if track['spotify_track_id'] == target_track_id:
            target_track = track
            break
    
    if not target_track:
        raise ValueError(f"Target track ID {target_track_id} not found in dataset.")
    
    target_vector = target_track['feature_vector'].reshape(1, -1)
    
    # Build array of all vectors
    all_vectors = np.array([track['feature_vector'] for track in all_tracks_results])
    
    # Compute cosine similarity
    similarities = cosine_similarity(target_vector, all_vectors)[0]
    
    # Sort and get top N (skip the target itself if present)
    similar_indices = similarities.argsort()[::-1]
    
    recommendations = []
    count = 0
    for idx in similar_indices:
        candidate = all_tracks_results[idx]
        if candidate['spotify_track_id'] == target_track_id:
            continue  # skip target itself
        recommendations.append({
            "spotify_track_id": candidate['spotify_track_id'],
            "title": candidate['title'],
            "artist": candidate['artist'],
            "similarity": similarities[idx]
        })
        count += 1
        if count >= top_n:
            break
    
    return recommendations

########################
# Example Usage
########################
if __name__ == "__main__":
    # Suppose you already built the dataframe and results for multiple tracks
    track_ids = ["0VjIjW4GlUZAMYd2vXMi3b", "29bl4Sr23RrFR0o8mSvPJ2", "1U3A66OHQyTu4N2QTMsP86"]
    df, results = build_feature_dataframe(track_ids)
    
    # Pick a target track
    target_id = "0VjIjW4GlUZAMYd2vXMi3b"
    
    similar_tracks = find_similar_tracks(target_id, results, top_n=2)
    
    print(f"\nTop 2 tracks similar to {target_id} ({results[0]['title']}):")
    for i, rec in enumerate(similar_tracks, start=1):
        print(f"{i}. {rec['title']} by {rec['artist']} (Similarity: {rec['similarity']:.3f})")
