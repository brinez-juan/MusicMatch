import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the SBERT model (this may take a few seconds to load)
sbert = SentenceTransformer('all-MiniLM-L6-v2')

########################
# 3. Lyrics → sentiment & embedding
########################
def embed_lyrics(lyrics_text):
    if not lyrics_text:
        return np.zeros(384)
    return sbert.encode(lyrics_text, normalize_embeddings=True)