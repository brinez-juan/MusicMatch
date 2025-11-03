import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

def create_spotify_client():
    """
    Crea una instancia autenticada del cliente de Spotify usando credenciales del archivo .env
    """
    load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
    scope = os.getenv("SPOTIPY_SCOPE")

    if not all([client_id, client_secret, redirect_uri, scope]):
        raise ValueError("Faltan variables de entorno. Verifica tu archivo .env")

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope
    ))

    return sp
