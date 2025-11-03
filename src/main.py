from auth import create_spotify_client

def main():
    sp = create_spotify_client()
    user = sp.current_user()

    print("\n✅ Conexión exitosa con Spotify API")
    print("----------------------------------")
    print(f"Nombre: {user['display_name']}")
    print(f"País: {user['country']}")
    print(f"Email: {user['email']}")
    print(f"Seguidores: {user['followers']['total']}")
    print(f"ID de usuario: {user['id']}")

if __name__ == "__main__":
    main()
