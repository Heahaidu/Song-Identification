import os
import base64
import requests
import json
import sqlite3
import time
import sys
import os.path

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
DB_FILE = "songs.db"
TOKEN_FILE = "token.json"
LIMIT = 50  # Hằng số limit chung cho các yêu cầu API (search, playlist tracks)
REQUEST_DELAY = 2.0  # Thời gian chờ giữa các yêu cầu API (giây)

QUERIES = [
    "vpop", "lofi", "reverb", "chill",
    "Sơn Tùng M-TP", "Mỹ Tâm", "Hồ Ngọc Hà", "Đen Vâu", "Chillies", "14 casper", "soobin", "changg", "buitruonglinh", "ponk", "babydoll",
]
PLAYLIST_IDS = [
    "0aiBKNSqiPnhtcw1QlXK5s"
]

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            song_name TEXT NOT NULL,
            artists TEXT, -- JSON list of artist IDs
            external_urls TEXT,
            image_url TEXT,
            duration_ms INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS scrape_state (
            id INTEGER PRIMARY KEY,
            query TEXT,
            market TEXT,
            offset INTEGER,
            playlist_id TEXT,
            query_index INTEGER,
            playlist_index INTEGER,
            last_updated TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS artists (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_token():
    # Kiểm tra file token.json
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            token_data = json.load(f)
            access_token = token_data.get('access_token')
            expires_at = token_data.get('expires_at', 0)
            # Nếu token chưa hết hạn, trả về token
            if access_token and expires_at > time.time():
                return access_token

    # Nếu không có token hoặc token hết hạn, gọi API để lấy token mới
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + auth_base64,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {"grant_type": "client_credentials"}

    result = requests.post(url, headers=headers, data=data)
    result.raise_for_status()
    token_data = result.json()
    access_token = token_data["access_token"]
    expires_in = token_data.get("expires_in", 3600)  # Mặc định 3600 giây nếu không có expires_in
    expires_at = time.time() + expires_in - 60  # Trừ 60 giây để an toàn

    # Lưu token vào file token.json
    with open(TOKEN_FILE, 'w') as f:
        json.dump({
            'access_token': access_token,
            'expires_at': expires_at
        }, f, indent=4)

    return access_token

def search(query, market='VN', limit=LIMIT, offset=0):
    url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': 'Bearer ' + get_token(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    params = {
        'q': query,
        'type': 'track',
        'market': market,
        'limit': limit,
        'offset': offset
    }

    result = requests.get(url, headers=headers, params=params)
    result.raise_for_status()
    return result.json().get('tracks', {}).get('items', [])

def get_playlist_tracks(playlist_id, market='VN', limit=LIMIT, offset=0):
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {
        'Authorization': 'Bearer ' + get_token(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    params = {
        'market': market,
        'limit': limit,
        'offset': offset
    }

    result = requests.get(url, headers=headers, params=params)
    result.raise_for_status()
    return [item['track'] for item in result.json().get('items', []) if item['track']]

def get_artist_top_tracks(artist_id, market='VN'):
    url = f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks'
    headers = {
        'Authorization': 'Bearer ' + get_token(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    params = {
        'market': market
    }

    result = requests.get(url, headers=headers, params=params)
    result.raise_for_status()
    return result.json().get('tracks', [])

def check_song_exists(song_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM tracks WHERE id = ?", (song_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def check_artist_exists(artist_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM artists WHERE id = ?", (artist_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def save_artist(artist_id, artist_name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO artists (id, name)
        VALUES (?, ?)
    ''', (artist_id, artist_name))
    conn.commit()
    conn.close()

def save_song(song):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO tracks (id, song_name, artists, external_urls, image_url, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        song['id'],
        song['song_name'],
        song['artists'],
        song['external_urls'],
        song['image_url'],
        song['duration_ms']
    ))
    conn.commit()
    conn.close()

def save_state(query, market, offset, playlist_id, query_index, playlist_index):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO scrape_state (id, query, market, offset, playlist_id, query_index, playlist_index, last_updated)
        VALUES (1, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (query, market, offset, playlist_id, query_index, playlist_index))
    conn.commit()
    conn.close()

def load_state():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT query, market, offset, playlist_id, query_index, playlist_index FROM scrape_state WHERE id = 1")
    state = c.fetchone()
    conn.close()
    return state if state else (QUERIES[0], 'VN', 0, None, 0, 0)

def scrape_artist_songs(artist_id, artist_name, market='VN', total_songs=0, max_songs=10000):
    if check_artist_exists(artist_id):
        print(f"Đã bỏ qua nghệ sĩ đã tồn tại: {artist_name} (ID: {artist_id})")
        return total_songs

    songs_added = 0
    try:
        tracks = get_artist_top_tracks(artist_id, market)
        
        for track in tracks:
            if total_songs >= max_songs:
                break

            song_id = track['id']
            if check_song_exists(song_id):
                print(f"Bỏ qua bài hát đã tồn tại: {song_id}")
                continue

            track_artists = json.dumps([artist['id'] for artist in track['artists']])
            song = {
                'id': song_id,
                'song_name': track['name'],
                'artists': track_artists,
                'external_urls': track['external_urls']['spotify'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else '',
                'duration_ms': track['duration_ms']
            }

            save_song(song)
            total_songs += 1
            songs_added += 1
            print(f"Đã lưu bài hát top track: {song['song_name']} (ID: {song_id}, Total: {total_songs}/{max_songs})")

        if songs_added > 0:
            save_artist(artist_id, artist_name)
            print(f"Đã lưu nghệ sĩ: {artist_name} (ID: {artist_id}) sau khi lưu {songs_added} bài hát")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get('Retry-After', 10))
            print(f"Lỗi 429: Quá nhiều yêu cầu khi lấy top tracks của nghệ sĩ {artist_id}. Chờ {retry_after} giây trước khi dừng.")
            time.sleep(retry_after)
        raise

    time.sleep(REQUEST_DELAY)
    return total_songs

def scrape_songs(max_songs=20000, batch_size=50):
    init_db()
    total_songs = 0
    query, market, offset, playlist_id, query_index, playlist_index = load_state()

    print(f"Bắt đầu cào từ query '{query}', playlist '{playlist_id}', offset {offset}, query_index {query_index}, playlist_index {playlist_index}")

    try:
        # while playlist_index < len(PLAYLIST_IDS) and total_songs < max_songs:
        #     playlist_id = PLAYLIST_IDS[playlist_index]
        #     print(f"Cào từ playlist: {playlist_id}")
        #     while total_songs < max_songs:
        #         items = get_playlist_tracks(playlist_id, market, limit=LIMIT, offset=offset)
        #         if not items:
        #             print(f"Hết bài hát trong playlist '{playlist_id}'")
        #             offset = 0
        #             playlist_index += 1
        #             playlist_id = None
        #             save_state(query, market, offset, playlist_id, query_index, playlist_index)
        #             break

        #         for track in items:
        #             song_id = track['id']
        #             if check_song_exists(song_id):
        #                 print(f"Bỏ qua bài hát đã tồn tại: {song_id}")
        #                 continue

        #             for artist in track['artists']:
        #                 artist_id = artist['id']
        #                 artist_name = artist['name']
        #                 total_songs = scrape_artist_songs(artist_id, artist_name, market, total_songs, max_songs)
        #                 if total_songs >= max_songs:
        #                     print(f"Đã cào đủ {max_songs} bài hát.")
        #                     save_state(query, market, offset, playlist_id, query_index, playlist_index)
        #                     return

        #             track_artists = json.dumps([artist['id'] for artist in track['artists']])
        #             song = {
        #                 'id': song_id,
        #                 'song_name': track['name'],
        #                 'artists': track_artists,
        #                 'external_urls': track['external_urls']['spotify'],
        #                 'image_url': track['album']['images'][0]['url'] if track['album']['images'] else '',
        #                 'duration_ms': track['duration_ms']
        #             }

        #             save_song(song)
        #             total_songs += 1
        #             print(f"Đã lưu bài hát: {song['song_name']} (ID: {song_id}, Total: {total_songs}/{max_songs})")

        #         offset += LIMIT
        #         save_state(query, market, offset, playlist_id, query_index, playlist_index)

        #         if total_songs >= max_songs:
        #             print(f"Đã cào đủ {max_songs} bài hát.")
        #             break

        #         if offset >= 950:
        #             offset = 0
        #             playlist_index += 1
        #             playlist_id = None
        #             save_state(query, market, offset, playlist_id, query_index, playlist_index)
        #             break

        #         time.sleep(REQUEST_DELAY)

        while total_songs < max_songs:
            items = search(query, market, limit=LIMIT, offset=offset)
            if not items:
                print(f"Không còn bài hát cho query '{query}'")
                offset = 0
                query_index +=1
                if (query_index >= len(QUERIES)):
                    print(f'Đã hết query.')
                    break
                query = QUERIES[query_index]
                save_state(query, market, offset, None, query_index, playlist_index)
                continue

            for track in items:
                song_id = track['id']
                if check_song_exists(song_id):
                    print(f"Bỏ qua bài hát đã tồn tại: {song_id}")
                    continue

                for artist in track['artists']:
                    artist_id = artist['id']
                    artist_name = artist['name']
                    total_songs = scrape_artist_songs(artist_id, artist_name, market, total_songs, max_songs)
                    if total_songs >= max_songs:
                        print(f"Đã cào đủ {max_songs} bài hát.")
                        save_state(query, market, offset, None, query_index, playlist_index)
                        return

                track_artists = json.dumps([artist['id'] for artist in track['artists']])
                song = {
                    'id': song_id,
                    'song_name': track['name'],
                    'artists': track_artists,
                    'external_urls': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else '',
                    'duration_ms': track['duration_ms']
                }

                save_song(song)
                total_songs += 1
                print(f"Đã lưu bài hát: {song['song_name']} (ID: {song_id}, Total: {total_songs}/{max_songs})")

            offset += LIMIT
            save_state(query, market, offset, None, query_index, playlist_index)

            if total_songs >= max_songs:
                print(f"Đã cào đủ {max_songs} bài hát.")
                break

            if offset >= 300:
                offset = 0
                query_index +=1
                if (query_index >= len(QUERIES)):
                    print(f'Đã hết query.')
                    break
                query = QUERIES[query_index]
                save_state(query, market, offset, None, query_index, playlist_index)

            time.sleep(REQUEST_DELAY)
    except KeyboardInterrupt as e:
        print(f"Tạm dừng: {e}. Trạng thái đã được lưu.")
        save_state(query, market, offset, playlist_id, query_index, playlist_index)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get('Retry-After', 10))
            print(f"Lỗi 429: Quá nhiều yêu cầu. Chờ {retry_after} giây trước khi dừng. Trạng thái đã được lưu.")
            time.sleep(retry_after)
        else:
            print(f"Lỗi HTTP: {e}. Trạng thái đã được lưu.")
        save_state(query, market, offset, playlist_id, query_index, playlist_index)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi không xác định: {e}. Trạng thái đã được lưu.")
        save_state(query, market, offset, playlist_id, query_index, playlist_index)
        sys.exit(1)

if __name__ == "__main__":
    scrape_songs(max_songs=20000)