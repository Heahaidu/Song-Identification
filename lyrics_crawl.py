import os
import json
import time
import requests
import sqlite3
import re
import pyotp
import binascii
import base64


SP_DC = os.getenv("SP_DC")
DB_FILE = "songs.db"
LYRICS_TOKEN_FILE = "spotify_token.json"

TOKEN_URL = 'https://open.spotify.com/get_access_token'
LYRICS_URL = 'https://spclient.wg.spotify.com/color-lyrics/v2/track/'
SERVER_TIME_URL = 'https://open.spotify.com/server-time'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS lyrics (
            track_id TEXT PRIMARY KEY,
            lyrics TEXT,
            FOREIGN KEY(track_id) REFERENCES tracks(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS no_lyrics (
            track_id TEXT PRIMARY KEY,
            FOREIGN KEY(track_id) REFERENCES tracks(id)
        )
    ''')
    conn.commit()
    conn.close()

def clean_hex(hex_str):
    valid_chars = r'[0-9a-fA-F]'
    cleaned = ''.join(re.findall(valid_chars, hex_str))
    if len(cleaned) % 2 != 0:
        cleaned = cleaned[:-1]
    return cleaned

def generate_totp(server_time_seconds):
    secret_cipher = [12, 56, 76, 33, 88, 44, 88, 33, 78, 78, 11, 66, 22, 22, 55, 69, 54]
    processed = []
    for i, byte in enumerate(secret_cipher):
        processed.append(byte ^ (i % 33 + 9))
    processed_str = ''.join(map(str, processed))
    utf8_bytes = processed_str.encode('utf-8')
    hex_str = binascii.hexlify(utf8_bytes).decode('ascii')
    cleaned_hex = clean_hex(hex_str)
    try:
        secret_bytes = binascii.unhexlify(cleaned_hex)
    except binascii.Error as e:
        raise Exception(f"Invalid hex string: {e}")
    secret_base32 = base64.b32encode(secret_bytes).decode('ascii').replace('=', '')
    totp = pyotp.TOTP(secret_base32, interval=30, digest='sha1', digits=6)
    return totp.at(server_time_seconds)

def get_server_time_params():
    response = requests.get(SERVER_TIME_URL)
    if response.status_code != 200:
        raise Exception('Failed to get server time')
    server_time_data = response.json()
    if not server_time_data or 'serverTime' not in server_time_data:
        raise Exception('Invalid server time response')
    server_time_seconds = server_time_data['serverTime']
    totp = generate_totp(server_time_seconds)
    timestamp = int(time.time())
    params = {
        'reason': 'transport',
        'productType': 'web-player',
        'totp': totp,
        'totpVer': '5',
        'ts': str(timestamp)
    }
    return params

def check_token_expire():
    if os.path.exists(LYRICS_TOKEN_FILE):
        try:
            with open(LYRICS_TOKEN_FILE, 'r') as f:
                json_data = json.load(f)
            time_left = json_data.get('accessTokenExpirationTimestampMs')
            time_now = int(time.time() * 1000)
            if time_left < time_now:
                new_token = get_token()
                with open(LYRICS_TOKEN_FILE, 'w', encoding='utf-8') as f:
                    json.dump(new_token, f, ensure_ascii=False, indent=4)
            return
        except (json.JSONDecodeError, IOError):
            pass
    new_token = get_token()
    with open(LYRICS_TOKEN_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_token, f, ensure_ascii=False, indent=4)

def get_token():
    if SP_DC is None:
        raise Exception('SP_DC is not set')
    try:
        params = get_server_time_params()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Cookie": f"sp_dc={SP_DC}"
        }
        response = requests.get(TOKEN_URL, params=params, headers=headers, timeout=600)
        if not response.ok:
            raise Exception(f"Token request failed: {response.text}")
        token_json = response.json()
        if not token_json or token_json.get("isAnonymous") is True:
            raise Exception("The SP_DC set seems to be invalid, please correct it!")
        return token_json
    except requests.RequestException as e:
        raise Exception(f"Token request failed: {str(e)}")
    except IOError as e:
        raise Exception(f"Unable to open file: {str(e)}")

def get_lyrics(track_id):
    check_token_expire()
    with open(LYRICS_TOKEN_FILE, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    token = json_data.get('accessToken')
    formatted_url = f"{LYRICS_URL}{track_id}?format=json&market=from_token"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36",
        "App-platform": "WebPlayer",
        'Accept': 'application/json, charset=UTF-8',
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(formatted_url, headers=headers, timeout=10)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"No lyrics found for track {track_id}")
            return None
        print(f"HTTP error while fetching lyrics for track {track_id}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while fetching lyrics for track {track_id}: {e}")
        return None

def filter_lyrics(lyrics_data):
    if lyrics_data is None or not lyrics_data or 'lyrics' not in lyrics_data or 'lines' not in lyrics_data['lyrics']:
        return None
    non_lyric_patterns = re.compile(r'^(♪)$', re.IGNORECASE)
    lines = [line.get('words', '').strip() for line in lyrics_data['lyrics']['lines']
             if line.get('words', '').strip() and not non_lyric_patterns.match(line.get('words', '').strip())]
    if not lines:
        return None
    return '. '.join(lines) + '.'

def save_lyrics(track_id, lyrics_text):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO lyrics (track_id, lyrics)
        VALUES (?, ?)
    ''', (track_id, lyrics_text))
    conn.commit()
    conn.close()
    print(f"Saved lyrics for track {track_id}")

def save_no_lyrics(track_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO no_lyrics (track_id)
        VALUES (?)
    ''', (track_id,))
    conn.commit()
    conn.close()
    print(f"Marked track {track_id} as having no lyrics")

def crawl_lyrics():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT id 
        FROM tracks 
        WHERE id NOT IN (SELECT track_id FROM lyrics) 
        AND id NOT IN (SELECT track_id FROM no_lyrics)
    ''')
    tracks = c.fetchall()
    conn.close()

    total_tracks = len(tracks)
    processed = 0

    print(f"Starting to crawl lyrics for {total_tracks} tracks without lyrics.")
    for (track_id,) in tracks:
        try:
            lyrics_text = filter_lyrics(get_lyrics(track_id))
            if lyrics_text:
                save_lyrics(track_id, lyrics_text)
                processed += 1
            else:
                save_no_lyrics(track_id)
                processed += 1
            print(f"Processed: {track_id} ({processed}/{total_tracks})")
        except Exception as e:
            print(f"Error crawling lyrics for track {track_id}: {e}")
            break
        except KeyboardInterrupt as e:
            print(f"The process has stopped ({processed}/{total_tracks}).")
        time.sleep(2.0)  

    print(f"Completed crawling lyrics. Processed {processed}/{total_tracks} tracks.")

if __name__ == "__main__":
    crawl_lyrics()