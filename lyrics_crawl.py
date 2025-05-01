import os
import json
import time
import requests
import binascii
import base64
import pyotp
import re

token_url = 'https://open.spotify.com/get_access_token'
lyrics_url = 'https://spclient.wg.spotify.com/color-lyrics/v2/track/'
server_time_url = 'https://open.spotify.com/server-time'
token_file = 'spotify_token.json'

SP_DC = os.getenv('SP_DC')

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

    totp = pyotp.TOTP(
            secret_base32,
            interval=30,
            digest='sha1',
            digits=6
        )
    
    return totp.at(server_time_seconds)
    
def get_server_time_params():
    response = requests.get(server_time_url)

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
    file_exists = os.path.exists(token_file)
    
    if file_exists:
        with open(token_file, 'r') as f:
            json_data = json.load(f)
        
        time_left = json_data.get('accessTokenExpirationTimestampMs')
        time_now = int(time.time() * 1000)

        if (time_left < time_now):
            with open(token_file, 'w', encoding='utf-8') as f:
                json.dump(get_token(), f, ensure_ascii=False, indent=4)
        return
    if (not file_exists):
        with open(token_file, 'w', encoding='utf-8') as f:
            json.dump(get_token(), f, ensure_ascii=False, indent=4)

def get_token():

    if SP_DC is None:
        raise Exception('SP_DC is not set')
    try:
        params = get_server_time_params()

        headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Cookie": f"sp_dc={SP_DC}"
            }
        
        response = requests.get(token_url, params=params, headers=headers, timeout=600)

        if not response.ok:
                raise Exception(f"Token request failed: {response.text}")
    
        token_json = response.json()

        if not token_json or token_json.get("isAnonymous") is True:
                raise Exception("The SP_DC set seems to be invalid, please correct it!")
        
        return json.loads(response.content)

    except requests.RequestException as e:
        raise Exception(f"Token request failed: {str(e)}")
    except IOError as e:
            raise Exception(f"Unable to open file: {str(e)}")
    
def get_lyrics(track_id):
    with open(token_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    token = json_data.get('accessToken')

    formatted_url = f"{lyrics_url}{track_id}?format=json&market=from_token"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36",
        "App-platform": "WebPlayer",
        'Accept': 'application/json, charset=UTF-8',
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(formatted_url, headers=headers, timeout=10)

    return response.content

check_token_expire()

lyrics = get_lyrics("3wn8HJNjkY4wzTBy35ZvQ6")

print(json.dumps(json.loads(lyrics), indent=4, ensure_ascii=False))