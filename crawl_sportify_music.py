import os 
import base64
import requests
import json

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + auth_base64,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {"grant_type": "client_credentials"}
    result = requests.post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]

    return token

def search(query='vietnam', type='track', market='VN', limit='10', offset='0'):
    url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': 'Bearer ' + get_token(),
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json, charset=UTF-8'
    }

    subdirectory = f"?q={query}&type={type}&market={market}&limit={limit}&offset={offset}"
    query_url = url + subdirectory
    reuslt = requests.get(query_url, headers=headers)
    json_result = json.loads(reuslt.content)

    tracks = json_result['tracks']
    items = tracks['items']

    return items

tracks = search()

filtered_data = [
    {
        'artists': track['album']['artists'],
        'external_urls': track['album']['external_urls'],
        'images': track['album']['images'],
        'name': track['album']['name'],
        'duration_ms': track['duration_ms']
    }
    for track in tracks
]

print(json.dumps(filtered_data, indent=4, ensure_ascii=False))