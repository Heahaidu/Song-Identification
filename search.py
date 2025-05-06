import sqlite3
import json
from fuzzywuzzy import fuzz

# Kết nối với SQLite
DB_FILE = "songs.db"

def init_db():
    """Khởi tạo kết nối với cơ sở dữ liệu SQLite."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Kiểm tra bảng songs
    c.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            id TEXT PRIMARY KEY,
            song_name TEXT NOT NULL,
            artists TEXT,
            external_urls TEXT,
            image_url TEXT,
            duration_ms INTEGER
        )
    ''')
    conn.commit()
    return conn

def format_duration(ms):
    """Chuyển đổi thời lượng từ ms sang phút:giây."""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

def check_song(query, use_fuzzy=False, limit=5):
    """Kiểm tra xem bài hát có tồn tại trong SQLite."""
    conn = init_db()
    c = conn.cursor()

    if not query:
        conn.close()
        return []

    if use_fuzzy:
        # Tải tất cả bài hát để so sánh fuzzy
        c.execute("SELECT id, song_name, artists, external_urls, image_url, duration_ms FROM songs")
        songs = c.fetchall()
        results = []
        for song in songs:
            song_name_score = fuzz.partial_ratio(query.lower(), song[1].lower())
            artists = json.loads(song[2]) if song[2] else []
            artist_score = max([fuzz.partial_ratio(query.lower(), artist.lower()) for artist in artists], default=0)
            score = max(song_name_score, artist_score)
            if score > 80:  # Ngưỡng fuzzy matching
                results.append({
                    'id': song[0],
                    'song_name': song[1],
                    'artists': artists,
                    'external_urls': song[3],
                    'image_url': song[4],
                    'duration_ms': song[5],
                    'match_score': score
                })
        results = sorted(results, key=lambda x: x['match_score'], reverse=True)[:limit]
    else:
        # Tìm kiếm LIKE
        query_like = f"%{query}%"
        c.execute('''
            SELECT id, song_name, artists, external_urls, image_url, duration_ms
            FROM songs
            WHERE song_name LIKE ? OR artists LIKE ?
            LIMIT ?
        ''', (query_like, query_like, limit))
        results = [
            {
                'id': row[0],
                'song_name': row[1],
                'artists': json.loads(row[2]) if row[2] else [],
                'external_urls': row[3],
                'image_url': row[4],
                'duration_ms': row[5]
            }
            for row in c.fetchall()
        ]

    conn.close()
    return results

def main():
    """Chương trình chính để kiểm tra bài hát."""
    print("Kiểm tra bài hát trong cơ sở dữ liệu SQLite")
    query = input("Nhập tên bài hát hoặc nghệ sĩ: ").strip()
    use_fuzzy = input("Sử dụng tìm kiếm gần đúng (xử lý lỗi chính tả)? (y/n): ").lower() == 'y'

    results = check_song(query, use_fuzzy=use_fuzzy, limit=5)

    if results:
        print("\nKết quả tìm kiếm:")
        for i, result in enumerate(results, 1):
            print(f"\nKết quả {i}:")
            print(f"Tên bài: {result['song_name']}")
            print(f"Nghệ sĩ: {', '.join(result['artists']) if result['artists'] else 'Không rõ'}")
            print(f"Thời lượng: {format_duration(result['duration_ms'])}")
            print(f"Link Spotify: {result['external_urls']}")
            if result['image_url']:
                print(f"Hình ảnh: {result['image_url']}")
            if use_fuzzy:
                print(f"Độ khớp: {result['match_score']}%")
            print("-" * 50)
    else:
        print("\nKhông tìm thấy bài hát nào khớp với từ khóa.")

if __name__ == "__main__":
    main()