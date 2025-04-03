import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from collections import Counter

# 1. Tải file âm thanh
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# 2. Tính spectrogram
def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spectrogram

# 3. Tìm các đỉnh
def find_peaks(spectrogram, threshold=-20):
    peaks = []
    for t in range(spectrogram.shape[1]):
        for f in range(spectrogram.shape[0]):
            if spectrogram[f, t] > threshold:
                peaks.append((t, f, spectrogram[f, t]))
    return peaks

# 4. Tạo hàm băm
def generate_hashes(peaks, max_distance=10):
    hashes = {}
    for i in range(len(peaks)):
        for j in range(i + 1, min(i + max_distance, len(peaks))):
            t1, f1, _ = peaks[i]
            t2, f2, _ = peaks[j]
            delta_t = t2 - t1
            hash_value = (f1 * 1000000) + (f2 * 1000) + delta_t
            hashes[hash_value] = t1  # Lưu thời gian để so khớp sau
    return hashes

# 5. Tạo cơ sở dữ liệu từ thư mục
def build_database(folder_path):
    database = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            y, sr = load_audio(file_path)
            if y is not None:
                spectrogram = compute_spectrogram(y, sr)
                peaks = find_peaks(spectrogram)
                hashes = generate_hashes(peaks)
                database[file_name] = hashes
                print(f"Added {file_name} with {len(hashes)} hashes")
    return database

# 6. Tìm bài hát khớp
def find_match(target_file, database):
    # Xử lý file mục tiêu
    y, sr = load_audio(target_file)
    if y is None:
        return None
    spectrogram = compute_spectrogram(y, sr)
    peaks = find_peaks(spectrogram)
    target_hashes = generate_hashes(peaks)
    print(f"Target file has {len(target_hashes)} hashes")

    # So khớp với cơ sở dữ liệu
    match_counts = Counter()
    for song_name, song_hashes in database.items():
        common_hashes = set(target_hashes.keys()) & set(song_hashes.keys())
        match_counts[song_name] = len(common_hashes)
        print(f"{song_name}: {len(common_hashes)} matching hashes")

    # Tìm bài có số lượng hash khớp nhiều nhất
    if match_counts:
        best_match = match_counts.most_common(1)[0]
        if best_match[1] > 0:  # Có ít nhất 1 hash khớp
            return best_match[0], best_match[1]  # Tên bài, số hash khớp
    return None, 0

# 7. Chạy chương trình
if __name__ == "__main__":
    # Thay bằng đường dẫn thư mục chứa các file MP3 và file cần tìm
    folder_path = "./mp3"  # Ví dụ: "C:/music"
    target_file = "./Đại Lộ Mặt Trời - Chillies.mp3"  # File bạn muốn tìm

    # Tạo cơ sở dữ liệu từ thư mục
    print("Building database...")
    database = build_database(folder_path)

    if not database:
        print("No songs found in the folder.")
    else:
        # Tìm bài khớp
        print("\nSearching for match...")
        best_match, match_count = find_match(target_file, database)
        
        if best_match:
            print(f"\nBest match: {best_match} with {match_count} matching hashes")
        else:
            print("No match found in the database.")