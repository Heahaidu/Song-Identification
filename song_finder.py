# import sqlite3
# import re
# import numpy as np
# from rank_bm25 import BM25Okapi
# from flask import Flask, request, jsonify
# import json
# import os
# import pickle
# from typing import List, Dict
# from underthesea import word_tokenize
# from langdetect import detect
# from nltk.corpus import stopwords
# import nltk
# from datetime import datetime
# from rapidfuzz import fuzz

# app = Flask(__name__)

# # Configuration
# DB_PATH = "songs.db"
# INDEX_FILE = "bm25_index.pkl"
# STOPWORDS = {}
# query_cache = {}  # Cache for search queries
# documents = []  # List of tokenized documents
# track_metadata = {}  # Metadata for tracks
# bm25_index = None  # BM25 index
# feedback_weights = {}  # Feedback weights {track_id: {'correct': count, 'wrong': count}}

# # Download NLTK resources
# try:
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt_tab', quiet=True)
#     for lang in ['english', 'vietnamese', 'spanish', 'french', 'german']:
#         if lang in stopwords.fileids():
#             STOPWORDS[lang] = set(stopwords.words(lang))
# except Exception as e:
#     print(f"Warning: NLTK resource download failed: {str(e)}. Stopwords filtering may be disabled.")

# def clean_text(text: str) -> str:
#     """Clean text by removing special characters and converting to lowercase"""
#     text = re.sub(r'[^\w\s]', ' ', text.lower())
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def tokenize(text: str, lang: str = None) -> List[str]:
#     """Tokenize text into words based on language"""
#     if not text:
#         print("Tokenize: Empty input text")
#         return []
#     cleaned_text = clean_text(text)
#     if not cleaned_text:
#         print("Tokenize: Empty cleaned text")
#         return []
#     if not lang:
#         try:
#             lang = detect(cleaned_text)
#         except:
#             lang = 'en'
    
#     # Tokenize based on language
#     try:
#         if lang.startswith('vi'):
#             tokens = word_tokenize(cleaned_text)
#         else:
#             tokens = nltk.word_tokenize(cleaned_text)
#     except Exception as e:
#         print(f"Tokenize: NLTK failed for lang {lang}, text: {cleaned_text[:50]}... Error: {str(e)}")
#         tokens = cleaned_text.split()
    
#     # Remove stopwords
#     if lang in STOPWORDS:
#         tokens = [t for t in tokens if t not in STOPWORDS[lang]]
    
#     print(f"Tokenize: Lang {lang}, Text: {cleaned_text[:50]}... -> Tokens: {tokens[:10]}")
#     return tokens

# def save_index():
#     """Save BM25 index and metadata to file"""
#     try:
#         with open(INDEX_FILE, 'wb') as f:
#             pickle.dump({
#                 'documents': documents,
#                 'track_metadata': track_metadata,
#                 'feedback_weights': feedback_weights
#             }, f)
#         print(f"Saved BM25 index to {INDEX_FILE}")
#     except Exception as e:
#         print(f"Error saving index: {str(e)}")

# def load_index():
#     """Load BM25 index and metadata from file"""
#     global documents, track_metadata, feedback_weights, bm25_index
#     try:
#         if os.path.exists(INDEX_FILE):
#             with open(INDEX_FILE, 'rb') as f:
#                 data = pickle.load(f)
#             documents = data['documents']
#             track_metadata = data['track_metadata']
#             feedback_weights = data['feedback_weights']
#             if documents:
#                 bm25_index = BM25Okapi(documents)
#                 print(f"Loaded BM25 index from {INDEX_FILE}, {len(documents)} documents")
#                 return True
#         return False
#     except Exception as e:
#         print(f"Error loading index: {str(e)}")
#         return False

# def check_db_changed() -> bool:
#     """Check if database has changed since last index save"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM tracks")
#         track_count = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM lyrics")
#         lyrics_count = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM artists")
#         artists_count = cursor.fetchone()[0]
#         conn.close()
        
#         if os.path.exists(INDEX_FILE):
#             with open(INDEX_FILE, 'rb') as f:
#                 data = pickle.load(f)
#             # Assume index is invalid if track count doesn't match
#             return len(data['track_metadata']) != track_count
#         return True
#     except Exception as e:
#         print(f"Error checking DB: {str(e)}")
#         return True

# def init_bm25_index():
#     """Initialize BM25 index from SQLite database or load from file"""
#     global bm25_index, track_metadata, documents, feedback_weights
    
#     # Try loading from file
#     if not check_db_changed() and load_index():
#         return
    
#     print("Creating BM25 index...")
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Create feedback table with is_correct column
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS feedback (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             query TEXT,
#             track_id INTEGER,
#             is_correct BOOLEAN,
#             timestamp DATETIME
#         )
#     """)
    
#     # Query tracks and lyrics
#     cursor.execute("""
#         SELECT t.id, t.song_name, t.artists, t.external_urls, t.image_url, l.lyrics 
#         FROM tracks t 
#         LEFT JOIN lyrics l ON t.id = l.track_id
#     """)
    
#     documents.clear()
#     track_metadata.clear()
#     valid_documents = 0
#     print("Processing tracks and preparing documents...")
#     for idx, row in enumerate(cursor.fetchall()):
#         track_id, song_name, artists_json, external_urls, image_url, lyrics = row
#         # Parse artists JSON array
#         try:
#             artist_ids = json.loads(artists_json) if artists_json else []
#         except json.JSONDecodeError:
#             print(f"Warning: Invalid JSON in artists for track {song_name} (ID: {track_id}): {artists_json}")
#             artist_ids = []
        
#         # Get artist names
#         artist_names = []
#         primary_artist = None
#         if artist_ids:
#             format_strings = ','.join(['?'] * len(artist_ids))
#             cursor.execute(f"SELECT id, name FROM artists WHERE id IN ({format_strings})", artist_ids)
#             artist_dict = {str(row[0]): row[1] for row in cursor.fetchall()}
#             artist_names = [artist_dict.get(str(aid)) for aid in artist_ids if str(aid) in artist_dict]
#             primary_artist = artist_names[0] if artist_names else None
#             print(f"Track {idx + 1}: {song_name}, Artists: {artist_names}, Primary: {primary_artist}")
#         else:
#             print(f"Track {idx + 1}: {song_name}, No artists found")
        
#         # Create document: lyrics + song_name (x4) + primary_artist (x3) + other_artists (x1)
#         doc_parts = []
#         if lyrics:
#             doc_parts.append(lyrics)
#         doc_parts.append((song_name + " ") * 4)
#         if primary_artist:
#             doc_parts.append((primary_artist + " ") * 3)
#         if len(artist_names) > 1:
#             doc_parts.append(" ".join(artist_names[1:]))
#         doc = " ".join(doc_parts)
#         doc = clean_text(doc)
#         tokenized_doc = tokenize(doc)
#         if tokenized_doc:
#             documents.append(tokenized_doc)
#             track_metadata[str(valid_documents)] = {
#                 "id": track_id,
#                 "song_name": song_name,
#                 "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
#                 "external_urls": external_urls,
#                 "image_url": image_url,
#                 "has_lyrics": bool(lyrics)
#             }
#             valid_documents += 1
#         else:
#             print(f"Warning: Empty document for track {song_name} (ID: {track_id})")
    
#     print(f"Total tracks processed: {idx + 1}, Valid documents: {valid_documents}")
    
#     if documents:
#         print(f"Creating BM25 index for {len(documents)} documents...")
#         bm25_index = BM25Okapi(documents)
        
#         # Load feedback weights
#         cursor.execute("SELECT track_id, is_correct, COUNT(*) as count FROM feedback GROUP BY track_id, is_correct")
#         feedback_weights = {}
#         for row in cursor.fetchall():
#             track_id, is_correct, count = row
#             if str(track_id) not in feedback_weights:
#                 feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
#             if is_correct:
#                 feedback_weights[str(track_id)]['correct'] = count
#             else:
#                 feedback_weights[str(track_id)]['wrong'] = count
        
#         print("BM25 index creation completed!")
#         save_index()
#     else:
#         print("Error: No valid documents found. BM25 index not created.")
#         bm25_index = None
    
#     conn.close()

# def add_new_song(song_name: str, artist_names: List[str], external_urls: str, image_url: str, lyrics: str) -> bool:
#     """Add a new song to SQLite and BM25 index"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
        
#         # Get or insert artists
#         artist_ids = []
#         for name in artist_names:
#             cursor.execute("SELECT id FROM artists WHERE name = ?", (name,))
#             artist_row = cursor.fetchone()
#             if artist_row:
#                 artist_id = artist_row[0]
#             else:
#                 cursor.execute("INSERT INTO artists (name) VALUES (?)", (name,))
#                 artist_id = cursor.lastrowid
#             artist_ids.append(artist_id)
        
#         # Insert into tracks with artists as JSON array
#         artists_json = json.dumps(artist_ids)
#         cursor.execute("INSERT INTO tracks (song_name, artists, external_urls, image_url) VALUES (?, ?, ?, ?)", 
#                       (song_name, artists_json, external_urls, image_url))
#         track_id = cursor.lastrowid
        
#         # Insert into lyrics if provided
#         if lyrics:
#             cursor.execute("INSERT INTO lyrics (track_id, lyrics) VALUES (?, ?)", (track_id, lyrics))
        
#         # Update BM25 index
#         doc_parts = []
#         if lyrics:
#             doc_parts.append(lyrics)
#         doc_parts.append((song_name + " ") * 4)
#         if artist_names:
#             doc_parts.append((artist_names[0] + " ") * 3)
#         if len(artist_names) > 1:
#             doc_parts.append(" ".join(artist_names[1:]))
#         doc = " ".join(doc_parts)
#         doc = clean_text(doc)
#         tokenized_doc = tokenize(doc)
#         if tokenized_doc:
#             documents.append(tokenized_doc)
#             idx = len(track_metadata)
#             track_metadata[str(idx)] = {
#                 "id": track_id,
#                 "song_name": song_name,
#                 "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
#                 "external_urls": external_urls,
#                 "image_url": image_url,
#                 "has_lyrics": bool(lyrics)
#             }
            
#             # Rebuild BM25 index
#             global bm25_index
#             bm25_index = BM25Okapi(documents)
#             save_index()
#         else:
#             print(f"Warning: Empty document for new song {song_name}")
        
#         conn.commit()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"Error adding song: {str(e)}")
#         return False

# def save_feedback(query: str, track_id: int, is_correct: bool):
#     """Save user feedback to SQLite"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("INSERT INTO feedback (query, track_id, is_correct, timestamp) VALUES (?, ?, ?, ?)",
#                       (query, track_id, is_correct, datetime.now()))
#         conn.commit()
#         # Update feedback weights
#         cursor.execute("SELECT is_correct, COUNT(*) FROM feedback WHERE track_id = ? GROUP BY is_correct", (track_id,))
#         feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
#         for row in cursor.fetchall():
#             is_correct, count = row
#             if is_correct:
#                 feedback_weights[str(track_id)]['correct'] = count
#             else:
#                 feedback_weights[str(track_id)]['wrong'] = count
#         conn.close()
#         save_index()
#     except Exception as e:
#         print(f"Error saving feedback: {str(e)}")

# def search_songs(query: str, k: int = 10) -> List[Dict]:
#     """Search songs by song_name, artist_name, or lyrics using BM25"""
#     print(f"Searching for query: {query[:50]}...")
#     query_clean = clean_text(query)
#     if not query_clean:
#         print("No valid query.")
#         return []
    
#     # Check cache
#     if query_clean in query_cache:
#         print(f"Returning cached results for query: {query_clean}")
#         return query_cache[query_clean]
    
#     if bm25_index is None:
#         print("Error: BM25 index not initialized.")
#         return []
    
#     # Fuzzy string matching
#     string_scores = {}
#     query_tokens = tokenize(query_clean)
#     for idx, metadata in track_metadata.items():
#         song_name_clean = clean_text(metadata["song_name"])
#         artist_name_clean = clean_text(metadata["artist_name"] or "")
#         # Full string similarity
#         score_song = fuzz.ratio(query_clean, song_name_clean)
#         if score_song > 95:
#             string_scores[int(idx)] = float(score_song + 30)
#         elif score_song > 70:
#             string_scores[int(idx)] = float(score_song)
#         if artist_name_clean:
#             score_artist = fuzz.ratio(query_clean, artist_name_clean)
#             if score_artist > 95:
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist + 30))
#             elif score_artist > 70:
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist))
#         # Partial match
#         for q in query_tokens:
#             if q in song_name_clean or (artist_name_clean and q in artist_name_clean):
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), 60.0)
    
#     # BM25 search
#     tokenized_query = tokenize(query_clean)
#     bm25_scores = bm25_index.get_scores(tokenized_query)
    
#     # Combine scores
#     results = []
#     for idx, bm25_score in enumerate(bm25_scores):
#         if str(idx) in track_metadata:
#             metadata = track_metadata[str(idx)]
#             string_score = string_scores.get(idx, 0.0)
#             # Feedback boost: +10 for correct, -5 for wrong
#             fb = feedback_weights.get(str(metadata["id"]), {'correct': 0, 'wrong': 0})
#             feedback_boost = float(fb['correct'] * 10.0 - fb['wrong'] * 5.0)
#             combined_score = float(max(bm25_score * 100, string_score) + feedback_boost)
#             if combined_score > 0:
#                 results.append({
#                     "song_name": metadata["song_name"],
#                     "artist_name": metadata["artist_name"],
#                     "external_urls": metadata["external_urls"],
#                     "image_url": metadata["image_url"],
#                     "match_score": round(combined_score, 2),
#                     "has_lyrics": metadata["has_lyrics"],
#                     "track_id": metadata["id"]
#                 })
    
#     # Sort and limit results
#     results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:k]
#     query_cache[query_clean] = results
#     result_strings = [f"{r['song_name']} by {r['artist_name'] or 'Unknown'}" for r in results]
#     print(f"Found {len(results)} results: {result_strings}")
#     return results

# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     try:
#         with open("index.html", "r") as f:
#             return f.read()
#     except FileNotFoundError:
#         return jsonify({"error": "index.html not found"}), 500

# @app.route('/search', methods=['POST'])
# def search():
#     """Handle search requests"""
#     try:
#         data = request.json
#         if not data:
#             return jsonify({"error": "Invalid JSON body"}), 400
        
#         query = data.get('query', data.get('lyrics', ''))
#         if not query:
#             return jsonify({"error": "No query or lyrics provided in JSON body"}), 400
        
#         results = search_songs(query)
#         return jsonify(results)
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Search error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/add_song', methods=['POST'])
# def add_song():
#     """Handle adding new songs"""
#     try:
#         data = request.json
#         song_name = data.get('song_name', '')
#         artist_names = data.get('artist_names', [])
#         external_urls = data.get('external_urls', '')
#         image_url = data.get('image_url', '')
#         lyrics = data.get('lyrics', '')
        
#         if not song_name or not artist_names:
#             return jsonify({"error": "Song name and at least one artist name are required"}), 400
        
#         success = add_new_song(song_name, artist_names, external_urls, image_url, lyrics)
#         if success:
#             return jsonify({"message": "Song added successfully"})
#         else:
#             return jsonify({"error": "Failed to add song"}), 500
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Add song error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     print('feedback')
#     """Handle user feedback"""
#     try:
#         data = request.json
#         query = data.get('query', '')
#         track_id = data.get('track_id', None)
#         is_correct = data.get('is_correct', None)
        
#         if not query or track_id is None or is_correct is None:
#             return jsonify({"error": "Query, track_id, and is_correct are required"}), 400
        
#         save_feedback(query, track_id, is_correct)
#         return jsonify({"message": "Feedback saved successfully"})
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Feedback error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == "__main__":
#     print("Initializing BM25 index...")
#     init_bm25_index()
#     print("Starting Flask server...")
#     app.run(debug=True, host='0.0.0.0', port=5001)
# ================================================================
# import sqlite3
# import re
# import numpy as np
# from flask import Flask, request, jsonify
# import json
# import os
# import pickle
# from typing import List, Dict, Tuple
# from underthesea import word_tokenize
# from langdetect import detect
# from nltk.corpus import stopwords
# import nltk
# from datetime import datetime
# from rapidfuzz import fuzz
# from math import log

# app = Flask(__name__)

# # Configuration
# DB_PATH = "songs.db"
# INDEX_FILE = "index.pkl"
# STOPWORDS = {}
# query_cache = {}  # Cache for search queries
# documents = []  # List of tokenized documents
# track_metadata = {}  # Metadata for tracks
# inverted_index = None  # Inverted Index
# feedback_weights = {}  # Feedback weights {track_id: {'correct': count, 'wrong': count}}

# # BM25 parameters
# k1 = 1.5
# b = 0.75

# # Download NLTK resources
# try:
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt_tab', quiet=True)
#     for lang in ['english', 'vietnamese', 'spanish', 'french', 'german']:
#         if lang in stopwords.fileids():
#             STOPWORDS[lang] = set(stopwords.words(lang))
# except Exception as e:
#     print(f"Warning: NLTK resource download failed: {str(e)}. Stopwords filtering may be disabled.")

# def clean_text(text: str) -> str:
#     """Clean text by removing special characters and converting to lowercase"""
#     text = re.sub(r'[^\w\s]', ' ', text.lower())
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def tokenize(text: str, lang: str = None) -> List[str]:
#     """Tokenize text into words based on language"""
#     if not text:
#         print("Tokenize: Empty input text")
#         return []
#     cleaned_text = clean_text(text)
#     if not cleaned_text:
#         print("Tokenize: Empty cleaned text")
#         return []
#     if not lang:
#         try:
#             lang = detect(cleaned_text)
#         except:
#             lang = 'en'
    
#     try:
#         if lang.startswith('vi'):
#             tokens = word_tokenize(cleaned_text)
#         else:
#             tokens = nltk.word_tokenize(cleaned_text)
#     except Exception as e:
#         print(f"Tokenize: NLTK failed for lang {lang}, text: {cleaned_text[:50]}... Error: {str(e)}")
#         tokens = cleaned_text.split()
    
#     if lang in STOPWORDS:
#         tokens = [t for t in tokens if t not in STOPWORDS[lang]]
    
#     print(f"Tokenize: Lang {lang}, Text: {cleaned_text[:50]}... -> Tokens: {tokens[:10]}")
#     return tokens

# class InvertedIndex:
#     """Custom Inverted Index for BM25 scoring"""
#     def __init__(self):
#         self.index = {}  # {term: [(doc_id, tf, field_boost), ...]}
#         self.doc_lengths = {}  # {doc_id: length}
#         self.avgdl = 0  # Average document length
#         self.N = 0  # Number of documents

#     def add_document(self, doc_id: int, tokens: List[str], field_boost: float = 1.0):
#         """Add a document to the index"""
#         term_freq = {}
#         for term in tokens:
#             term_freq[term] = term_freq.get(term, 0) + 1
        
#         for term, tf in term_freq.items():
#             if term not in self.index:
#                 self.index[term] = []
#             self.index[term].append((doc_id, tf, field_boost))
        
#         doc_length = len(tokens)
#         self.doc_lengths[doc_id] = doc_length
#         self.N += 1
#         self.avgdl = (self.avgdl * (self.N - 1) + doc_length) / self.N if self.N > 0 else 0

#     def get_scores(self, query_tokens: List[str]) -> List[float]:
#         """Calculate BM25 scores for all documents"""
#         scores = [0.0] * self.N
#         for term in query_tokens:
#             if term in self.index:
#                 n = len(self.index[term])
#                 idf = log((self.N - n + 0.5) / (n + 0.5) + 1)
#                 for doc_id, tf, field_boost in self.index[term]:
#                     dl = self.doc_lengths[doc_id]
#                     score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
#                     scores[doc_id] += score * field_boost
#         return scores

# def save_index():
#     """Save Inverted Index and metadata to file"""
#     try:
#         with open(INDEX_FILE, 'wb') as f:
#             pickle.dump({
#                 'documents': documents,
#                 'track_metadata': track_metadata,
#                 'feedback_weights': feedback_weights,
#                 'inverted_index': inverted_index
#             }, f)
#         print(f"Saved index to {INDEX_FILE}")
#     except Exception as e:
#         print(f"Error saving index: {str(e)}")

# def load_index():
#     """Load Inverted Index and metadata from file"""
#     global documents, track_metadata, feedback_weights, inverted_index
#     try:
#         if os.path.exists(INDEX_FILE):
#             with open(INDEX_FILE, 'rb') as f:
#                 data = pickle.load(f)
#             documents = data['documents']
#             track_metadata = data['track_metadata']
#             feedback_weights = data['feedback_weights']
#             inverted_index = data['inverted_index']
#             print(f"Loaded index from {INDEX_FILE}, {inverted_index.N} documents")
#             return True
#         return False
#     except Exception as e:
#         print(f"Error loading index: {str(e)}")
#         return False

# def check_db_changed() -> bool:
#     """Check if database has changed since last index save"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM tracks")
#         track_count = cursor.fetchone()[0]
#         conn.close()
        
#         if os.path.exists(INDEX_FILE):
#             with open(INDEX_FILE, 'rb') as f:
#                 data = pickle.load(f)
#             return len(data['track_metadata']) != track_count
#         return True
#     except Exception as e:
#         print(f"Error checking DB: {str(e)}")
#         return True

# def init_index():
#     """Initialize Inverted Index from SQLite database or load from file"""
#     global inverted_index, track_metadata, documents, feedback_weights
    
#     if not check_db_changed() and load_index():
#         return
    
#     print("Creating Inverted Index...")
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS feedback (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             query TEXT,
#             track_id INTEGER,
#             is_correct BOOLEAN,
#             timestamp DATETIME
#         )
#     """)
    
#     cursor.execute("""
#         SELECT t.id, t.song_name, t.artists, t.external_urls, t.image_url, l.lyrics 
#         FROM tracks t 
#         LEFT JOIN lyrics l ON t.id = l.track_id
#     """)
    
#     documents.clear()
#     track_metadata.clear()
#     inverted_index = InvertedIndex()
#     valid_documents = 0
#     print("Processing tracks and preparing documents...")
#     for idx, row in enumerate(cursor.fetchall()):
#         track_id, song_name, artists_json, external_urls, image_url, lyrics = row
#         try:
#             artist_ids = json.loads(artists_json) if artists_json else []
#         except json.JSONDecodeError:
#             print(f"Warning: Invalid JSON in artists for track {song_name} (ID: {track_id}): {artists_json}")
#             artist_ids = []
        
#         artist_names = []
#         primary_artist = None
#         if artist_ids:
#             format_strings = ','.join(['?'] * len(artist_ids))
#             cursor.execute(f"SELECT id, name FROM artists WHERE id IN ({format_strings})", artist_ids)
#             artist_dict = {str(row[0]): row[1] for row in cursor.fetchall()}
#             artist_names = [artist_dict.get(str(aid)) for aid in artist_ids if str(aid) in artist_dict]
#             primary_artist = artist_names[0] if artist_names else None
#             print(f"Track {idx + 1}: {song_name}, Artists: {artist_names}, Primary: {primary_artist}")
#         else:
#             print(f"Track {idx + 1}: {song_name}, No artists found")
        
#         # Create document and add to index
#         tokenized_doc = []
#         if lyrics:
#             tokens = tokenize(lyrics)
#             inverted_index.add_document(valid_documents, tokens, field_boost=1.0)
#             tokenized_doc.extend(tokens)
#         song_tokens = tokenize(song_name)
#         inverted_index.add_document(valid_documents, song_tokens, field_boost=4.0)
#         tokenized_doc.extend(song_tokens)
#         if primary_artist:
#             artist_tokens = tokenize(primary_artist)
#             inverted_index.add_document(valid_documents, artist_tokens, field_boost=3.0)
#             tokenized_doc.extend(artist_tokens)
#         if len(artist_names) > 1:
#             other_artists = " ".join(artist_names[1:])
#             other_tokens = tokenize(other_artists)
#             inverted_index.add_document(valid_documents, other_tokens, field_boost=1.0)
#             tokenized_doc.extend(other_tokens)
        
#         if tokenized_doc:
#             documents.append(tokenized_doc)
#             track_metadata[str(valid_documents)] = {
#                 "id": track_id,
#                 "song_name": song_name,
#                 "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
#                 "external_urls": external_urls,
#                 "image_url": image_url,
#                 "has_lyrics": bool(lyrics)
#             }
#             valid_documents += 1
#         else:
#             print(f"Warning: Empty document for track {song_name} (ID: {track_id})")
    
#     print(f"Total tracks processed: {idx + 1}, Valid documents: {valid_documents}")
    
#     if documents:
#         cursor.execute("SELECT track_id, is_correct, COUNT(*) as count FROM feedback GROUP BY track_id, is_correct")
#         feedback_weights = {}
#         for row in cursor.fetchall():
#             track_id, is_correct, count = row
#             if str(track_id) not in feedback_weights:
#                 feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
#             if is_correct:
#                 feedback_weights[str(track_id)]['correct'] = count
#             else:
#                 feedback_weights[str(track_id)]['wrong'] = count
        
#         print("Inverted Index creation completed!")
#         save_index()
#     else:
#         print("Error: No valid documents found. Inverted Index not created.")
#         inverted_index = None
    
#     conn.close()

# def add_new_song(song_name: str, artist_names: List[str], external_urls: str, image_url: str, lyrics: str) -> bool:
#     """Add a new song to SQLite and update Inverted Index"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
        
#         artist_ids = []
#         for name in artist_names:
#             cursor.execute("SELECT id FROM artists WHERE name = ?", (name,))
#             artist_row = cursor.fetchone()
#             if artist_row:
#                 artist_id = artist_row[0]
#             else:
#                 cursor.execute("INSERT INTO artists (name) VALUES (?)", (name,))
#                 artist_id = cursor.lastrowid
#             artist_ids.append(artist_id)
        
#         artists_json = json.dumps(artist_ids)
#         cursor.execute("INSERT INTO tracks (song_name, artists, external_urls, image_url) VALUES (?, ?, ?, ?)", 
#                       (song_name, artists_json, external_urls, image_url))
#         track_id = cursor.lastrowid
        
#         if lyrics:
#             cursor.execute("INSERT INTO lyrics (track_id, lyrics) VALUES (?, ?)", (track_id, lyrics))
        
#         # Update Inverted Index incrementally
#         tokenized_doc = []
#         doc_id = len(documents)
#         if lyrics:
#             tokens = tokenize(lyrics)
#             inverted_index.add_document(doc_id, tokens, field_boost=1.0)
#             tokenized_doc.extend(tokens)
#         song_tokens = tokenize(song_name)
#         inverted_index.add_document(doc_id, song_tokens, field_boost=4.0)
#         tokenized_doc.extend(song_tokens)
#         if artist_names:
#             artist_tokens = tokenize(artist_names[0])
#             inverted_index.add_document(doc_id, artist_tokens, field_boost=3.0)
#             tokenized_doc.extend(artist_tokens)
#         if len(artist_names) > 1:
#             other_artists = " ".join(artist_names[1:])
#             other_tokens = tokenize(other_artists)
#             inverted_index.add_document(doc_id, other_tokens, field_boost=1.0)
#             tokenized_doc.extend(other_tokens)
        
#         if tokenized_doc:
#             documents.append(tokenized_doc)
#             track_metadata[str(doc_id)] = {
#                 "id": track_id,
#                 "song_name": song_name,
#                 "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
#                 "external_urls": external_urls,
#                 "image_url": image_url,
#                 "has_lyrics": bool(lyrics)
#             }
#             save_index()
#         else:
#             print(f"Warning: Empty document for new song {song_name}")
        
#         conn.commit()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"Error adding song: {str(e)}")
#         return False

# def save_feedback(query: str, track_id: int, is_correct: bool):
#     """Save user feedback to SQLite"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("INSERT INTO feedback (query, track_id, is_correct, timestamp) VALUES (?, ?, ?, ?)",
#                       (query, track_id, is_correct, datetime.now()))
#         conn.commit()
#         cursor.execute("SELECT is_correct, COUNT(*) FROM feedback WHERE track_id = ? GROUP BY is_correct", (track_id,))
#         feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
#         for row in cursor.fetchall():
#             is_correct, count = row
#             if is_correct:
#                 feedback_weights[str(track_id)]['correct'] = count
#             else:
#                 feedback_weights[str(track_id)]['wrong'] = count
#         conn.close()
#         save_index()
#     except Exception as e:
#         print(f"Error saving feedback: {str(e)}")

# def search_songs(query: str, k: int = 10) -> List[Dict]:
#     """Search songs using Inverted Index with BM25 scoring"""
#     print(f"Searching for query: {query[:50]}...")
#     query_clean = clean_text(query)
#     if not query_clean:
#         print("No valid query.")
#         return []
    
#     if query_clean in query_cache:
#         print(f"Returning cached results for query: {query_clean}")
#         return query_cache[query_clean]
    
#     if inverted_index is None:
#         print("Error: Inverted Index not initialized.")
#         return []
    
#     # Fuzzy string matching
#     string_scores = {}
#     query_tokens = tokenize(query_clean)
#     for idx, metadata in track_metadata.items():
#         song_name_clean = clean_text(metadata["song_name"])
#         artist_name_clean = clean_text(metadata["artist_name"] or "")
#         score_song = fuzz.ratio(query_clean, song_name_clean)
#         if score_song > 95:
#             string_scores[int(idx)] = float(score_song + 30)
#         elif score_song > 70:
#             string_scores[int(idx)] = float(score_song)
#         if artist_name_clean:
#             score_artist = fuzz.ratio(query_clean, artist_name_clean)
#             if score_artist > 95:
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist + 30))
#             elif score_artist > 70:
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist))
#         for q in query_tokens:
#             if q in song_name_clean or (artist_name_clean and q in artist_name_clean):
#                 string_scores[int(idx)] = max(string_scores.get(int(idx), 0), 60.0)
    
#     # BM25 scoring with Inverted Index
#     bm25_scores = inverted_index.get_scores(query_tokens)
    
#     # Combine scores
#     results = []
#     for idx, bm25_score in enumerate(bm25_scores):
#         if str(idx) in track_metadata:
#             metadata = track_metadata[str(idx)]
#             string_score = string_scores.get(idx, 0.0)
#             fb = feedback_weights.get(str(metadata["id"]), {'correct': 0, 'wrong': 0})
#             feedback_boost = float(fb['correct'] * 10.0 - fb['wrong'] * 5.0)
#             combined_score = float(max(bm25_score * 100, string_score) + feedback_boost)
#             if combined_score > 0:
#                 results.append({
#                     "song_name": metadata["song_name"],
#                     "artist_name": metadata["artist_name"],
#                     "external_urls": metadata["external_urls"],
#                     "image_url": metadata["image_url"],
#                     "match_score": round(combined_score, 2),
#                     "has_lyrics": metadata["has_lyrics"],
#                     "track_id": metadata["id"]
#                 })
    
#     results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:k]
#     query_cache[query_clean] = results
#     result_strings = [f"{r['song_name']} by {r['artist_name'] or 'Unknown'}" for r in results]
#     print(f"Found {len(results)} results: {result_strings}")
#     return results

# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     try:
#         with open("index.html", "r") as f:
#             return f.read()
#     except FileNotFoundError:
#         return jsonify({"error": "index.html not found"}), 500

# @app.route('/search', methods=['POST'])
# def search():
#     """Handle search requests"""
#     try:
#         data = request.json
#         if not data:
#             return jsonify({"error": "Invalid JSON body"}), 400
        
#         query = data.get('query', data.get('lyrics', ''))
#         if not query:
#             return jsonify({"error": "No query or lyrics provided in JSON body"}), 400
        
#         results = search_songs(query)
#         return jsonify(results)
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Search error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/add_song', methods=['POST'])
# def add_song():
#     """Handle adding new songs"""
#     try:
#         data = request.json
#         song_name = data.get('song_name', '')
#         artist_names = data.get('artist_names', [])
#         external_urls = data.get('external_urls', '')
#         image_url = data.get('image_url', '')
#         lyrics = data.get('lyrics', '')
        
#         if not song_name or not artist_names:
#             return jsonify({"error": "Song name and at least one artist name are required"}), 400
        
#         success = add_new_song(song_name, artist_names, external_urls, image_url, lyrics)
#         if success:
#             return jsonify({"message": "Song added successfully"})
#         else:
#             return jsonify({"error": "Failed to add song"}), 500
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Add song error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     """Handle user feedback"""
#     try:
#         data = request.json
#         query = data.get('query', '')
#         track_id = data.get('track_id', None)
#         is_correct = data.get('is_correct', None)
        
#         if not query or track_id is None or is_correct is None:
#             return jsonify({"error": "Query, track_id, and is_correct are required"}), 400
        
#         save_feedback(query, track_id, is_correct)
#         return jsonify({"message": "Feedback saved successfully"})
#     except ValueError:
#         return jsonify({"error": "Invalid JSON format"}), 400
#     except Exception as e:
#         print(f"Feedback error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == "__main__":
#     print("Initializing Inverted Index...")
#     init_index()
#     print("Starting Flask server...")
#     app.run(debug=True, host='0.0.0.0', port=5001)

import sqlite3
import re
import numpy as np
from flask import Flask, request, jsonify
import json
import os
import pickle
from typing import List, Dict, Tuple
from underthesea import word_tokenize
from langdetect import detect
from nltk.corpus import stopwords
import nltk
from datetime import datetime
from rapidfuzz import fuzz
from math import log

app = Flask(__name__)

# Configuration
DB_PATH = "songs.db"
INDEX_FILE = "index.pkl"
STOPWORDS = {}
query_cache = {}  # Cache for search queries
documents = []  # List of tokenized documents
track_metadata = {}  # Metadata for tracks
inverted_index = None  # Inverted Index
feedback_weights = {}  # Feedback weights {track_id: {'correct': count, 'wrong': count}}

# BM25 parameters
k1 = 1.5
b = 0.75

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    for lang in ['english', 'vietnamese', 'spanish', 'french', 'german']:
        if lang in stopwords.fileids():
            STOPWORDS[lang] = set(stopwords.words(lang))
except Exception as e:
    print(f"Warning: NLTK resource download failed: {str(e)}. Stopwords filtering may be disabled.")

def clean_text(text: str) -> str:
    """Clean text by removing special characters and converting to lowercase"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str, lang: str = None) -> List[str]:
    """Tokenize text into words based on language"""
    if not text:
        print("Tokenize: Empty input text")
        return []
    cleaned_text = clean_text(text)
    if not cleaned_text:
        print("Tokenize: Empty cleaned text")
        return []
    if not lang:
        try:
            lang = detect(cleaned_text)
        except:
            lang = 'en'
    
    try:
        if lang.startswith('vi'):
            tokens = word_tokenize(cleaned_text)
        else:
            tokens = nltk.word_tokenize(cleaned_text)
    except Exception as e:
        print(f"Tokenize: NLTK failed for lang {lang}, text: {cleaned_text[:50]}... Error: {str(e)}")
        tokens = cleaned_text.split()
    
    if lang in STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS[lang]]
    
    print(f"Tokenize: Lang {lang}, Text: {cleaned_text[:50]}... -> Tokens: {tokens[:10]}")
    return tokens

def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-grams from a list of tokens"""
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

class InvertedIndex:
    """Custom Inverted Index for BM25 scoring"""
    def __init__(self):
        self.index = {}  # {term: [(doc_id, tf, field_boost), ...]}
        self.doc_lengths = {}  # {doc_id: length}
        self.avgdl = 0  # Average document length
        self.N = 0  # Number of documents

    def add_document(self, doc_id: int, tokens: List[str], field_boost: float = 1.0, is_lyrics: bool = False):
        """Add a document to the index, optionally with n-grams for lyrics"""
        term_freq = {}
        all_tokens = tokens.copy()
        if is_lyrics:
            # Add 2-grams and 3-grams for lyrics
            all_tokens.extend(generate_ngrams(tokens, 2))
            all_tokens.extend(generate_ngrams(tokens, 3))
        
        for term in all_tokens:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        for term, tf in term_freq.items():
            if term not in self.index:
                self.index[term] = []
            self.index[term].append((doc_id, tf, field_boost))
        
        doc_length = len(tokens)  # Use original token count for length
        self.doc_lengths[doc_id] = self.doc_lengths.get(doc_id, 0) + doc_length
        self.N = max(self.N, doc_id + 1)
        self.avgdl = (self.avgdl * (self.N - 1) + doc_length) / self.N if self.N > 0 else 0

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculate BM25 scores for all documents"""
        scores = [0.0] * self.N
        for term in query_tokens:
            if term in self.index:
                n = len(self.index[term])
                idf = log((self.N - n + 0.5) / (n + 0.5) + 1)
                for doc_id, tf, field_boost in self.index[term]:
                    dl = self.doc_lengths[doc_id]
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                    scores[doc_id] += score * field_boost
        return scores

def save_index():
    """Save Inverted Index and metadata to file"""
    try:
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'track_metadata': track_metadata,
                'feedback_weights': feedback_weights,
                'inverted_index': inverted_index
            }, f)
        print(f"Saved index to {INDEX_FILE}")
    except Exception as e:
        print(f"Error saving index: {str(e)}")

def load_index():
    """Load Inverted Index and metadata from file"""
    global documents, track_metadata, feedback_weights, inverted_index
    try:
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
            documents = data['documents']
            track_metadata = data['track_metadata']
            feedback_weights = data['feedback_weights']
            inverted_index = data['inverted_index']
            print(f"Loaded index from {INDEX_FILE}, {inverted_index.N} documents")
            return True
        return False
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return False

def check_db_changed() -> bool:
    """Check if database has changed since last index save"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        track_count = cursor.fetchone()[0]
        conn.close()
        
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
            return len(data['track_metadata']) != track_count
        return True
    except Exception as e:
        print(f"Error checking DB: {str(e)}")
        return True

def init_index():
    """Initialize Inverted Index from SQLite database or load from file"""
    global inverted_index, track_metadata, documents, feedback_weights
    
    if not check_db_changed() and load_index():
        return
    
    print("Creating Inverted Index...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            track_id INTEGER,
            is_correct BOOLEAN,
            timestamp DATETIME
        )
    """)
    
    cursor.execute("""
        SELECT t.id, t.song_name, t.artists, t.external_urls, t.image_url, l.lyrics 
        FROM tracks t 
        LEFT JOIN lyrics l ON t.id = l.track_id
    """)
    
    documents.clear()
    track_metadata.clear()
    inverted_index = InvertedIndex()
    valid_documents = 0
    print("Processing tracks and preparing documents...")
    for idx, row in enumerate(cursor.fetchall()):
        track_id, song_name, artists_json, external_urls, image_url, lyrics = row
        try:
            artist_ids = json.loads(artists_json) if artists_json else []
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in artists for track {song_name} (ID: {track_id}): {artists_json}")
            artist_ids = []
        
        artist_names = []
        primary_artist = None
        if artist_ids:
            format_strings = ','.join(['?'] * len(artist_ids))
            cursor.execute(f"SELECT id, name FROM artists WHERE id IN ({format_strings})", artist_ids)
            artist_dict = {str(row[0]): row[1] for row in cursor.fetchall()}
            artist_names = [artist_dict.get(str(aid)) for aid in artist_ids if str(aid) in artist_dict]
            primary_artist = artist_names[0] if artist_names else None
            print(f"Track {idx + 1}: {song_name}, Artists: {artist_names}, Primary: {primary_artist}")
        else:
            print(f"Track {idx + 1}: {song_name}, No artists found")
        
        # Create document and add to index
        tokenized_doc = []
        if lyrics:
            tokens = tokenize(lyrics)
            inverted_index.add_document(valid_documents, tokens, field_boost=4.0, is_lyrics=True)
            tokenized_doc.extend(tokens)
        song_tokens = tokenize(song_name)
        inverted_index.add_document(valid_documents, song_tokens, field_boost=3.0)
        tokenized_doc.extend(song_tokens)
        if primary_artist:
            artist_tokens = tokenize(primary_artist)
            inverted_index.add_document(valid_documents, artist_tokens, field_boost=3.0)
            tokenized_doc.extend(artist_tokens)
        if len(artist_names) > 1:
            other_artists = " ".join(artist_names[1:])
            other_tokens = tokenize(other_artists)
            inverted_index.add_document(valid_documents, other_tokens, field_boost=1.0)
            tokenized_doc.extend(other_tokens)
        
        if tokenized_doc:
            documents.append(tokenized_doc)
            track_metadata[str(valid_documents)] = {
                "id": track_id,
                "song_name": song_name,
                "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
                "external_urls": external_urls,
                "image_url": image_url,
                "has_lyrics": bool(lyrics)
            }
            valid_documents += 1
        else:
            print(f"Warning: Empty document for track {song_name} (ID: {track_id})")
    
    print(f"Total tracks processed: {idx + 1}, Valid documents: {valid_documents}")
    
    if documents:
        cursor.execute("SELECT track_id, is_correct, COUNT(*) as count FROM feedback GROUP BY track_id, is_correct")
        feedback_weights = {}
        for row in cursor.fetchall():
            track_id, is_correct, count = row
            if str(track_id) not in feedback_weights:
                feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
            if is_correct:
                feedback_weights[str(track_id)]['correct'] = count
            else:
                feedback_weights[str(track_id)]['wrong'] = count
        
        print("Inverted Index creation completed!")
        save_index()
    else:
        print("Error: No valid documents found. Inverted Index not created.")
        inverted_index = None
    
    conn.close()

def add_new_song(song_name: str, artist_names: List[str], external_urls: str, image_url: str, lyrics: str) -> bool:
    """Add a new song to SQLite and update Inverted Index"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        artist_ids = []
        for name in artist_names:
            cursor.execute("SELECT id FROM artists WHERE name = ?", (name,))
            artist_row = cursor.fetchone()
            if artist_row:
                artist_id = artist_row[0]
            else:
                cursor.execute("INSERT INTO artists (name) VALUES (?)", (name,))
                artist_id = cursor.lastrowid
            artist_ids.append(artist_id)
        
        artists_json = json.dumps(artist_ids)
        cursor.execute("INSERT INTO tracks (song_name, artists, external_urls, image_url) VALUES (?, ?, ?, ?)", 
                      (song_name, artists_json, external_urls, image_url))
        track_id = cursor.lastrowid
        
        if lyrics:
            cursor.execute("INSERT INTO lyrics (track_id, lyrics) VALUES (?, ?)", (track_id, lyrics))
        
        # Update Inverted Index incrementally
        tokenized_doc = []
        doc_id = len(documents)
        if lyrics:
            tokens = tokenize(lyrics)
            inverted_index.add_document(doc_id, tokens, field_boost=4.0, is_lyrics=True)
            tokenized_doc.extend(tokens)
        song_tokens = tokenize(song_name)
        inverted_index.add_document(doc_id, song_tokens, field_boost=3.0)
        tokenized_doc.extend(song_tokens)
        if artist_names:
            artist_tokens = tokenize(artist_names[0])
            inverted_index.add_document(doc_id, artist_tokens, field_boost=3.0)
            tokenized_doc.extend(artist_tokens)
        if len(artist_names) > 1:
            other_artists = " ".join(artist_names[1:])
            other_tokens = tokenize(other_artists)
            inverted_index.add_document(doc_id, other_tokens, field_boost=1.0)
            tokenized_doc.extend(other_tokens)
        
        if tokenized_doc:
            documents.append(tokenized_doc)
            track_metadata[str(doc_id)] = {
                "id": track_id,
                "song_name": song_name,
                "artist_name": ", ".join(artist_names) if artist_names else "Unknown",
                "external_urls": external_urls,
                "image_url": image_url,
                "has_lyrics": bool(lyrics)
            }
            save_index()
        else:
            print(f"Warning: Empty document for new song {song_name}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding song: {str(e)}")
        return False

def save_feedback(query: str, track_id: int, is_correct: bool):
    """Save user feedback to SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (query, track_id, is_correct, timestamp) VALUES (?, ?, ?, ?)",
                      (query, track_id, is_correct, datetime.now()))
        conn.commit()
        cursor.execute("SELECT is_correct, COUNT(*) FROM feedback WHERE track_id = ? GROUP BY is_correct", (track_id,))
        feedback_weights[str(track_id)] = {'correct': 0, 'wrong': 0}
        for row in cursor.fetchall():
            is_correct, count = row
            if is_correct:
                feedback_weights[str(track_id)]['correct'] = count
            else:
                feedback_weights[str(track_id)]['wrong'] = count
        conn.close()
        save_index()
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")

def search_songs(query: str, k: int = 10) -> List[Dict]:
    """Search songs using Inverted Index with BM25 scoring and phrase matching"""
    print(f"Searching for query: {query[:50]}...")
    query_clean = clean_text(query)
    if not query_clean:
        print("No valid query.")
        return []
    
    if query_clean in query_cache:
        print(f"Returning cached results for query: {query_clean}")
        return query_cache[query_clean]
    
    if inverted_index is None:
        print("Error: Inverted Index not initialized.")
        return []
    
    # Fuzzy string and phrase matching
    string_scores = {}
    query_tokens = tokenize(query_clean)
    for idx, metadata in track_metadata.items():
        song_name_clean = clean_text(metadata["song_name"])
        artist_name_clean = clean_text(metadata["artist_name"] or "")
        lyrics_clean = clean_text(metadata.get("lyrics", "")) if metadata["has_lyrics"] else ""
        
        # Fuzzy matching for song and artist
        score_song = fuzz.ratio(query_clean, song_name_clean)
        if score_song > 95:
            string_scores[int(idx)] = float(score_song + 30)
        elif score_song > 70:
            string_scores[int(idx)] = float(score_song)
        if artist_name_clean:
            score_artist = fuzz.ratio(query_clean, artist_name_clean)
            if score_artist > 95:
                string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist + 30))
            elif score_artist > 70:
                string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_artist))
        
        # Phrase matching for lyrics
        if lyrics_clean and len(query_clean.split()) > 2:  # Only for longer queries
            score_lyrics = fuzz.partial_ratio(query_clean, lyrics_clean)
            if score_lyrics > 95:
                string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_lyrics + 50))
            elif score_lyrics > 80:
                string_scores[int(idx)] = max(string_scores.get(int(idx), 0), float(score_lyrics + 20))
        
        # Token matching
        for q in query_tokens:
            if q in song_name_clean or (artist_name_clean and q in artist_name_clean) or (lyrics_clean and q in lyrics_clean):
                string_scores[int(idx)] = max(string_scores.get(int(idx), 0), 60.0)
    
    # BM25 scoring with Inverted Index
    query_ngrams = query_tokens + generate_ngrams(query_tokens, 2) + generate_ngrams(query_tokens, 3)
    bm25_scores = inverted_index.get_scores(query_ngrams)
    
    # Combine scores
    results = []
    for idx, bm25_score in enumerate(bm25_scores):
        if str(idx) in track_metadata:
            metadata = track_metadata[str(idx)]
            string_score = string_scores.get(idx, 0.0)
            fb = feedback_weights.get(str(metadata["id"]), {'correct': 0, 'wrong': 0})
            feedback_boost = float(fb['correct'] * 10.0 - fb['wrong'] * 5.0)
            combined_score = float(max(bm25_score * 100, string_score) + feedback_boost)
            if combined_score > 0:
                results.append({
                    "song_name": metadata["song_name"],
                    "artist_name": metadata["artist_name"],
                    "external_urls": metadata["external_urls"],
                    "image_url": metadata["image_url"],
                    "match_score": round(combined_score, 2),
                    "has_lyrics": metadata["has_lyrics"],
                    "track_id": metadata["id"]
                })
    
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:k]
    query_cache[query_clean] = results
    result_strings = [f"{r['song_name']} by {r['artist_name'] or 'Unknown'}" for r in results]
    print(f"Found {len(results)} results: {result_strings}")
    return results

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 500

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with optional field prefix"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        query = data.get('query', '').strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Xác định loại truy vấn
        lower_query = query.lower()
        results = []

        if lower_query.startswith("artist:"):
            keyword = query[7:].strip().lower()
            cursor.execute("""
                SELECT t.id, t.song_name, group_concat(a.name), l.lyrics, t.image_url, t.external_urls
                FROM tracks t
                LEFT JOIN lyrics l ON t.id = l.track_id
                JOIN json_each(t.artists) AS je ON 1
                JOIN artists a ON a.id = je.value
                WHERE lower(a.name) LIKE ?
                GROUP BY t.id
            """, ('%' + keyword + '%',))
            results = cursor.fetchall()

        elif lower_query.startswith("name:"):
            keyword = query[5:].strip().lower()
            cursor.execute("""
                SELECT t.id, t.song_name, group_concat(a.name), l.lyrics, t.image_url, t.external_urls
                FROM tracks t
                LEFT JOIN lyrics l ON t.id = l.track_id
                JOIN json_each(t.artists) AS je ON 1
                JOIN artists a ON a.id = je.value
                WHERE lower(t.song_name) LIKE ?
                GROUP BY t.id
            """, ('%' + keyword + '%',))
            results = cursor.fetchall()

        elif lower_query.startswith("lyrics:"):
            keyword = query[7:].strip().lower()
            cursor.execute("""
                SELECT t.id, t.song_name, group_concat(a.name), l.lyrics, t.image_url, t.external_urls
                FROM tracks t
                LEFT JOIN lyrics l ON t.id = l.track_id
                JOIN json_each(t.artists) AS je ON 1
                JOIN artists a ON a.id = je.value
                WHERE lower(l.lyrics) LIKE ?
                GROUP BY t.id
            """, ('%' + keyword + '%',))
            results = cursor.fetchall()

        else:
            # Truy vấn tổng hợp nếu không có tiền tố
            keyword = query.lower()
            cursor.execute("""
                SELECT t.id, t.song_name, group_concat(a.name), l.lyrics, t.image_url, t.external_urls
                FROM tracks t
                LEFT JOIN lyrics l ON t.id = l.track_id
                JOIN json_each(t.artists) AS je ON 1
                JOIN artists a ON a.id = je.value
                WHERE lower(t.song_name) LIKE ?
                   OR lower(l.lyrics) LIKE ?
                   OR lower(a.name) LIKE ?
                GROUP BY t.id
            """, (
                '%' + keyword + '%',
                '%' + keyword + '%',
                '%' + keyword + '%',
            ))
            results = cursor.fetchall()

        conn.close()

        # Convert results
        formatted = [
            {
                "track_id": row[0],
                "song_name": row[1],
                "artist_name": row[2],
                "has_lyrics": bool(row[3]),
                "image_url": row[4],
                "external_urls": row[5],
                "match_score": 100.0
            }
            for row in results
        ]
        return jsonify(formatted)

    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/add_song', methods=['POST'])
def add_song():
    """Handle adding new songs"""
    try:
        data = request.json
        song_name = data.get('song_name', '')
        artist_names = data.get('artist_names', [])
        external_urls = data.get('external_urls', '')
        image_url = data.get('image_url', '')
        lyrics = data.get('lyrics', '')
        
        if not song_name or not artist_names:
            return jsonify({"error": "Song name and at least one artist name are required"}), 400
        
        success = add_new_song(song_name, artist_names, external_urls, image_url, lyrics)
        if success:
            return jsonify({"message": "Song added successfully"})
        else:
            return jsonify({"error": "Failed to add song"}), 500
    except ValueError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        print(f"Add song error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback"""
    try:
        data = request.json
        query = data.get('query', '')
        track_id = data.get('track_id', None)
        is_correct = data.get('is_correct', None)
        
        if not query or track_id is None or is_correct is None:
            return jsonify({"error": "Query, track_id, and is_correct are required"}), 400
        
        save_feedback(query, track_id, is_correct)
        return jsonify({"message": "Feedback saved successfully"})
    except ValueError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Initializing Inverted Index...")
    init_index()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)