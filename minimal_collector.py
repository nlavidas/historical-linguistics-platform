"""Minimal text collector - lightweight, no crashes"""
import sqlite3
import requests
import time
from pathlib import Path
from datetime import datetime

db = Path(__file__).parent / "corpus_platform.db"

# Fix database
conn = sqlite3.connect(db)
conn.execute("DROP TABLE IF EXISTS corpus_items")
conn.execute("""CREATE TABLE corpus_items (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    language TEXT,
    content TEXT,
    word_count INTEGER,
    date_added TEXT,
    status TEXT,
    metadata_quality REAL DEFAULT 75
)""")
conn.commit()
conn.close()

print("Database ready")

# Simple texts to collect
texts = [
    ("http://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice", "en"),
    ("http://www.gutenberg.org/files/84/84-0.txt", "Frankenstein", "en"),
    ("http://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland", "en"),
    ("http://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities", "en"),
    ("http://www.gutenberg.org/files/174/174-0.txt", "Dorian Gray", "en"),
]

# Collect
for url, title, lang in texts:
    try:
        print(f"\nCollecting: {title}")
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            words = len(r.text.split())
            
            conn = sqlite3.connect(db)
            conn.execute("""INSERT OR REPLACE INTO corpus_items 
                (url, title, language, content, word_count, date_added, status, metadata_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (url, title, lang, r.text, words, datetime.now().isoformat(), 'completed', 100.0))
            conn.commit()
            conn.close()
            
            print(f"  Saved: {words:,} words")
        time.sleep(3)
    except Exception as e:
        print(f"  Error: {e}")

# Check results
conn = sqlite3.connect(db)
count = conn.execute("SELECT COUNT(*) FROM corpus_items").fetchone()[0]
total_words = conn.execute("SELECT SUM(word_count) FROM corpus_items").fetchone()[0] or 0
conn.close()

print(f"\n=== COMPLETE ===")
print(f"Texts: {count}")
print(f"Words: {total_words:,}")
