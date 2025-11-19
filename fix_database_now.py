#!/usr/bin/env python3
"""Fix database structure immediately"""

import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "corpus_platform.db"

print("Fixing database structure...")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop old table
try:
    cursor.execute("DROP TABLE IF EXISTS corpus_items")
    print("✓ Dropped old table")
except:
    pass

# Create new table with correct structure
cursor.execute("""
    CREATE TABLE corpus_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT,
        language TEXT,
        content TEXT,
        proiel_xml TEXT,
        word_count INTEGER,
        date_added TEXT,
        status TEXT,
        metadata_quality REAL DEFAULT 0,
        annotation_score REAL DEFAULT 0,
        tokens_count INTEGER DEFAULT 0,
        lemmas_count INTEGER DEFAULT 0,
        pos_tags_count INTEGER DEFAULT 0,
        dependencies_count INTEGER DEFAULT 0
    )
""")

conn.commit()
conn.close()

print("✓ Database fixed with correct structure")
print("✓ Ready to collect texts")
