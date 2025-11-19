#!/usr/bin/env python3
"""Quick status check and improvement script"""
import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path(__file__).parent / "corpus_platform.db"

print("="*80)
print("SYSTEM STATUS CHECK")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as texts,
            SUM(word_count) as words,
            AVG(metadata_quality) as avg_quality,
            AVG(annotation_score) as avg_annotation
        FROM corpus_items
    """)
    
    texts, words, avg_quality, avg_annotation = cursor.fetchone()
    
    print(f"Texts Collected:        {texts or 0}")
    print(f"Total Words:            {words or 0:,}")
    print(f"Avg Metadata Quality:   {avg_quality or 0:.1f}%")
    print(f"Avg Annotation Score:   {avg_annotation or 0:.1f}%")
    print()
    
    # Get recent texts
    cursor.execute("""
        SELECT id, title, language, word_count, date_added
        FROM corpus_items
        ORDER BY id DESC
        LIMIT 5
    """)
    
    print("Recent Texts:")
    print("-"*80)
    for row in cursor.fetchall():
        text_id, title, lang, words, date = row
        print(f"{text_id:3d}. {title[:50]:50s} {lang:5s} {words:8,} words")
    
    conn.close()
    
    print()
    print("="*80)
    print("STATUS: RUNNING")
    print("="*80)
    
except Exception as e:
    print(f"Error: {e}")
    print("Creating database structure...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corpus_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            language TEXT,
            content TEXT,
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
    
    print("✓ Database structure created")
    print("✓ Ready for collection")
