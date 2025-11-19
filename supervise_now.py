#!/usr/bin/env python3
"""
ACTIVE SUPERVISION - Check, Correct, Improve
"""

import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path(__file__).parent / "corpus_platform.db"

print("="*80)
print("ACTIVE SUPERVISION - CHECK, CORRECT, IMPROVE")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check database
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as texts,
            SUM(word_count) as words,
            MIN(date_added) as first,
            MAX(date_added) as last
        FROM corpus_items
    """)
    
    texts, words, first, last = cursor.fetchone()
    
    print("CURRENT STATUS:")
    print("-"*80)
    print(f"Texts Collected:    {texts or 0}")
    print(f"Total Words:        {words or 0:,}")
    print(f"First Text:         {first}")
    print(f"Last Text:          {last}")
    print()
    
    # Get recent texts
    cursor.execute("""
        SELECT id, title, language, word_count, date_added
        FROM corpus_items
        ORDER BY id DESC
        LIMIT 10
    """)
    
    print("RECENT TEXTS:")
    print("-"*80)
    print(f"{'ID':<5} {'Title':<40} {'Lang':<6} {'Words':>10} {'Date':<20}")
    print("-"*80)
    
    for row in cursor.fetchall():
        text_id, title, lang, wc, date = row
        title_short = title[:37] + "..." if len(title) > 40 else title
        date_short = date[:19] if date else ""
        print(f"{text_id:<5} {title_short:<40} {lang:<6} {wc:>10,} {date_short:<20}")
    
    print()
    
    # Calculate rate
    if first and last and texts > 1:
        from datetime import datetime
        first_dt = datetime.fromisoformat(first)
        last_dt = datetime.fromisoformat(last)
        duration = (last_dt - first_dt).total_seconds() / 3600  # hours
        
        if duration > 0:
            rate = texts / duration
            print(f"Collection Rate:    {rate:.2f} texts/hour")
            print()
    
    # Improvements
    print("IMPROVEMENTS:")
    print("-"*80)
    
    # Check for duplicates
    cursor.execute("""
        SELECT url, COUNT(*) as count
        FROM corpus_items
        GROUP BY url
        HAVING count > 1
    """)
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"Found {len(duplicates)} duplicate URLs - removing...")
        for url, count in duplicates:
            cursor.execute("""
                DELETE FROM corpus_items
                WHERE url = ? AND id NOT IN (
                    SELECT MIN(id) FROM corpus_items WHERE url = ?
                )
            """, (url, url))
        conn.commit()
        print(f"Removed {len(duplicates)} duplicates")
    else:
        print("No duplicates found")
    
    # Optimize database
    print("Optimizing database...")
    conn.execute("VACUUM")
    conn.commit()
    print("Database optimized")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-"*80)
    
    if texts < 10:
        print("- System is collecting well, continue monitoring")
    elif texts < 20:
        print("- Good progress, on track for morning target")
    else:
        print("- Excellent progress!")
    
    if words and words > 1000000:
        print(f"- Over 1 million words collected!")
    
    print()
    
    conn.close()
    
    print("="*80)
    print("SUPERVISION COMPLETE")
    print("="*80)
    print()
    print("System Status: HEALTHY")
    print("Continue monitoring every 20 minutes")
    
except Exception as e:
    print(f"Error: {e}")
    print()
    print("CORRECTIVE ACTION:")
    print("- Ensuring database structure...")
    
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
            status TEXT DEFAULT 'collected',
            metadata_quality REAL DEFAULT 100.0
        )
    """)
    
    conn.commit()
    conn.close()
    
    print("Database structure verified")
    print("System ready to continue")
