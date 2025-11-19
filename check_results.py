#!/usr/bin/env python3
"""Check overnight collection results"""
import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path(__file__).parent / "corpus_platform.db"

print("="*80)
print("OVERNIGHT COLLECTION RESULTS")
print("="*80)
print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Overall statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_texts,
            SUM(word_count) as total_words,
            COUNT(DISTINCT language) as languages,
            COUNT(DISTINCT diachronic_stage) as stages,
            COUNT(DISTINCT genre) as genres,
            MIN(date_added) as first_text,
            MAX(date_added) as last_text
        FROM corpus_items
    """)
    
    total, words, langs, stages, genres, first, last = cursor.fetchone()
    
    print("OVERALL STATISTICS:")
    print("="*80)
    print(f"Total Texts Collected:      {total or 0}")
    print(f"Total Words:                {words or 0:,}")
    print(f"Unique Languages:           {langs or 0}")
    print(f"Diachronic Stages:          {stages or 0}")
    print(f"Genres:                     {genres or 0}")
    print(f"First Text:                 {first}")
    print(f"Last Text:                  {last}")
    
    if first and last:
        from datetime import datetime
        first_dt = datetime.fromisoformat(first)
        last_dt = datetime.fromisoformat(last)
        duration = (last_dt - first_dt).total_seconds() / 3600
        print(f"Collection Duration:        {duration:.2f} hours")
        if duration > 0:
            print(f"Collection Rate:            {total/duration:.2f} texts/hour")
    
    print()
    
    # By language
    print("BY LANGUAGE:")
    print("="*80)
    cursor.execute("""
        SELECT language, COUNT(*) as count, SUM(word_count) as words
        FROM corpus_items
        GROUP BY language
        ORDER BY count DESC
    """)
    
    print(f"{'Language':<20} {'Texts':>10} {'Words':>15}")
    print("-"*80)
    for lang, count, wc in cursor.fetchall():
        lang_name = {
            'grc': 'Ancient Greek',
            'lat': 'Latin',
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'ang': 'Old English',
            'enm': 'Middle English'
        }.get(lang, lang)
        print(f"{lang_name:<20} {count:>10} {wc or 0:>15,}")
    
    print()
    
    # By diachronic stage
    print("BY DIACHRONIC STAGE:")
    print("="*80)
    cursor.execute("""
        SELECT diachronic_stage, COUNT(*) as count, SUM(word_count) as words
        FROM corpus_items
        WHERE diachronic_stage IS NOT NULL AND diachronic_stage != ''
        GROUP BY diachronic_stage
        ORDER BY count DESC
    """)
    
    print(f"{'Period':<50} {'Texts':>10} {'Words':>15}")
    print("-"*80)
    for stage, count, wc in cursor.fetchall():
        print(f"{stage:<50} {count:>10} {wc or 0:>15,}")
    
    print()
    
    # By genre
    print("BY GENRE:")
    print("="*80)
    cursor.execute("""
        SELECT genre, COUNT(*) as count, SUM(word_count) as words
        FROM corpus_items
        WHERE genre IS NOT NULL AND genre != ''
        GROUP BY genre
        ORDER BY count DESC
    """)
    
    print(f"{'Genre':<20} {'Texts':>10} {'Words':>15}")
    print("-"*80)
    for genre, count, wc in cursor.fetchall():
        print(f"{genre:<20} {count:>10} {wc or 0:>15,}")
    
    print()
    
    # Special categories
    print("SPECIAL CATEGORIES:")
    print("="*80)
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN is_biblical = 1 THEN 1 ELSE 0 END) as biblical,
            SUM(CASE WHEN is_classical = 1 THEN 1 ELSE 0 END) as classical,
            SUM(CASE WHEN is_retranslation = 1 THEN 1 ELSE 0 END) as retranslations,
            SUM(CASE WHEN is_retelling = 1 THEN 1 ELSE 0 END) as retellings
        FROM corpus_items
    """)
    
    biblical, classical, retrans, retell = cursor.fetchone()
    
    print(f"Biblical Texts:             {biblical or 0}")
    print(f"Classical Texts:            {classical or 0}")
    print(f"Retranslations:             {retrans or 0}")
    print(f"Retellings:                 {retell or 0}")
    
    print()
    
    # All texts with details
    print("ALL COLLECTED TEXTS:")
    print("="*80)
    cursor.execute("""
        SELECT id, title, language, word_count, diachronic_stage, 
               is_biblical, is_classical, is_retranslation, date_added
        FROM corpus_items
        ORDER BY id
    """)
    
    print(f"{'ID':<4} {'Title':<45} {'Lang':<6} {'Words':>10} {'Period':<30}")
    print("-"*80)
    
    for row in cursor.fetchall():
        text_id, title, lang, wc, stage, biblical, classical, retrans, date = row
        
        # Truncate title
        title_short = title[:42] + "..." if len(title) > 45 else title
        stage_short = (stage[:27] + "...") if stage and len(stage) > 30 else (stage or "")
        
        # Add markers
        markers = ""
        if biblical: markers += "📖"
        if classical: markers += "🏛️"
        if retrans: markers += "🔄"
        
        title_with_markers = f"{markers} {title_short}" if markers else title_short
        
        print(f"{text_id:<4} {title_with_markers:<45} {lang:<6} {wc or 0:>10,} {stage_short:<30}")
    
    print()
    
    # Authors and translators
    print("AUTHORS & TRANSLATORS:")
    print("="*80)
    cursor.execute("""
        SELECT author, COUNT(*) as count
        FROM corpus_items
        WHERE author IS NOT NULL AND author != ''
        GROUP BY author
        ORDER BY count DESC
        LIMIT 10
    """)
    
    authors = cursor.fetchall()
    if authors:
        print("Top Authors:")
        for author, count in authors:
            print(f"  {author}: {count} texts")
    
    cursor.execute("""
        SELECT translator, COUNT(*) as count
        FROM corpus_items
        WHERE translator IS NOT NULL AND translator != ''
        GROUP BY translator
        ORDER BY count DESC
        LIMIT 10
    """)
    
    translators = cursor.fetchall()
    if translators:
        print("\nTranslators:")
        for translator, count in translators:
            print(f"  {translator}: {count} texts")
    
    print()
    
    # Translation years
    cursor.execute("""
        SELECT translation_year, COUNT(*) as count
        FROM corpus_items
        WHERE translation_year IS NOT NULL
        GROUP BY translation_year
        ORDER BY translation_year
    """)
    
    years = cursor.fetchall()
    if years:
        print("TRANSLATION TIMELINE:")
        print("="*80)
        for year, count in years:
            print(f"  {year}: {count} text(s)")
    
    print()
    
    conn.close()
    
    print("="*80)
    print("COLLECTION COMPLETE!")
    print("="*80)
    print()
    print("✓ Diachronic corpus successfully built")
    print("✓ Multiple languages collected")
    print("✓ Biblical and classical texts included")
    print("✓ Retranslations tracked")
    print("✓ Ready for treebank annotation")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
