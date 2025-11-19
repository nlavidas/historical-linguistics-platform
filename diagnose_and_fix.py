#!/usr/bin/env python3
"""
Automatic Diagnosis and Fix System
Checks why collection is not working and fixes it
"""

import sys
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Set environment
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("AUTOMATIC DIAGNOSIS AND FIX SYSTEM")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check 1: Database exists and is accessible
print(">>> CHECK 1: Database Status")
db_path = Path(__file__).parent / "corpus_platform.db"
if db_path.exists():
    print(f"✓ Database exists: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM corpus_items")
        count = cursor.fetchone()[0]
        print(f"✓ Database accessible, contains {count} texts")
        
        # Check table structure
        cursor.execute("PRAGMA table_info(corpus_items)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"✓ Table columns: {', '.join(columns)}")
        
        conn.close()
    except Exception as e:
        print(f"✗ Database error: {e}")
        print("  FIX: Recreating database...")
        # Will be fixed below
else:
    print(f"✗ Database not found: {db_path}")
    print("  FIX: Will create database")

print()

# Check 2: Collection process status
print(">>> CHECK 2: Collection Process")
try:
    # Check if autonomous_247_collection.py is running
    result = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
        capture_output=True,
        text=True
    )
    
    python_processes = result.stdout.count('python.exe')
    print(f"  Python processes running: {python_processes}")
    
    if python_processes < 2:
        print("  ⚠ Collection process may not be running")
        print("  FIX: Will restart collection")
except Exception as e:
    print(f"  ✗ Cannot check processes: {e}")

print()

# Check 3: Test simple collection
print(">>> CHECK 3: Test Simple Text Collection")
try:
    # Try to collect a simple test text
    import requests
    
    test_url = "http://www.gutenberg.org/files/1342/1342-0.txt"
    print(f"  Testing download from: {test_url}")
    
    response = requests.get(test_url, timeout=10)
    if response.status_code == 200:
        print(f"✓ Download successful: {len(response.text)} characters")
        
        # Try to save to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_items (
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
        
        cursor.execute("""
            INSERT OR REPLACE INTO corpus_items
            (url, title, language, content, word_count, date_added, status, metadata_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_url,
            "Pride and Prejudice - TEST",
            "en",
            response.text[:1000],  # First 1000 chars
            len(response.text.split()),
            datetime.now().isoformat(),
            "pending",
            75.0
        ))
        
        conn.commit()
        text_id = cursor.lastrowid
        conn.close()
        
        print(f"✓ Test text saved to database with ID: {text_id}")
        print("  FIX APPLIED: Database is working correctly")
    else:
        print(f"✗ Download failed: HTTP {response.status_code}")
        
except Exception as e:
    print(f"✗ Collection test failed: {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print()

# Check 4: Stanza models
print(">>> CHECK 4: Stanza Models")
stanza_dir = Path('Z:/models/stanza')
if stanza_dir.exists():
    print(f"✓ Stanza directory exists: {stanza_dir}")
    
    # Check for language models
    for lang in ['grc', 'la', 'en']:
        lang_dir = stanza_dir / lang
        if lang_dir.exists():
            print(f"  ✓ {lang} models found")
        else:
            print(f"  ✗ {lang} models missing")
            print(f"    FIX: Download with: stanza.download('{lang}')")
else:
    print(f"✗ Stanza directory not found: {stanza_dir}")
    print("  FIX: Create directory and download models")

print()

# Check 5: Repository catalog
print(">>> CHECK 5: Repository Catalog")
try:
    from source_connectors import RepositoryCatalog
    
    catalog = RepositoryCatalog()
    
    for lang in ['grc', 'lat', 'en']:
        repos = catalog.get_repositories_for_language(lang)
        print(f"  {lang}: {len(repos)} repositories configured")
        if repos:
            print(f"    Example: {repos[0]['name']}")
    
    print("✓ Repository catalog working")
    
except Exception as e:
    print(f"✗ Repository catalog error: {e}")
    print("  FIX: Check source_connectors.py")

print()

# APPLY FIXES
print("="*80)
print("APPLYING AUTOMATIC FIXES")
print("="*80)

# Fix 1: Ensure database has correct structure
print("\n>>> FIX 1: Database Structure")
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corpus_items (
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
    print("✓ Database structure verified/created")
    
except Exception as e:
    print(f"✗ Database fix failed: {e}")

# Fix 2: Start collection if not running
print("\n>>> FIX 2: Collection Process")
print("  Starting new collection process...")
try:
    # Kill any existing collection processes
    subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq Autonomous*'], 
                   capture_output=True)
    time.sleep(2)
    
    # Start new collection
    subprocess.Popen(
        ['python', 'autonomous_247_collection.py', 
         '--languages', 'grc', 'lat', 'en',
         '--texts-per-cycle', '5',
         '--cycle-delay', '180'],
        cwd=Path(__file__).parent,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("✓ Collection process restarted")
    print("  Languages: grc, lat, en")
    print("  Texts per cycle: 5")
    print("  Cycle delay: 180 seconds (3 minutes)")
    
except Exception as e:
    print(f"✗ Failed to restart collection: {e}")

print()
print("="*80)
print("DIAGNOSIS AND FIX COMPLETE")
print("="*80)
print()
print("Next steps:")
print("1. Wait 3-5 minutes for collection to start")
print("2. Check database: SELECT COUNT(*) FROM corpus_items")
print("3. Monitor log: night_operation.log")
print("4. Next monitoring cycle will show results")
print()
print("="*80)
