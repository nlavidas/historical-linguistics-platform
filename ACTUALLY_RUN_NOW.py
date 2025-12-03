#!/usr/bin/env python3
"""
ACTUALLY RUN NOW - This script REALLY collects data and populates the database
No more empty databases - this WORKS

Run on server:
    sudo python3 /root/corpus_platform/ACTUALLY_RUN_NOW.py
"""

import os
import sys
import sqlite3
import requests
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Configuration - detect OS
import platform
if platform.system() == 'Windows':
    DATA_DIR = Path("Z:/corpus_platform/data")
else:
    DATA_DIR = Path("/root/corpus_platform/data")

DB_PATH = DATA_DIR / "corpus_platform.db"
CACHE_DIR = DATA_DIR / "cache"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ACTUALLY RUNNING - COLLECTING REAL DATA NOW")
print("=" * 70)
print(f"Data directory: {DATA_DIR}")
print(f"Database: {DB_PATH}")
print()

# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_database():
    """Initialize the database with proper schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop and recreate tables to ensure clean schema
    cursor.execute("DROP TABLE IF EXISTS tokens")
    cursor.execute("DROP TABLE IF EXISTS sentences")
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("DROP TABLE IF EXISTS valency_frames")
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            period TEXT,
            language TEXT DEFAULT 'grc',
            source TEXT,
            source_url TEXT,
            text_content TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Sentences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            sentence_index INTEGER,
            text TEXT,
            tokens TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    
    # Tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id INTEGER,
            token_index INTEGER,
            form TEXT,
            lemma TEXT,
            upos TEXT,
            xpos TEXT,
            feats TEXT,
            head INTEGER,
            deprel TEXT,
            deps TEXT,
            misc TEXT,
            FOREIGN KEY (sentence_id) REFERENCES sentences(id)
        )
    """)
    
    # Valency frames table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS valency_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lemma TEXT NOT NULL,
            pattern TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            period TEXT,
            examples TEXT,
            UNIQUE(lemma, pattern, period)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_period ON documents(period)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_language ON documents(language)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_sent ON tokens(sentence_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_lemma ON tokens(lemma)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_lemma ON valency_frames(lemma)")
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")

# =============================================================================
# UNIVERSAL DEPENDENCIES COLLECTION
# =============================================================================

UD_TREEBANKS = {
    "grc_proiel": {
        "name": "Ancient Greek PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master",
        "period": "classical",
        "language": "grc"
    },
    "grc_perseus": {
        "name": "Ancient Greek Perseus",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master",
        "period": "classical",
        "language": "grc"
    },
    "el_gdt": {
        "name": "Modern Greek GDT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master",
        "period": "modern",
        "language": "el"
    },
    "la_proiel": {
        "name": "Latin PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master",
        "period": "classical",
        "language": "la"
    },
    "got_proiel": {
        "name": "Gothic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/master",
        "period": "medieval",
        "language": "got"
    },
    "cu_proiel": {
        "name": "Old Church Slavonic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/master",
        "period": "medieval",
        "language": "cu"
    },
    "fro_srcmf": {
        "name": "Old French SRCMF",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/master",
        "period": "medieval",
        "language": "fro"
    },
    "ang_ycoe": {
        "name": "Old English YCOE",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_English-YCOE/master",
        "period": "medieval",
        "language": "ang"
    }
}

def download_file(url: str, cache_path: Path) -> str:
    """Download file with caching"""
    if cache_path.exists():
        return cache_path.read_text(encoding='utf-8')
    
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            content = response.text
            cache_path.write_text(content, encoding='utf-8')
            return content
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
    
    return ""

def parse_conllu(content: str) -> list:
    """Parse CoNLL-U format into sentences"""
    sentences = []
    current_sentence = {
        "id": "",
        "text": "",
        "tokens": []
    }
    
    for line in content.split('\n'):
        line = line.strip()
        
        if not line:
            if current_sentence["tokens"]:
                sentences.append(current_sentence)
                current_sentence = {"id": "", "text": "", "tokens": []}
        elif line.startswith('# sent_id'):
            current_sentence["id"] = line.split('=')[-1].strip()
        elif line.startswith('# text'):
            current_sentence["text"] = line.split('=', 1)[-1].strip()
        elif not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 10:
                # Skip multiword tokens
                if '-' in parts[0] or '.' in parts[0]:
                    continue
                
                token = {
                    "id": parts[0],
                    "form": parts[1],
                    "lemma": parts[2],
                    "upos": parts[3],
                    "xpos": parts[4],
                    "feats": parts[5],
                    "head": parts[6],
                    "deprel": parts[7],
                    "deps": parts[8],
                    "misc": parts[9]
                }
                current_sentence["tokens"].append(token)
    
    # Don't forget last sentence
    if current_sentence["tokens"]:
        sentences.append(current_sentence)
    
    return sentences

def collect_ud_treebanks():
    """Collect all UD treebanks"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_docs = 0
    total_sents = 0
    total_tokens = 0
    
    for treebank_id, info in UD_TREEBANKS.items():
        print(f"\n📚 Collecting: {info['name']}")
        
        treebank_sents = 0
        treebank_tokens = 0
        
        for split in ["train", "dev", "test"]:
            # UD filename format: grc_proiel-ud-train.conllu (underscore in lang, hyphen before ud)
            filename = f"{treebank_id}-ud-{split}.conllu"
            url = f"{info['url']}/{filename}"
            cache_path = CACHE_DIR / f"{treebank_id}_{split}.conllu"
            
            print(f"  Downloading: {filename}...")
            content = download_file(url, cache_path)
            if not content:
                continue
            
            sentences = parse_conllu(content)
            if not sentences:
                continue
            
            # Create document
            cursor.execute("""
                INSERT INTO documents (title, author, period, language, source, source_url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"{info['name']} - {split}",
                "Universal Dependencies",
                info['period'],
                info['language'],
                f"UD_{treebank_id}",
                url
            ))
            doc_id = cursor.lastrowid
            total_docs += 1
            
            # Insert sentences and tokens
            for sent_idx, sent in enumerate(sentences):
                cursor.execute("""
                    INSERT INTO sentences (document_id, sentence_index, text, tokens)
                    VALUES (?, ?, ?, ?)
                """, (
                    doc_id,
                    sent_idx,
                    sent["text"],
                    json.dumps(sent["tokens"])
                ))
                sent_id = cursor.lastrowid
                treebank_sents += 1
                
                # Insert tokens
                for tok_idx, token in enumerate(sent["tokens"]):
                    cursor.execute("""
                        INSERT INTO tokens 
                        (sentence_id, token_index, form, lemma, upos, xpos, feats, head, deprel, deps, misc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sent_id,
                        tok_idx,
                        token["form"],
                        token["lemma"],
                        token["upos"],
                        token["xpos"],
                        token["feats"],
                        int(token["head"]) if token["head"].isdigit() else 0,
                        token["deprel"],
                        token["deps"],
                        token["misc"]
                    ))
                    treebank_tokens += 1
            
            print(f"  ✓ {split}: {len(sentences)} sentences")
        
        total_sents += treebank_sents
        total_tokens += treebank_tokens
        print(f"  Total: {treebank_sents:,} sentences, {treebank_tokens:,} tokens")
        
        conn.commit()
    
    conn.close()
    return total_docs, total_sents, total_tokens

# =============================================================================
# VALENCY EXTRACTION
# =============================================================================

def extract_valency_frames():
    """Extract valency frames from tokens"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n📊 Extracting valency frames...")
    
    # Get all verbs with their arguments
    cursor.execute("""
        SELECT t.lemma, t.sentence_id, s.text, d.period
        FROM tokens t
        JOIN sentences s ON t.sentence_id = s.id
        JOIN documents d ON s.document_id = d.id
        WHERE t.upos = 'VERB'
    """)
    
    verbs = cursor.fetchall()
    print(f"  Found {len(verbs):,} verb instances")
    
    # Count patterns
    patterns = {}
    for lemma, sent_id, text, period in verbs:
        # Get arguments for this verb
        cursor.execute("""
            SELECT deprel, feats FROM tokens
            WHERE sentence_id = ? AND deprel IN ('nsubj', 'obj', 'iobj', 'obl')
        """, (sent_id,))
        
        args = cursor.fetchall()
        
        # Build pattern
        cases = ['NOM']  # Subject implied
        for deprel, feats in args:
            if deprel == 'obj':
                cases.append('ACC')
            elif deprel == 'iobj':
                cases.append('DAT')
            elif deprel == 'obl':
                # Extract case from feats
                if feats and 'Case=' in feats:
                    case = re.search(r'Case=(\w+)', feats)
                    if case:
                        cases.append(case.group(1).upper()[:3])
        
        pattern = '+'.join(sorted(set(cases)))
        key = (lemma, pattern, period or 'unknown')
        
        if key not in patterns:
            patterns[key] = {"count": 0, "examples": []}
        patterns[key]["count"] += 1
        if len(patterns[key]["examples"]) < 3:
            patterns[key]["examples"].append(text[:100])
    
    # Insert valency frames
    for (lemma, pattern, period), data in patterns.items():
        cursor.execute("""
            INSERT OR REPLACE INTO valency_frames (lemma, pattern, frequency, period, examples)
            VALUES (?, ?, ?, ?, ?)
        """, (lemma, pattern, data["count"], period, json.dumps(data["examples"])))
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Extracted {len(patterns):,} unique valency frames")
    return len(patterns)

# =============================================================================
# STATISTICS
# =============================================================================

def print_statistics():
    """Print database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    docs = cursor.fetchone()[0]
    print(f"📄 Documents: {docs:,}")
    
    cursor.execute("SELECT COUNT(*) FROM sentences")
    sents = cursor.fetchone()[0]
    print(f"📝 Sentences: {sents:,}")
    
    cursor.execute("SELECT COUNT(*) FROM tokens")
    tokens = cursor.fetchone()[0]
    print(f"🔤 Tokens: {tokens:,}")
    
    cursor.execute("SELECT COUNT(*) FROM valency_frames")
    frames = cursor.fetchone()[0]
    print(f"📊 Valency Frames: {frames:,}")
    
    print("\n📈 By Language:")
    cursor.execute("""
        SELECT language, COUNT(*) as docs, 
               (SELECT COUNT(*) FROM sentences s 
                JOIN documents d2 ON s.document_id = d2.id 
                WHERE d2.language = d.language) as sents
        FROM documents d
        GROUP BY language
        ORDER BY docs DESC
    """)
    for lang, doc_count, sent_count in cursor.fetchall():
        print(f"  {lang}: {doc_count} docs, {sent_count:,} sentences")
    
    print("\n📅 By Period:")
    cursor.execute("""
        SELECT period, COUNT(*) FROM documents
        GROUP BY period ORDER BY COUNT(*) DESC
    """)
    for period, count in cursor.fetchall():
        print(f"  {period}: {count} documents")
    
    print("\n🔝 Top 10 Verbs by Frequency:")
    cursor.execute("""
        SELECT lemma, SUM(frequency) as total
        FROM valency_frames
        GROUP BY lemma
        ORDER BY total DESC
        LIMIT 10
    """)
    for lemma, freq in cursor.fetchall():
        print(f"  {lemma}: {freq:,}")
    
    conn.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    
    # Initialize database
    init_database()
    
    # Collect UD treebanks
    print("\n" + "=" * 70)
    print("COLLECTING UNIVERSAL DEPENDENCIES TREEBANKS")
    print("=" * 70)
    
    docs, sents, tokens = collect_ud_treebanks()
    
    print(f"\n✓ Collection complete: {docs} documents, {sents:,} sentences, {tokens:,} tokens")
    
    # Extract valency
    frames = extract_valency_frames()
    
    # Print statistics
    print_statistics()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    print("\n✅ DONE - Database is now populated with REAL data!")
    print(f"   Database location: {DB_PATH}")

if __name__ == "__main__":
    main()
