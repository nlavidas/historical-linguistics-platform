#!/usr/bin/env python3
"""
RUN ON SERVER - Simple, robust script that WORKS
No crashes, no blocking, just data collection

Usage on server:
    cd /root/corpus_platform
    git pull origin master
    python3 RUN_ON_SERVER.py
"""

import os
import sys
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("/root/corpus_platform/data")
DB_PATH = DATA_DIR / "corpus_platform.db"
CACHE_DIR = DATA_DIR / "cache"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CORPUS PLATFORM - Data Collection")
print("=" * 60)
print(f"Database: {DB_PATH}")
print()

# =============================================================================
# DATABASE
# =============================================================================

def init_db():
    """Initialize database with clean schema"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop old tables
    c.execute("DROP TABLE IF EXISTS tokens")
    c.execute("DROP TABLE IF EXISTS sentences") 
    c.execute("DROP TABLE IF EXISTS documents")
    c.execute("DROP TABLE IF EXISTS valency_frames")
    
    # Create tables
    c.execute("""CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        period TEXT,
        language TEXT DEFAULT 'grc',
        source TEXT,
        source_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""CREATE TABLE sentences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        sentence_index INTEGER,
        text TEXT,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )""")
    
    c.execute("""CREATE TABLE tokens (
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
        FOREIGN KEY (sentence_id) REFERENCES sentences(id)
    )""")
    
    c.execute("""CREATE TABLE valency_frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lemma TEXT NOT NULL,
        pattern TEXT NOT NULL,
        frequency INTEGER DEFAULT 1,
        period TEXT,
        UNIQUE(lemma, pattern, period)
    )""")
    
    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(document_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tok_sent ON tokens(sentence_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tok_lemma ON tokens(lemma)")
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")

# =============================================================================
# CONLL-U PARSER
# =============================================================================

def parse_conllu(content):
    """Parse CoNLL-U content"""
    sentences = []
    current = {"id": "", "text": "", "tokens": []}
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            if current["tokens"]:
                sentences.append(current)
                current = {"id": "", "text": "", "tokens": []}
        elif line.startswith('# sent_id'):
            current["id"] = line.split('=')[-1].strip()
        elif line.startswith('# text'):
            current["text"] = line.split('=', 1)[-1].strip()
        elif not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 10 and '-' not in parts[0] and '.' not in parts[0]:
                current["tokens"].append({
                    "id": parts[0],
                    "form": parts[1],
                    "lemma": parts[2],
                    "upos": parts[3],
                    "xpos": parts[4],
                    "feats": parts[5],
                    "head": parts[6],
                    "deprel": parts[7]
                })
    
    if current["tokens"]:
        sentences.append(current)
    
    return sentences

# =============================================================================
# DATA COLLECTION
# =============================================================================

# UD Treebanks to collect
TREEBANKS = {
    "grc_proiel": ("Ancient Greek PROIEL", "classical", "grc"),
    "grc_perseus": ("Ancient Greek Perseus", "classical", "grc"),
    "el_gdt": ("Modern Greek GDT", "modern", "el"),
    "la_proiel": ("Latin PROIEL", "classical", "la"),
    "got_proiel": ("Gothic PROIEL", "medieval", "got"),
    "cu_proiel": ("Old Church Slavonic PROIEL", "medieval", "cu"),
}

def download_file(url, cache_path):
    """Download file with caching"""
    if cache_path.exists():
        return cache_path.read_text(encoding='utf-8')
    
    try:
        import requests
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            cache_path.write_text(r.text, encoding='utf-8')
            return r.text
    except Exception as e:
        print(f"    Error: {e}")
    return ""

def collect_treebanks():
    """Collect all treebanks"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    total_docs = 0
    total_sents = 0
    total_toks = 0
    
    for tb_id, (name, period, lang) in TREEBANKS.items():
        print(f"\n📚 {name}")
        tb_sents = 0
        tb_toks = 0
        
        for split in ["train", "dev", "test"]:
            filename = f"{tb_id}-ud-{split}.conllu"
            url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_{tb_id.replace('_', '-').title()}/master/{filename}"
            
            # Fix URL format
            if tb_id == "grc_proiel":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master/{filename}"
            elif tb_id == "grc_perseus":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/{filename}"
            elif tb_id == "el_gdt":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master/{filename}"
            elif tb_id == "la_proiel":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master/{filename}"
            elif tb_id == "got_proiel":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/master/{filename}"
            elif tb_id == "cu_proiel":
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/master/{filename}"
            
            cache_path = CACHE_DIR / f"{tb_id}_{split}.conllu"
            
            print(f"  {split}...", end=" ", flush=True)
            content = download_file(url, cache_path)
            
            if not content:
                print("skip")
                continue
            
            sentences = parse_conllu(content)
            if not sentences:
                print("empty")
                continue
            
            # Insert document
            c.execute("""INSERT INTO documents (title, author, period, language, source, source_url)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     (f"{name} - {split}", "Universal Dependencies", period, lang, f"UD_{tb_id}", url))
            doc_id = c.lastrowid
            total_docs += 1
            
            # Insert sentences and tokens
            for sent_idx, sent in enumerate(sentences):
                c.execute("INSERT INTO sentences (document_id, sentence_index, text) VALUES (?, ?, ?)",
                         (doc_id, sent_idx, sent["text"]))
                sent_id = c.lastrowid
                tb_sents += 1
                
                for tok in sent["tokens"]:
                    c.execute("""INSERT INTO tokens (sentence_id, token_index, form, lemma, upos, xpos, feats, head, deprel)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                             (sent_id, int(tok["id"]), tok["form"], tok["lemma"], tok["upos"],
                              tok["xpos"], tok["feats"], int(tok["head"]) if tok["head"].isdigit() else 0, tok["deprel"]))
                    tb_toks += 1
            
            print(f"{len(sentences):,} sentences")
            conn.commit()
        
        total_sents += tb_sents
        total_toks += tb_toks
        print(f"  Total: {tb_sents:,} sentences, {tb_toks:,} tokens")
    
    conn.close()
    return total_docs, total_sents, total_toks

def extract_valency():
    """Extract valency frames from verbs"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all verbs with their dependents
    c.execute("""
        SELECT t1.lemma, t1.sentence_id, t2.deprel, d.period
        FROM tokens t1
        JOIN tokens t2 ON t1.sentence_id = t2.sentence_id AND t2.head = t1.token_index
        JOIN sentences s ON t1.sentence_id = s.id
        JOIN documents d ON s.document_id = d.id
        WHERE t1.upos = 'VERB'
    """)
    
    frames = {}
    for lemma, sent_id, deprel, period in c.fetchall():
        key = (lemma, period)
        if key not in frames:
            frames[key] = set()
        frames[key].add(deprel)
    
    # Insert frames
    count = 0
    for (lemma, period), deprels in frames.items():
        pattern = "+".join(sorted(deprels))
        try:
            c.execute("""INSERT OR REPLACE INTO valency_frames (lemma, pattern, frequency, period)
                        VALUES (?, ?, 1, ?)""", (lemma, pattern, period))
            count += 1
        except:
            pass
    
    conn.commit()
    conn.close()
    return count

def print_stats():
    """Print database statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    c.execute("SELECT COUNT(*) FROM documents")
    print(f"📄 Documents: {c.fetchone()[0]:,}")
    
    c.execute("SELECT COUNT(*) FROM sentences")
    print(f"📝 Sentences: {c.fetchone()[0]:,}")
    
    c.execute("SELECT COUNT(*) FROM tokens")
    print(f"🔤 Tokens: {c.fetchone()[0]:,}")
    
    c.execute("SELECT COUNT(*) FROM valency_frames")
    print(f"📊 Valency Frames: {c.fetchone()[0]:,}")
    
    print("\n📈 By Language:")
    c.execute("SELECT language, COUNT(*) FROM documents GROUP BY language")
    for lang, count in c.fetchall():
        c.execute("SELECT COUNT(*) FROM sentences s JOIN documents d ON s.document_id = d.id WHERE d.language = ?", (lang,))
        sents = c.fetchone()[0]
        print(f"  {lang}: {count} docs, {sents:,} sentences")
    
    print("\n🔝 Top 10 Verbs:")
    c.execute("""SELECT lemma, COUNT(*) as cnt FROM tokens WHERE upos = 'VERB' 
                GROUP BY lemma ORDER BY cnt DESC LIMIT 10""")
    for lemma, cnt in c.fetchall():
        print(f"  {lemma}: {cnt:,}")
    
    conn.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()
    
    print("\n[1/4] Initializing database...")
    init_db()
    
    print("\n[2/4] Collecting treebanks...")
    docs, sents, toks = collect_treebanks()
    print(f"\n✓ Collected: {docs} documents, {sents:,} sentences, {toks:,} tokens")
    
    print("\n[3/4] Extracting valency frames...")
    frames = extract_valency()
    print(f"✓ Extracted {frames:,} valency frames")
    
    print("\n[4/4] Statistics...")
    print_stats()
    
    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    print(f"\n✅ DONE! Database: {DB_PATH}")
