#!/usr/bin/env python3
"""Import external corpus exports into corpus_platform.db safely.

Usage examples
--------------

JSON (recommended):
    python import_external_corpus.py --input exported_texts.json --format json

CSV:
    python import_external_corpus.py --input exported_texts.csv --format csv

Expected fields (if present they are used, if missing they default):
    url, title, language, content, word_count, date_added, status,
    genre, period, author, translator, original_language, translation_year,
    is_retranslation, is_retelling, is_biblical, is_classical,
    text_type, diachronic_stage

Existing rows with the same URL are preserved (INSERT OR IGNORE),
so you will not overwrite already annotated texts.
"""

import argparse
import csv
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable, List

DB_PATH = Path(__file__).parent / "corpus_platform.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("import_external_corpus")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure corpus_items table exists with extended diachronic + annotation fields."""
    cur = conn.cursor()

    # Base table (compatible with DIACHRONIC_MULTILINGUAL_COLLECTOR)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            language TEXT,
            content TEXT,
            word_count INTEGER,
            date_added TEXT,
            status TEXT DEFAULT 'collected',
            metadata_quality REAL DEFAULT 0
        )
        """
    )

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(corpus_items)")}

    # Diachronic + metadata fields
    extra_cols = [
        ("genre", "TEXT"),
        ("period", "TEXT"),
        ("author", "TEXT"),
        ("translator", "TEXT"),
        ("original_language", "TEXT"),
        ("translation_year", "INTEGER"),
        ("is_retranslation", "BOOLEAN DEFAULT 0"),
        ("is_retelling", "BOOLEAN DEFAULT 0"),
        ("is_biblical", "BOOLEAN DEFAULT 0"),
        ("is_classical", "BOOLEAN DEFAULT 0"),
        ("text_type", "TEXT"),
        ("diachronic_stage", "TEXT"),
        # Treebank + annotation fields used by autonomous_247_collection & annotation_worker_247
        ("proiel_xml", "TEXT"),
        ("has_treebank", "BOOLEAN DEFAULT 0"),
        ("treebank_format", "TEXT"),
        ("annotation_date", "TEXT"),
        ("annotation_quality", "REAL DEFAULT 0"),
        ("annotation_score", "REAL DEFAULT 0"),
        ("tokens_count", "INTEGER DEFAULT 0"),
        ("lemmas_count", "INTEGER DEFAULT 0"),
        ("pos_tags_count", "INTEGER DEFAULT 0"),
        ("dependencies_count", "INTEGER DEFAULT 0"),
    ]

    for name, col_type in extra_cols:
        if name not in existing_cols:
            cur.execute(f"ALTER TABLE corpus_items ADD COLUMN {name} {col_type}")

    conn.commit()


def load_json(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Allow {"items": [...]} wrappers
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        return [data]
    return data


def load_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


BOOL_FIELDS = {
    "is_retranslation",
    "is_retelling",
    "is_biblical",
    "is_classical",
}


def normalize_bool(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().lower()
    return 1 if s in {"1", "true", "yes", "y"} else 0


def normalize_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map external row into the fields used by corpus_items."""
    row = dict(raw)  # copy

    # Ensure mandatory fields
    url = (row.get("url") or "").strip()
    title = (row.get("title") or "").strip()
    language = (row.get("language") or "").strip()
    content = row.get("content") or ""

    # Word count: trust incoming if present, else compute
    try:
        wc = int(row.get("word_count") or 0)
    except Exception:
        wc = 0
    if wc == 0 and isinstance(content, str):
        wc = len(content.split())

    date_added = row.get("date_added") or datetime.now().isoformat()
    status = row.get("status") or "collected"

    # Booleans
    for bf in BOOL_FIELDS:
        if bf in row:
            row[bf] = normalize_bool(row[bf])

    # Build normalized dict with defaults
    normalized = {
        "url": url,
        "title": title,
        "language": language,
        "content": content,
        "word_count": wc,
        "date_added": date_added,
        "status": status,
        "genre": row.get("genre", ""),
        "period": row.get("period", ""),
        "author": row.get("author", ""),
        "translator": row.get("translator", ""),
        "original_language": row.get("original_language", ""),
        "translation_year": row.get("translation_year"),
        "is_retranslation": row.get("is_retranslation", 0),
        "is_retelling": row.get("is_retelling", 0),
        "is_biblical": row.get("is_biblical", 0),
        "is_classical": row.get("is_classical", 0),
        "text_type": row.get("text_type", ""),
        "diachronic_stage": row.get("diachronic_stage", ""),
    }

    return normalized


def import_items(items: Iterable[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    stmt = (
        """
        INSERT OR IGNORE INTO corpus_items (
            url, title, language, content, word_count, date_added, status,
            genre, period, author, translator, original_language, translation_year,
            is_retranslation, is_retelling, is_biblical, is_classical,
            text_type, diachronic_stage, metadata_quality
        ) VALUES (
            :url, :title, :language, :content, :word_count, :date_added, :status,
            :genre, :period, :author, :translator, :original_language, :translation_year,
            :is_retranslation, :is_retelling, :is_biblical, :is_classical,
            :text_type, :diachronic_stage, :metadata_quality
        )
        """
    )

    for raw in items:
        normalized = normalize_row(raw)
        if not normalized["url"] or not normalized["content"]:
            skipped += 1
            continue

        # Simple metadata quality: presence of title, language, wc, date
        quality_checks = 0
        if normalized["title"]:
            quality_checks += 1
        if normalized["language"]:
            quality_checks += 1
        if normalized["word_count"] > 0:
            quality_checks += 1
        if normalized["date_added"]:
            quality_checks += 1
        normalized["metadata_quality"] = (quality_checks / 4) * 100

        cur.execute(stmt, normalized)
        if cur.rowcount > 0:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()

    logger.info("Import complete: %s inserted, %s skipped (duplicates or invalid).", inserted, skipped)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import external corpus exports into corpus_platform.db")
    p.add_argument("--input", required=True, help="Path to JSON or CSV file")
    p.add_argument("--format", choices=["json", "csv"], required=True, help="Input format")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        logger.error("Input file does not exist: %s", path)
        return

    logger.info("Loading data from %s (%s)", path, args.format)

    if args.format == "json":
        items = load_json(path)
    else:
        items = load_csv(path)

    if not items:
        logger.warning("No items found in input file.")
        return

    logger.info("Loaded %s items; importing into %s", len(items), DB_PATH)
    import_items(items)


if __name__ == "__main__":
    main()
