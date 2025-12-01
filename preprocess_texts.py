#!/usr/bin/env python3
"""Preprocess corpus texts for downstream annotation and research.

This module focuses on:
- Normalising whitespace and newlines.
- Deriving a simple sentence segmentation for key languages (initially grc, lat).

Sentences are stored in a separate table corpus_sentences so that
PROIEL, open LLM annotators and other tools can reuse the same segmentation.

Run manually:
    python preprocess_texts.py

It is also wired into the professional cycle to run automatically.
"""

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess_texts")

TARGET_LANGS = ("grc", "lat")

# Rough, language-agnostic sentence splitter tuned slightly for Greek/Latin.
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?··;])\s+")


def normalise_text(text: str) -> str:
    """Apply light, safe normalisation (do not change content radically)."""
    if not text:
        return ""
    # Normalise newlines and collapse extra blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter.

    For Greek and Latin this is only an approximation, but it is deterministic
    and easy to inspect. Later you can replace this with a more advanced
    segmenter if desired without changing downstream code.
    """
    text = normalise_text(text)
    if not text:
        return []
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def ensure_sentences_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            sent_index INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            language TEXT,
            created_at TEXT,
            UNIQUE(item_id, sent_index)
        )
        """
    )
    conn.commit()


def items_needing_sentences(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join("?" for _ in TARGET_LANGS)
    params = list(TARGET_LANGS)

    cur.execute(
        f"""
        SELECT id, language, content
        FROM corpus_items
        WHERE language IN ({placeholders})
          AND content IS NOT NULL
          AND LENGTH(content) > 500
          AND id NOT IN (SELECT DISTINCT item_id FROM corpus_sentences)
        """,
        params,
    )
    rows = cur.fetchall()
    return rows


def preprocess(conn: sqlite3.Connection) -> int:
    """Preprocess items and insert sentence rows. Returns number of items processed."""
    ensure_sentences_table(conn)
    rows = items_needing_sentences(conn)

    if not rows:
        logger.info("No items need preprocessing (sentences already present).")
        return 0

    logger.info("Found %s items needing sentence segmentation.", len(rows))

    cur = conn.cursor()
    processed = 0

    for row in rows:
        item_id = row["id"]
        lang = row["language"] or ""
        text = row["content"] or ""

        sentences = split_sentences(text)
        if not sentences:
            logger.info("Item %s (%s): no sentences after normalisation; skipping.", item_id, lang)
            continue

        logger.info("Item %s (%s): inserting %s sentences.", item_id, lang, len(sentences))

        for idx, sent in enumerate(sentences):
            cur.execute(
                """
                INSERT OR IGNORE INTO corpus_sentences
                (item_id, sent_index, sentence, language, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    idx,
                    sent,
                    lang,
                    datetime.now().isoformat(),
                ),
            )

        processed += 1

    conn.commit()
    return processed


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        count = preprocess(conn)
    finally:
        conn.close()

    logger.info("Preprocessing complete for %s items.", count)


if __name__ == "__main__":
    main()
