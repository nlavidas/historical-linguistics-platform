#!/usr/bin/env python3
"""Export corpus_platform.db texts for CorpusExplorer v2.0.

Creates a folder research_exports/corpusexplorer with:
  - metadata.csv  : one row per text with rich metadata
  - text_<id>.txt : plain UTF-8 text files

CorpusExplorer can then import these as a corpus using "plain text files
with metadata" (directory import).
"""

import csv
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EXPORT_DIR = ROOT / "research_exports" / "corpusexplorer"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("export_corpusexplorer")


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_corpus() -> None:
    ensure_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id, url, title, language, content, word_count, date_added, status,
            genre, period, author, translator, original_language, translation_year,
            is_retranslation, is_retelling, is_biblical, is_classical,
            text_type, diachronic_stage
        FROM corpus_items
        ORDER BY date_added
        """
    )

    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.warning("No rows found in corpus_items; nothing to export.")
        return

    metadata_path = EXPORT_DIR / "metadata.csv"
    fieldnames = [
        "id",
        "file",
        "url",
        "title",
        "author",
        "language",
        "genre",
        "period",
        "diachronic_stage",
        "original_language",
        "translator",
        "translation_year",
        "is_retranslation",
        "is_retelling",
        "is_biblical",
        "is_classical",
        "text_type",
        "word_count",
        "date_added",
        "status",
    ]

    with metadata_path.open("w", encoding="utf-8", newline="") as f_meta:
        writer = csv.DictWriter(f_meta, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            text_id = row["id"]
            filename = f"text_{text_id}.txt"
            text_path = EXPORT_DIR / filename

            content = row["content"] or ""
            text_path.write_text(content, encoding="utf-8")

            writer.writerow(
                {
                    "id": text_id,
                    "file": filename,
                    "url": row["url"] or "",
                    "title": row["title"] or "",
                    "author": row["author"] or "",
                    "language": row["language"] or "",
                    "genre": row["genre"] or "",
                    "period": row["period"] or "",
                    "diachronic_stage": row["diachronic_stage"] or "",
                    "original_language": row["original_language"] or "",
                    "translator": row["translator"] or "",
                    "translation_year": row["translation_year"] or "",
                    "is_retranslation": row["is_retranslation"] or 0,
                    "is_retelling": row["is_retelling"] or 0,
                    "is_biblical": row["is_biblical"] or 0,
                    "is_classical": row["is_classical"] or 0,
                    "text_type": row["text_type"] or "",
                    "word_count": row["word_count"] or 0,
                    "date_added": row["date_added"] or "",
                    "status": row["status"] or "",
                }
            )

    logger.info("Exported %s texts to %s", len(rows), EXPORT_DIR)


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        return

    logger.info("Starting export for CorpusExplorer at %s", datetime.now().isoformat())
    export_corpus()
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
