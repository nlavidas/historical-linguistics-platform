#!/usr/bin/env python3
"""Export corpus_platform.db texts as minimal TEI XML for TXM.

Creates research_exports/txm with one TEI file per text.
TXM can import this directory as a TEI corpus.
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EXPORT_DIR = ROOT / "research_exports" / "txm"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("export_txm")


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def make_tei(row) -> str:
    lang = (row["language"] or "").strip() or "und"
    title = escape(row["title"] or "")
    author = escape(row["author"] or "")
    period = escape(row["period"] or "")
    genre = escape(row["genre"] or "")
    diachronic_stage = escape(row["diachronic_stage"] or "")
    content = escape(row["content"] or "")

    tei = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>{title}</title>
        {f'<author>{author}</author>' if author else ''}
      </titleStmt>
      <publicationStmt>
        <p>Automatically exported from HFRI Corpus Platform on {datetime.now().date()}</p>
      </publicationStmt>
      <sourceDesc>
        <p>Diachronic corpus item from corpus_platform.db</p>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <textClass>
        {f'<keywords type="period"><term>{period}</term></keywords>' if period else ''}
        {f'<keywords type="genre"><term>{genre}</term></keywords>' if genre else ''}
        {f'<keywords type="diachronic_stage"><term>{diachronic_stage}</term></keywords>' if diachronic_stage else ''}
      </textClass>
    </profileDesc>
  </teiHeader>
  <text xml:lang="{lang}">
    <body>
      <p>{content}</p>
    </body>
  </text>
</TEI>
"""
    return tei


def export_corpus() -> None:
    ensure_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id, url, title, author, language, genre, period,
            diachronic_stage, content
        FROM corpus_items
        ORDER BY date_added
        """
    )

    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.warning("No rows found in corpus_items; nothing to export.")
        return

    for row in rows:
        text_id = row["id"]
        filename = f"tei_{text_id}.xml"
        path = EXPORT_DIR / filename
        path.write_text(make_tei(row), encoding="utf-8")

    logger.info("Exported %s TEI files to %s", len(rows), EXPORT_DIR)


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        return

    logger.info("Starting export for TXM at %s", datetime.now().isoformat())
    export_corpus()
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
