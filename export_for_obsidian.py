#!/usr/bin/env python3
"""Export evaluation and item-level notes as Markdown for Obsidian.

This script:
- Reads the latest evaluation_report_*.csv from research_exports/evaluation.
- Uses metadata from corpus_platform.db.
- Writes:
  - An overview note summarising the run.
  - Per-item notes for each evaluated text.

All notes are written to research_exports/obsidian_notes/ on drive Z:.
You can point Obsidian to that folder as a vault or subfolder.
"""

import csv
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EVAL_DIR = ROOT / "research_exports" / "evaluation"
OBSIDIAN_DIR = ROOT / "research_exports" / "obsidian_notes"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("export_for_obsidian")


def load_latest_evaluation_csv() -> Path | None:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(EVAL_DIR.glob("evaluation_report_*.csv"))
    if not reports:
        return None
    return reports[-1]


def export_obsidian_notes() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)

    eval_path = load_latest_evaluation_csv()
    if not eval_path:
        logger.info("No evaluation reports found; skipping Obsidian export.")
        return

    logger.info("Using evaluation report: %s", eval_path)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Map id -> row from DB for richer metadata
    cur.execute(
        """
        SELECT id, url, title, language, genre, period, author, translator,
               original_language, text_type, diachronic_stage, word_count,
               date_added, status, metadata_quality
        FROM corpus_items
        WHERE content IS NOT NULL AND LENGTH(content) > 500
        """
    )
    items_by_id = {row["id"]: row for row in cur.fetchall()}

    # Read evaluation CSV
    with eval_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.info("Evaluation report is empty; nothing to export.")
        conn.close()
        return

    # Compute simple overview stats
    total = len(rows)
    meta_scores = [float(r["metadata_completeness"]) for r in rows]
    avg_meta = sum(meta_scores) / total if total else 0.0

    low_meta = [r for r in rows if float(r["metadata_completeness"]) < 0.6]
    no_multi_ai = [r for r in rows if r["has_multi_ai"] == "0"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overview_name = f"overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    overview_path = OBSIDIAN_DIR / overview_name

    with overview_path.open("w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"date: {timestamp}\n")
        f.write("type: overview\n")
        f.write("---\n\n")
        f.write("# Nightly Corpus Overview\n\n")
        f.write(f"- Total evaluated items: {total}\n")
        f.write(f"- Average metadata completeness: {avg_meta:.2f}\n")
        f.write(f"- Items with low metadata (<0.60): {len(low_meta)}\n")
        f.write(f"- Items missing multi-AI annotations: {len(no_multi_ai)}\n\n")

        if low_meta:
            f.write("## Items with low metadata completeness\n\n")
            for r in low_meta[:50]:  # cap listing to keep note readable
                f.write(f"- ID {r['id']}: {r.get('title', '').strip()} (score {float(r['metadata_completeness']):.2f})\n")
            f.write("\n")

        if no_multi_ai:
            f.write("## Items without multi-AI JSON\n\n")
            for r in no_multi_ai[:50]:
                f.write(f"- ID {r['id']}: {r.get('title', '').strip()}\n")
            f.write("\n")

    # Per-item notes
    for r in rows:
        item_id = int(r["id"])
        db_row = items_by_id.get(item_id)
        if db_row is None:
            continue

        note_path = OBSIDIAN_DIR / f"item_{item_id}.md"
        meta_score = float(r["metadata_completeness"])

        with note_path.open("w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"id: {item_id}\n")
            f.write(f"url: {db_row['url']}\n")
            f.write(f"title: {db_row['title']}\n")
            f.write(f"language: {db_row['language']}\n")
            f.write(f"period: {db_row['period']}\n")
            f.write(f"author: {db_row['author']}\n")
            f.write(f"diachronic_stage: {db_row['diachronic_stage']}\n")
            f.write(f"word_count: {db_row['word_count']}\n")
            f.write(f"status: {db_row['status']}\n")
            f.write(f"metadata_quality: {db_row['metadata_quality']}\n")
            f.write(f"metadata_completeness: {meta_score:.2f}\n")
            f.write(f"has_multi_ai: {r['has_multi_ai']}\n")
            f.write(f"has_tokens: {r['has_tokens']}\n")
            f.write(f"has_sentences: {r['has_sentences']}\n")
            f.write(f"has_pos: {r['has_pos']}\n")
            f.write(f"has_lemmas: {r['has_lemmas']}\n")
            f.write(f"has_dependencies: {r['has_dependencies']}\n")
            f.write("type: item_note\n")
            f.write("---\n\n")

            f.write(f"# {item_id} – {db_row['title']}\n\n")
            f.write(f"- URL: {db_row['url']}\n")
            f.write(f"- Language: {db_row['language']}\n")
            f.write(f"- Period: {db_row['period']}\n")
            f.write(f"- Author: {db_row['author']}\n")
            f.write(f"- Diachronic stage: {db_row['diachronic_stage']}\n")
            f.write(f"- Word count: {db_row['word_count']}\n")
            f.write(f"- Status: {db_row['status']}\n")
            f.write("\n")
            f.write("## Evaluation summary\n\n")
            f.write(f"- Metadata completeness: {meta_score:.2f}\n")
            f.write(f"- Has multi-AI JSON: {r['has_multi_ai']}\n")
            f.write(f"- Tokens: {r['has_tokens']}\n")
            f.write(f"- Sentences: {r['has_sentences']}\n")
            f.write(f"- POS tags: {r['has_pos']}\n")
            f.write(f"- Lemmas: {r['has_lemmas']}\n")
            f.write(f"- Dependencies: {r['has_dependencies']}\n")
            f.write("\n")
            f.write("Multi-AI JSON path (relative to corpus root): `research_exports/multi_ai/"
                    f"item_{item_id}_multi_ai.json`\n")

    conn.close()
    logger.info("Wrote overview and %s item notes to %s", len(rows), OBSIDIAN_DIR)


def main() -> None:
    logger.info("=== OBSIDIAN EXPORT START ===")
    export_obsidian_notes()
    logger.info("=== OBSIDIAN EXPORT COMPLETE ===")


if __name__ == "__main__":
    main()
