#!/usr/bin/env python3
"""Evaluate corpus metadata and annotations using multi-AI outputs.

This script runs AFTER the professional cycle:
- Reads texts and their metadata from corpus_platform.db.
- Looks for corresponding multi-AI JSON files in research_exports/multi_ai.
- Computes simple, transparent metrics:
  - metadata completeness (how many key fields are filled)
  - annotation coverage (e.g. has_tokens, has_sentences, has_pos, etc.)
- Writes a CSV report to research_exports/evaluation/ with one row per text.

Later, you can add a GPT-5.1 (or other LLM) layer that reads the CSV and
JSONs and writes qualitative comments, but that is OPTIONAL and separate
from this core evaluation script.
"""

import csv
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
MULTI_AI_DIR = ROOT / "research_exports" / "multi_ai"
EVAL_DIR = ROOT / "research_exports" / "evaluation"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluate_metadata_and_annotations")


def load_multi_ai_json(item_id: int) -> Dict[str, Any]:
    path = MULTI_AI_DIR / f"item_{item_id}_multi_ai.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.warning("Could not read multi-AI JSON for %s: %s", item_id, exc)
        return {}


def metadata_completeness(row: sqlite3.Row) -> float:
    """Compute a simple completeness score over key metadata fields [0, 1]."""
    fields = [
        "title",
        "language",
        "genre",
        "period",
        "author",
        "translator",
        "original_language",
        "text_type",
        "diachronic_stage",
    ]
    filled = 0
    for f in fields:
        val = row[f]
        if val is not None and str(val).strip():
            filled += 1
    return filled / len(fields)


def annotation_coverage(ai_data: Dict[str, Any]) -> Dict[str, int]:
    """Return simple binary features indicating what kinds of signals exist."""
    if not ai_data:
        return {
            "has_multi_ai": 0,
            "has_tokens": 0,
            "has_sentences": 0,
            "has_pos": 0,
            "has_lemmas": 0,
            "has_dependencies": 0,
        }

    # heuristic checks for common structures
    tokens = ai_data.get("tokens") or ai_data.get("token_list") or []
    sentences = ai_data.get("sentences") or []
    pos_tags = ai_data.get("pos_tags") or []
    lemmas = ai_data.get("lemmas") or []
    deps = ai_data.get("dependencies") or ai_data.get("dep_edges") or []

    return {
        "has_multi_ai": 1,
        "has_tokens": 1 if tokens else 0,
        "has_sentences": 1 if sentences else 0,
        "has_pos": 1 if pos_tags else 0,
        "has_lemmas": 1 if lemmas else 0,
        "has_dependencies": 1 if deps else 0,
    }


def evaluate() -> Path:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    MULTI_AI_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, url, title, language, genre, period, author,
               translator, original_language, text_type, diachronic_stage,
               word_count, date_added, status, metadata_quality
        FROM corpus_items
        WHERE content IS NOT NULL AND LENGTH(content) > 500
        """
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"evaluation_report_{now}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "id",
                "url",
                "title",
                "language",
                "word_count",
                "status",
                "metadata_quality",
                "metadata_completeness",
                "has_multi_ai",
                "has_tokens",
                "has_sentences",
                "has_pos",
                "has_lemmas",
                "has_dependencies",
            ]
        )

        count = 0
        for row in cur.fetchall():
            item_id = row["id"]
            ai_data = load_multi_ai_json(item_id)
            meta_score = metadata_completeness(row)
            cov = annotation_coverage(ai_data)

            writer.writerow(
                [
                    item_id,
                    row["url"],
                    row["title"],
                    row["language"],
                    row["word_count"],
                    row["status"],
                    row["metadata_quality"],
                    f"{meta_score:.2f}",
                    cov["has_multi_ai"],
                    cov["has_tokens"],
                    cov["has_sentences"],
                    cov["has_pos"],
                    cov["has_lemmas"],
                    cov["has_dependencies"],
                ]
            )
            count += 1

    conn.close()
    logger.info("Wrote evaluation report for %s items to %s", count, out_path)
    return out_path


def main() -> None:
    logger.info("=== EVALUATION STAGE START ===")
    out_path = evaluate()
    logger.info("=== EVALUATION STAGE COMPLETE === %s", out_path)


if __name__ == "__main__":
    main()
