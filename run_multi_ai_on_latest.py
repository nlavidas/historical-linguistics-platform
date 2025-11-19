#!/usr/bin/env python3
"""Run Multi-AI ensemble annotation on latest corpus texts.

- Selects the N most recent texts from corpus_platform.db (default: 50).
- For each text, runs MultiAIAnnotator.annotate_ensemble(...).
- Saves one JSON file per text in research_exports/multi_ai/.
- Skips texts that already have a JSON file.

This stage uses only open-source / community models that are installed
locally (Stanza, spaCy, NLTK, Transformers, Ollama, etc.). Missing
libraries are handled gracefully by MultiAIAnnotator.
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from multi_ai_annotator import MultiAIAnnotator

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EXPORT_DIR = ROOT / "research_exports" / "multi_ai"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_multi_ai_on_latest")


def get_latest_texts(limit: int = 50) -> List[Tuple[int, str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, language, content
        FROM corpus_items
        WHERE content IS NOT NULL AND LENGTH(content) > 500
        ORDER BY date_added DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def run_multi_ai(limit: int = 50) -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    texts = get_latest_texts(limit=limit)
    if not texts:
        logger.info("No texts found in corpus_items; nothing to annotate.")
        return

    annotator = MultiAIAnnotator()

    total = len(texts)
    logger.info("Starting multi-AI ensemble annotation on %s latest texts.", total)

    for idx, (item_id, lang, content) in enumerate(texts, start=1):
        out_path = EXPORT_DIR / f"item_{item_id}_multi_ai.json"
        if out_path.exists():
            logger.info("[%s/%s] Skipping ID %s (already annotated).", idx, total, item_id)
            continue

        logger.info("[%s/%s] Annotating ID %s (lang=%s)", idx, total, item_id, lang)
        try:
            result = annotator.annotate_ensemble(content or "", language=lang or "en")
            annotator.save_annotations(result, str(out_path))
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("Multi-AI annotation failed for ID %s: %s", item_id, exc)

    logger.info("Multi-AI ensemble annotation stage complete.")


def main() -> None:
    logger.info("=== MULTI-AI ENSEMBLE STAGE START === %s", datetime.now().isoformat())
    run_multi_ai(limit=50)
    logger.info("=== MULTI-AI ENSEMBLE STAGE END === %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
