#!/usr/bin/env python3
"""24/7 Local Annotation Worker

Continuously scans corpus_platform.db for texts that have not yet been
annotated with PROIEL/Stanza and enriches them with treebanks and
annotation statistics.

Designed for local use (Z: drive) using existing athdgc PROIEL tools.
"""

import os
import sys
import time
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure local package imports work
sys.path.insert(0, str(Path(__file__).parent))

# Stanza resources
os.environ.setdefault("STANZA_RESOURCES_DIR", str(Path("Z:/models/stanza")))

try:
    import stanza
except ImportError as e:  # pragma: no cover - runtime environment detail
    print(f"Missing dependency: {e}")
    print("Run: pip install stanza")
    sys.exit(1)

from athdgc.proiel_processor import PROIELProcessor
from athdgc.valency_lexicon import ValencyLexicon

LOG_FILE = Path(__file__).parent / "annotation_worker.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("annotation_worker")

SUPPORTED_LANGS = {"grc", "lat"}


class AnnotationWorker:
    """24/7 annotator operating on existing corpus items."""

    def __init__(self, batch_size: int = 1, idle_sleep: int = 900) -> None:
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.batch_size = batch_size
        self.idle_sleep = idle_sleep

        self.proiel_processor = PROIELProcessor()
        self.valency_lexicon = ValencyLexicon()
        self.pipelines: Dict[str, Any] = {}

        logger.info("=" * 80)
        logger.info("ANNOTATION WORKER INITIALIZED")
        logger.info("Database: %s", self.db_path)
        logger.info("Supported languages: %s", ", ".join(sorted(SUPPORTED_LANGS)))
        logger.info("=" * 80)

        self._ensure_schema()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensure corpus_items has the columns needed for annotation metrics."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Base table (if created only by DIACHRONIC_MULTILINGUAL_COLLECTOR)
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
                    status TEXT,
                    metadata_quality REAL DEFAULT 0
                )
                """
            )

            # Add extended annotation columns when missing
            existing_cols = {
                row[1] for row in cur.execute("PRAGMA table_info(corpus_items)")
            }

            extra_cols = [
                ("proiel_xml", "TEXT"),
                ("annotation_score", "REAL DEFAULT 0"),
                ("tokens_count", "INTEGER DEFAULT 0"),
                ("lemmas_count", "INTEGER DEFAULT 0"),
                ("pos_tags_count", "INTEGER DEFAULT 0"),
                ("dependencies_count", "INTEGER DEFAULT 0"),
                ("has_treebank", "BOOLEAN DEFAULT 0"),
                ("treebank_format", "TEXT"),
                ("annotation_date", "TEXT"),
                ("annotation_quality", "REAL DEFAULT 0"),
                ("treebank_quality", "TEXT"),
                ("valency_patterns_count", "INTEGER DEFAULT 0"),
            ]

            for name, col_type in extra_cols:
                if name not in existing_cols:
                    cur.execute(f"ALTER TABLE corpus_items ADD COLUMN {name} {col_type}")

            # Backfill treebank_quality for any existing rows using simple thresholds
            # on tokens_count and annotation_score so older annotations are classified.
            cur.execute(
                """
                UPDATE corpus_items
                SET treebank_quality = CASE
                    WHEN tokens_count >= 10000 AND annotation_score >= 90 THEN 'excellent'
                    WHEN tokens_count >= 2000 AND annotation_score >= 75 THEN 'good'
                    WHEN tokens_count > 0 AND annotation_score > 0 THEN 'partial'
                    ELSE 'none'
                END
                WHERE treebank_quality IS NULL OR treebank_quality = ''
                """
            )

            conn.commit()
            conn.close()
            logger.info("Schema check complete; annotation columns ensured.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Schema setup failed: %s", exc)

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------
    def get_pipeline(self, language: str) -> Optional[stanza.Pipeline]:
        if language not in SUPPORTED_LANGS:
            logger.warning("Language %s not supported for annotation", language)
            return None

        if language not in self.pipelines:
            try:
                logger.info("Loading Stanza pipeline for %s...", language)
                self.pipelines[language] = stanza.Pipeline(
                    language,
                    processors="tokenize,pos,lemma,depparse",
                    verbose=False,
                )
                logger.info("Pipeline for %s loaded.", language)
            except Exception as exc:  # pragma: no cover - runtime
                logger.error("Failed to load pipeline for %s: %s", language, exc)
                return None

        return self.pipelines[language]

    # ------------------------------------------------------------------
    # Core annotation logic
    # ------------------------------------------------------------------
    def fetch_pending_items(self) -> List[sqlite3.Row]:
        """Return a small batch of items that still need annotation."""
        conn = self._get_connection()
        cur = conn.cursor()

        placeholders = ",".join("?" for _ in SUPPORTED_LANGS)
        params: List[Any] = list(SUPPORTED_LANGS)
        params.append(self.batch_size)

        cur.execute(
            f"""
            SELECT id, url, title, language, content, word_count
            FROM corpus_items
            WHERE (proiel_xml IS NULL OR proiel_xml = '')
              AND language IN ({placeholders})
              AND content IS NOT NULL
              AND LENGTH(content) > 500
            ORDER BY word_count ASC, date_added ASC
            LIMIT ?
            """,
            params,
        )

        rows = cur.fetchall()
        conn.close()
        return rows

    def annotate_item(self, row: sqlite3.Row) -> None:
        """Annotate a single corpus item and write results back to DB."""
        item_id = row["id"]
        language = row["language"]
        title = row["title"] or "(untitled)"

        logger.info("Annotating ID %s - %s [%s]", item_id, title[:80], language)

        pipeline = self.get_pipeline(language)
        if not pipeline:
            logger.warning("Skipping ID %s: no pipeline for %s", item_id, language)
            return

        text = row["content"]
        if not text or len(text) < 500:
            logger.warning("Skipping ID %s: content too short", item_id)
            return

        try:
            # Run Stanza (allows PROIELProcessor to reuse analyses if desired)
            _ = pipeline(text)

            proiel_result = self.proiel_processor.annotate_proiel(text, language)
            if not proiel_result or "proiel_xml" not in proiel_result:
                logger.warning("No PROIEL XML produced for ID %s", item_id)
                return

            proiel_xml = proiel_result["proiel_xml"]
            stats = proiel_result.get("statistics", {})

            tokens = int(stats.get("tokens", 0) or 0)
            lemmas = int(stats.get("lemmas", 0) or 0)
            pos_tags = int(stats.get("pos_tags", 0) or 0)
            dependencies = int(stats.get("dependencies", 0) or 0)

            annotation_score = 0.0
            if tokens > 0:
                lemma_cov = (lemmas / tokens) * 100
                pos_cov = (pos_tags / tokens) * 100
                dep_cov = (dependencies / tokens) * 100
                annotation_score = (lemma_cov + pos_cov + dep_cov) / 3

            if tokens > 0 and annotation_score > 0:
                if tokens >= 10000 and annotation_score >= 90.0:
                    treebank_quality = "excellent"
                elif tokens >= 2000 and annotation_score >= 75.0:
                    treebank_quality = "good"
                else:
                    treebank_quality = "partial"
            else:
                treebank_quality = "none"

            # Simple metadata quality: title + language + word_count + date
            quality_checks = 0
            if title:
                quality_checks += 1
            if language:
                quality_checks += 1
            if (row["word_count"] or 0) > 0:
                quality_checks += 1
            quality_checks += 1  # date will be filled now
            metadata_quality = (quality_checks / 4) * 100

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE corpus_items
                SET proiel_xml = ?,
                    annotation_score = ?,
                    tokens_count = ?,
                    lemmas_count = ?,
                    pos_tags_count = ?,
                    dependencies_count = ?,
                    has_treebank = 1,
                    treebank_format = 'proiel',
                    annotation_date = ?,
                    annotation_quality = ?,
                    treebank_quality = ?,
                    status = COALESCE(status, 'completed')
                WHERE id = ?
                """,
                (
                    proiel_xml,
                    annotation_score,
                    tokens,
                    lemmas,
                    pos_tags,
                    dependencies,
                    datetime.now().isoformat(),
                    metadata_quality,
                    treebank_quality,
                    item_id,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                "✓ Annotated ID %s | tokens=%s, lemmas=%s, pos=%s, deps=%s, score=%.2f",
                item_id,
                tokens,
                lemmas,
                pos_tags,
                dependencies,
                annotation_score,
            )
        except Exception as exc:  # pragma: no cover - runtime
            logger.error("Annotation failed for ID %s: %s", item_id, exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        logger.info("Starting 24/7 annotation loop (batch_size=%s, idle_sleep=%ss)", self.batch_size, self.idle_sleep)

        while True:
            try:
                batch = self.fetch_pending_items()
                if not batch:
                    logger.info(
                        "No pending items for annotation. Sleeping %s seconds...",
                        self.idle_sleep,
                    )
                    time.sleep(self.idle_sleep)
                    continue

                logger.info("Found %s items pending annotation.", len(batch))
                for row in batch:
                    self.annotate_item(row)

            except Exception as exc:  # pragma: no cover - runtime
                logger.error("Unexpected error in main loop: %s", exc)
                time.sleep(60)


def main() -> None:
    worker = AnnotationWorker(batch_size=1, idle_sleep=900)
    worker.run_forever()


if __name__ == "__main__":
    main()
