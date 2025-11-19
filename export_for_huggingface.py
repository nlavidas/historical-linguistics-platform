#!/usr/bin/env python3
"""Export corpus to a local Hugging Face-style dataset (private by default).

This script:
- Reads texts and metadata from corpus_platform.db.
- Writes a JSONL file under research_exports/hf_dataset/data/train.jsonl.
- Writes a minimal README.md describing the dataset.

You can later create a *private* dataset on the Hugging Face Hub and upload
this folder manually using `huggingface-cli` (not required for local use).
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
HF_DIR = ROOT / "research_exports" / "hf_dataset"
DATA_DIR = HF_DIR / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("export_for_huggingface")


def export_dataset() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, url, title, language, content, word_count, date_added,
               status, genre, period, author, translator, original_language,
               translation_year, is_retranslation, is_retelling,
               is_biblical, is_classical, text_type, diachronic_stage,
               metadata_quality
        FROM corpus_items
        WHERE content IS NOT NULL AND LENGTH(content) > 0
        """
    )

    out_path = DATA_DIR / "train.jsonl"
    count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for (
            item_id,
            url,
            title,
            language,
            content,
            word_count,
            date_added,
            status,
            genre,
            period,
            author,
            translator,
            original_language,
            translation_year,
            is_retranslation,
            is_retelling,
            is_biblical,
            is_classical,
            text_type,
            diachronic_stage,
            metadata_quality,
        ) in cur.fetchall():
            record = {
                "id": item_id,
                "url": url,
                "title": title,
                "text": content,
                "language": language,
                "word_count": word_count,
                "date_added": date_added,
                "status": status,
                "genre": genre,
                "period": period,
                "author": author,
                "translator": translator,
                "original_language": original_language,
                "translation_year": translation_year,
                "is_retranslation": bool(is_retranslation) if is_retranslation is not None else None,
                "is_retelling": bool(is_retelling) if is_retelling is not None else None,
                "is_biblical": bool(is_biblical) if is_biblical is not None else None,
                "is_classical": bool(is_classical) if is_classical is not None else None,
                "text_type": text_type,
                "diachronic_stage": diachronic_stage,
                "metadata_quality": metadata_quality,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    conn.close()

    logger.info("Wrote %s examples to %s", count, out_path)

    readme_path = HF_DIR / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Diachronic Multilingual Corpus (Local HF Dataset)\n\n"
            "This folder contains a locally exported Hugging Face-style dataset\n"
            "generated from `corpus_platform.db`. It is intended for **private**\n"
            "use on the external drive Z:. If you choose to upload it to the\n"
            "Hugging Face Hub, configure the dataset as **private**.\n\n"
            "Files:\n\n"
            "- `data/train.jsonl` — texts with metadata.\n\n",
            encoding="utf-8",
        )
        logger.info("Created README at %s", readme_path)


def main() -> None:
    logger.info("=== HUGGING FACE EXPORT START === %s", datetime.now().isoformat())
    export_dataset()
    logger.info("=== HUGGING FACE EXPORT COMPLETE === %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
