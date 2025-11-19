#!/usr/bin/env python3
"""Generate a detailed daily summary report for the researcher.

This script runs AFTER the nightly professional cycle and produces a
Markdown report with:
- core corpus statistics (by language, status, etc.),
- evaluation summary (metadata completeness, annotation coverage),
- export and multi-AI stats,
- a clear section for HUMAN FEEDBACK about priorities and fixes.

Output: research_exports/daily_reports/daily_report_YYYYMMDD_HHMMSS.md
"""

import csv
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EVAL_DIR = ROOT / "research_exports" / "evaluation"
MULTI_AI_DIR = ROOT / "research_exports" / "multi_ai"
EXPORT_CE_DIR = ROOT / "research_exports" / "corpusexplorer"
EXPORT_TXM_DIR = ROOT / "research_exports" / "txm"
HF_DIR = ROOT / "research_exports" / "hf_dataset"
DAILY_DIR = ROOT / "research_exports" / "daily_reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_daily_summary")


def latest_evaluation_csv() -> Optional[Path]:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(EVAL_DIR.glob("evaluation_report_*.csv"))
    return reports[-1] if reports else None


def load_evaluation(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def corpus_stats(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT language, status, COUNT(*)
        FROM corpus_items
        GROUP BY language, status
        """
    )
    by_lang_status = defaultdict(lambda: defaultdict(int))
    total = 0
    for language, status, count in cur.fetchall():
        by_lang_status[language or "UNKNOWN"][status or "UNKNOWN"] += count
        total += count

    cur.execute("SELECT COUNT(*) FROM corpus_items WHERE content IS NOT NULL")
    with_text = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM corpus_items")
    total_rows = cur.fetchone()[0]

    return {
        "by_lang_status": by_lang_status,
        "total_items": total_rows,
        "with_text": with_text,
    }


def multi_ai_stats():
    if not MULTI_AI_DIR.exists():
        return {"json_files": 0}
    json_files = list(MULTI_AI_DIR.glob("item_*_multi_ai.json"))
    return {"json_files": len(json_files)}


def export_stats():
    ce_files = list(EXPORT_CE_DIR.rglob("*")) if EXPORT_CE_DIR.exists() else []
    txm_files = list(EXPORT_TXM_DIR.rglob("*.xml")) if EXPORT_TXM_DIR.exists() else []
    hf_train = HF_DIR / "data" / "train.jsonl"

    return {
        "corpusexplorer_files": len(ce_files),
        "txm_tei_files": len(txm_files),
        "hf_dataset_exists": hf_train.exists(),
        "hf_train_path": str(hf_train) if hf_train.exists() else "",
    }


def build_report() -> Path:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    eval_path = latest_evaluation_csv()

    if eval_path is not None:
        eval_rows = load_evaluation(eval_path)
    else:
        eval_rows = []

    cstats = corpus_stats(conn)
    mstats = multi_ai_stats()
    estats = export_stats()

    # Evaluation aggregates
    total_eval = len(eval_rows)
    meta_scores = [float(r["metadata_completeness"]) for r in eval_rows] if eval_rows else []
    avg_meta = sum(meta_scores) / total_eval if total_eval else 0.0

    cov_counts = Counter()
    for r in eval_rows:
        for key in [
            "has_multi_ai",
            "has_tokens",
            "has_sentences",
            "has_pos",
            "has_lemmas",
            "has_dependencies",
        ]:
            if r.get(key) == "1":
                cov_counts[key] += 1

    low_meta_items = [r for r in eval_rows if float(r["metadata_completeness"]) < 0.6]
    missing_multi_ai = [r for r in eval_rows if r.get("has_multi_ai") == "0"]

    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = f"daily_report_{now.strftime('%Y%m%d_%H%M%S')}.md"
    out_path = DAILY_DIR / fname

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Daily Corpus Summary – {ts}\n\n")

        f.write("## 1. Corpus overview\n\n")
        f.write(f"- Total items in corpus: {cstats['total_items']}\n")
        f.write(f"- Items with non-empty content: {cstats['with_text']}\n\n")

        f.write("### By language and status\n\n")
        for lang, statuses in sorted(cstats["by_lang_status"].items()):
            f.write(f"- **{lang}**\n")
            for status, count in sorted(statuses.items()):
                f.write(f"  - {status}: {count}\n")
            f.write("\n")

        f.write("## 2. Evaluation summary (metadata + annotations)\n\n")
        if not eval_rows:
            f.write("No evaluation report found for this run.\n\n")
        else:
            f.write(f"- Evaluated items: {total_eval}\n")
            f.write(f"- Average metadata completeness: {avg_meta:.2f}\n")
            f.write(f"- Items with low metadata completeness (<0.60): {len(low_meta_items)}\n")
            f.write(f"- Items without multi-AI JSON: {len(missing_multi_ai)}\n\n")

            f.write("### Annotation coverage counts\n\n")
            for key in [
                "has_multi_ai",
                "has_tokens",
                "has_sentences",
                "has_pos",
                "has_lemmas",
                "has_dependencies",
            ]:
                f.write(f"- {key}: {cov_counts.get(key, 0)} items\n")
            f.write("\n")

            if low_meta_items:
                f.write("### Items with low metadata completeness (top 30)\n\n")
                for r in low_meta_items[:30]:
                    title = (r.get("title") or "").strip()
                    f.write(
                        f"- ID {r['id']}: {title} – metadata score {float(r['metadata_completeness']):.2f}\n"
                    )
                f.write("\n")

            if missing_multi_ai:
                f.write("### Items without multi-AI annotations (top 30)\n\n")
                for r in missing_multi_ai[:30]:
                    title = (r.get("title") or "").strip()
                    f.write(f"- ID {r['id']}: {title}\n")
                f.write("\n")

        f.write("## 3. Multi-AI and exports\n\n")
        f.write(f"- Multi-AI JSON files: {mstats['json_files']}\n")
        f.write(f"- CorpusExplorer export files: {estats['corpusexplorer_files']}\n")
        f.write(f"- TXM TEI XML files: {estats['txm_tei_files']}\n")
        f.write(
            f"- Hugging Face dataset present: {'yes' if estats['hf_dataset_exists'] else 'no'}\n"
        )
        if estats["hf_dataset_exists"]:
            f.write(f"  - train.jsonl path: `{estats['hf_train_path']}`\n")
        f.write("\n")

        if eval_path is not None:
            f.write("## 4. Raw artefacts for this run\n\n")
            f.write(f"- Evaluation CSV: `{eval_path}`\n")
        f.write(
            "- Multi-AI JSON directory: `research_exports/multi_ai/` (one file per item)\n"
        )
        f.write(
            "- Obsidian notes directory: `research_exports/obsidian_notes/` (overview + item notes)\n\n"
        )

        f.write("## 5. Researcher feedback for next cycles\n\n")
        f.write("Fill this section manually during the day. The system will not overwrite it.\n\n")
        f.write("### 5.1 Priorities for the next nights\n\n")
        f.write("- [ ] Focus more on languages / periods:\n")
        f.write("- [ ] Metadata fields to improve (author, period, diachronic_stage, etc.):\n")
        f.write("- [ ] Annotation aspects to improve (dependencies, lemmas, etc.):\n\n")

        f.write("### 5.2 Problems noticed\n\n")
        f.write("- \n- \n\n")

        f.write("### 5.3 Ideas and experiments\n\n")
        f.write("- \n- \n\n")

    conn.close()
    logger.info("Daily summary written to %s", out_path)
    return out_path


def main() -> None:
    logger.info("=== DAILY SUMMARY REPORT START ===")
    path = build_report()
    logger.info("=== DAILY SUMMARY REPORT COMPLETE === %s", path)


if __name__ == "__main__":
    main()
