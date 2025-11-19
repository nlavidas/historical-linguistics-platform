#!/usr/bin/env python3
"""Run a full professional research cycle end-to-end.

Steps:
1. Optionally import any external corpus exports found in
   research_exports/incoming/*.json or *.csv into corpus_platform.db.
2. Start the autonomous night operation supervisor, which manages:
   - professional_metadata_dashboard.py
   - autonomous_247_collection.py
   - annotation_worker_247.py
   This runs until 08:00 Greece time.
3. After the night supervisor finishes, automatically export the
   enriched corpus to:
   - CorpusExplorer format (research_exports/corpusexplorer)
   - TXM TEI format (research_exports/txm)
4. Optionally run a multi-AI ensemble annotation stage over the
   latest texts (run_multi_ai_on_latest.py), which uses open-source
   community models (Stanza, spaCy, Transformers, NLTK, Ollama, etc.)
5. Run automatic evaluation of metadata and annotation coverage.
6. Export a local Hugging Face-style dataset for private use on Z:.
7. Export Obsidian-friendly Markdown notes on Z:.
8. Generate a detailed daily summary report with a feedback section.

Run:
    python run_professional_cycle.py

You can also schedule this script via Windows Task Scheduler if you
want it to start automatically in the evening.
"""

import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
INCOMING_DIR = ROOT / "research_exports" / "incoming"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_professional_cycle")


def run_import_stage() -> None:
    """Import any JSON/CSV exports found in the incoming folder."""
    if not INCOMING_DIR.exists():
        logger.info("No incoming folder found (%s); skipping import stage.", INCOMING_DIR)
        return

    json_files = sorted(INCOMING_DIR.glob("*.json"))
    csv_files = sorted(INCOMING_DIR.glob("*.csv"))

    if not json_files and not csv_files:
        logger.info("No JSON/CSV files in %s; nothing to import.", INCOMING_DIR)
        return

    logger.info("Import stage: found %s JSON and %s CSV files.", len(json_files), len(csv_files))

    for path in json_files:
        logger.info("Importing JSON: %s", path)
        result = subprocess.run(
            [sys.executable, "import_external_corpus.py", "--input", str(path), "--format", "json"],
            cwd=ROOT,
        )
        if result.returncode != 0:
            logger.warning("Import failed for %s (return code %s)", path, result.returncode)

    for path in csv_files:
        logger.info("Importing CSV: %s", path)
        result = subprocess.run(
            [sys.executable, "import_external_corpus.py", "--input", str(path), "--format", "csv"],
            cwd=ROOT,
        )
        if result.returncode != 0:
            logger.warning("Import failed for %s (return code %s)", path, result.returncode)


def run_night_supervisor() -> None:
    """Run the autonomous night operation until it finishes."""
    logger.info("Starting autonomous night operation supervisor...")
    result = subprocess.run([sys.executable, "autonomous_night_operation.py"], cwd=ROOT)
    logger.info("Night supervisor finished with return code %s", result.returncode)


def run_exports() -> None:
    """Export corpus to CorpusExplorer and TXM formats."""
    logger.info("Starting export for CorpusExplorer v2.0...")
    result_ce = subprocess.run(
        [sys.executable, "export_for_corpusexplorer.py"],
        cwd=ROOT,
    )
    logger.info("CorpusExplorer export finished with return code %s", result_ce.returncode)

    logger.info("Starting export for TXM...")
    result_txm = subprocess.run(
        [sys.executable, "export_for_txm.py"],
        cwd=ROOT,
    )
    logger.info("TXM export finished with return code %s", result_txm.returncode)


def run_multi_ai_stage() -> None:
    """Run multi-AI ensemble annotation over latest texts (optional stage)."""
    logger.info("Starting multi-AI ensemble annotation on latest texts...")
    result_multi = subprocess.run(
        [sys.executable, "run_multi_ai_on_latest.py"],
        cwd=ROOT,
    )
    logger.info("Multi-AI ensemble stage finished with return code %s", result_multi.returncode)


def run_evaluation_stage() -> None:
    """Run automatic evaluation of metadata and annotations."""
    logger.info("Starting evaluation stage (metadata + multi-AI coverage)...")
    result_eval = subprocess.run(
        [sys.executable, "evaluate_metadata_and_annotations.py"],
        cwd=ROOT,
    )
    logger.info("Evaluation stage finished with return code %s", result_eval.returncode)


def run_hf_export_stage() -> None:
    """Export a local Hugging Face-style dataset for private use."""
    logger.info("Starting Hugging Face dataset export stage...")
    result_hf = subprocess.run(
        [sys.executable, "export_for_huggingface.py"],
        cwd=ROOT,
    )
    logger.info("Hugging Face export stage finished with return code %s", result_hf.returncode)


def run_obsidian_export_stage() -> None:
    """Export Markdown notes for Obsidian on drive Z:."""
    logger.info("Starting Obsidian notes export stage...")
    result_obs = subprocess.run(
        [sys.executable, "export_for_obsidian.py"],
        cwd=ROOT,
    )
    logger.info("Obsidian export stage finished with return code %s", result_obs.returncode)


def run_daily_summary_stage() -> None:
    """Generate a detailed daily summary report with feedback section."""
    logger.info("Starting daily summary report stage...")
    result_daily = subprocess.run(
        [sys.executable, "generate_daily_summary.py"],
        cwd=ROOT,
    )
    logger.info("Daily summary report stage finished with return code %s", result_daily.returncode)


def main() -> None:
    logger.info("=== PROFESSIONAL RESEARCH CYCLE START === %s", datetime.now().isoformat())

    # 1. Import any external exports
    run_import_stage()

    # 2. Run supervised night pipeline
    run_night_supervisor()

    # 3. Export for external tools
    run_exports()

    # 4. Multi-AI ensemble annotation stage (open-source community models)
    run_multi_ai_stage()

    # 5. Automatic evaluation of metadata and annotation coverage
    run_evaluation_stage()

    # 6. Local Hugging Face-style dataset export (private on Z:)
    run_hf_export_stage()

    # 7. Obsidian Markdown notes export on Z:
    run_obsidian_export_stage()

    # 8. Detailed daily summary report with feedback section
    run_daily_summary_stage()

    logger.info("=== PROFESSIONAL RESEARCH CYCLE COMPLETE === %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
