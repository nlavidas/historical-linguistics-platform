#!/usr/bin/env python3
"""Generate a Windsurf-style HTML quality control dashboard.

This focuses on big, readable HTML (no external JS libraries, no AI branding):
- Overall corpus statistics from corpus_platform.db
- Coverage of processing layers from the latest evaluation_report_*.csv:
  - multi-source outputs, tokenization, sentence segmentation, POS, lemmata,
    dependencies
- A table of texts needing attention (weak metadata or annotation), with
  emphasis on Greek and Latin.

Output:
    research_exports/visual_reports/quality_control_dashboard_YYYYMMDD_HHMMSS.html

Run:
    python generate_quality_control_dashboard.py

This is complementary to generate_annotation_visualizations.py; it is meant to
feel like a simple control console for monitoring and manual correction.
"""

import csv
import logging
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EVAL_DIR = ROOT / "research_exports" / "evaluation"
VIS_DIR = ROOT / "research_exports" / "visual_reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_quality_control_dashboard")


def latest_evaluation_csv() -> Optional[Path]:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(EVAL_DIR.glob("evaluation_report_*.csv"))
    return reports[-1] if reports else None


def corpus_overview(conn: sqlite3.Connection) -> Dict:
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*) as total_texts,
               SUM(word_count) as total_words,
               AVG(metadata_quality) as avg_metadata,
               AVG(annotation_score) as avg_annotation
        FROM corpus_items
        """
    )
    total_texts, total_words, avg_meta, avg_annot = cur.fetchone()

    cur.execute(
        """
        SELECT language, COUNT(*), SUM(word_count)
        FROM corpus_items
        GROUP BY language
        ORDER BY COUNT(*) DESC
        """
    )
    by_lang = [
        {
            "language": row[0] or "unknown",
            "count": row[1] or 0,
            "words": row[2] or 0,
        }
        for row in cur.fetchall()
    ]

    return {
        "total_texts": total_texts or 0,
        "total_words": total_words or 0,
        "avg_metadata": avg_meta or 0.0,
        "avg_annotation": avg_annot or 0.0,
        "by_language": by_lang,
    }


def treebank_quality_summary(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT language, COALESCE(treebank_quality, 'none') as tq, COUNT(*)
        FROM corpus_items
        GROUP BY language, COALESCE(treebank_quality, 'none')
        """
    )
    data: Dict[str, Dict[str, int]] = {}
    for lang, tq, cnt in cur.fetchall():
        lang_key = lang or "unknown"
        if lang_key not in data:
            data[lang_key] = {}
        data[lang_key][tq] = cnt or 0
    return data


def texts_needing_attention(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return texts with poor metadata or annotation, prioritising Greek/Latin."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, language, word_count, metadata_quality, annotation_score,
               status, diachronic_stage
        FROM corpus_items
        WHERE
            (metadata_quality IS NOT NULL AND metadata_quality < 80)
         OR (language IN ('grc', 'lat') AND (annotation_score IS NULL OR annotation_score < 60))
        ORDER BY
            CASE COALESCE(language, '') WHEN 'grc' THEN 0 WHEN 'lat' THEN 1 ELSE 2 END,
            COALESCE(annotation_score, 0) ASC,
            COALESCE(metadata_quality, 0) ASC
        LIMIT 40
        """
    )
    return cur.fetchall()


def evaluation_summary(path: Optional[Path]) -> Dict:
    if path is None or not path.exists():
        return {
            "has_eval": False,
            "total_eval": 0,
            "avg_meta_completeness": 0.0,
            "coverage": {},
        }

    counts = Counter()
    meta_scores: List[float] = []
    total = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                meta_scores.append(float(row.get("metadata_completeness", "0") or 0.0))
            except ValueError:
                meta_scores.append(0.0)

            for key in [
                "has_multi_ai",
                "has_tokens",
                "has_sentences",
                "has_pos",
                "has_lemmas",
                "has_dependencies",
            ]:
                if row.get(key) == "1":
                    counts[key] += 1

    avg_meta = sum(meta_scores) / total if total else 0.0

    coverage: Dict[str, Dict[str, float]] = {}
    for key, c in counts.items():
        pct = (c / total) * 100.0 if total else 0.0
        coverage[key] = {"count": c, "percent": pct}

    return {
        "has_eval": True,
        "total_eval": total,
        "avg_meta_completeness": avg_meta,
        "coverage": coverage,
    }


def format_percent(x: float) -> str:
    return f"{x:.1f}%" if x > 0 else "0.0%"


def contrastive_clusters(conn: sqlite3.Connection) -> List[Tuple[str, str, str, int, int, int]]:
    """Aggregate contrastive clusters based on original_language and diachronic_stage.

    This mirrors the logic in the diachronic research agent: we group texts by
    their original language, target language and diachronic stage, and count
    how many are retranslations or retellings. This gives a compact overview
    of language contact chains such as Greek -> English across periods, or
    Latin Boethius -> Middle/Modern English.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(original_language, '') AS orig_lang,
            COALESCE(language, '')         AS lang,
            COALESCE(diachronic_stage, '') AS stage,
            COUNT(*)                       AS cnt,
            SUM(CASE WHEN is_retranslation THEN 1 ELSE 0 END) AS retrans,
            SUM(CASE WHEN is_retelling THEN 1 ELSE 0 END)     AS retell
        FROM corpus_items
        WHERE original_language IS NOT NULL AND original_language != ''
        GROUP BY orig_lang, lang, stage
        ORDER BY orig_lang, stage, lang
        """
    )
    return cur.fetchall()


def valency_by_stage(conn: sqlite3.Connection) -> List[Tuple[str, str, str, int]]:
    """Summarise valency pattern coverage by language and diachronic stage.

    Uses the valency_patterns_count field populated by collectors that extract
    PROIEL valency patterns. This gives a rough sense of how rich the
    Diachronic Contrastive Valency Lexicon is for each language/stage.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(language, '')         AS lang,
            COALESCE(diachronic_stage, '') AS stage,
            COUNT(*)                       AS texts,
            SUM(valency_patterns_count)    AS patterns
        FROM corpus_items
        WHERE valency_patterns_count IS NOT NULL AND valency_patterns_count > 0
        GROUP BY lang, stage
        ORDER BY lang, stage
        """
    )
    return cur.fetchall()


def build_html(
    overview: Dict,
    tb_summary: Dict[str, Dict[str, int]],
    eval_summary: Dict,
    attention: List[sqlite3.Row],
    contrastive: List[Tuple[str, str, str, int, int, int]],
    valency_rows: List[Tuple[str, str, str, int]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cov = eval_summary.get("coverage", {})
    total_eval = eval_summary.get("total_eval", 0)

    def cov_pct(key: str) -> str:
        info = cov.get(key)
        if not info:
            return "–"
        return format_percent(float(info.get("percent", 0.0)))

    qc_sent = cov_pct("has_sentences")
    qc_pos = cov_pct("has_pos")
    qc_lemmas = cov_pct("has_lemmas")
    qc_deps = cov_pct("has_dependencies")

    if eval_summary.get("has_eval") and total_eval:
        qc_note = (
            f"Based on {total_eval} evaluated texts. "
            "Targets: aim for 100% sentence segmentation, POS, lemmata and dependencies "
            "on Greek and Latin core texts."
        )
    else:
        qc_note = (
            "No evaluation report yet. Run the full professional cycle or "
            "evaluate_metadata_and_annotations.py."
        )

    by_lang_rows = []
    for row in overview.get("by_language", []):
        by_lang_rows.append(
            f"<tr><td>{row['language']}</td><td>{row['count']}</td>"
            f"<td>{row['words']:,}</td></tr>"
        )

    tb_rows = []
    for lang in sorted(tb_summary.keys()):
        info = tb_summary[lang]
        tb_rows.append(
            "<tr>"
            f"<td>{lang}</td>"
            f"<td>{info.get('excellent', 0)}</td>"
            f"<td>{info.get('good', 0)}</td>"
            f"<td>{info.get('partial', 0)}</td>"
            f"<td>{info.get('none', 0)}</td>"
            "</tr>"
        )

    att_rows = []
    for r in attention:
        title = (r["title"] or "").strip()
        if len(title) > 60:
            title = title[:57] + "…"
        att_rows.append(
            "<tr>"
            f"<td>{r['id']}</td>"
            f"<td>{r['language'] or ''}</td>"
            f"<td>{title}</td>"
            f"<td style='text-align:right'>{r['word_count'] or 0:,}</td>"
            f"<td style='text-align:right'>{(r['metadata_quality'] or 0):.1f}%</td>"
            f"<td style='text-align:right'>{(r['annotation_score'] or 0):.1f}%</td>"
            f"<td>{r['diachronic_stage'] or ''}</td>"
            "</tr>"
        )

    contrastive_rows = []
    for orig_lang, lang, stage, cnt, retrans, retell in contrastive:
        stage_display = stage or "(unspecified)"
        contrastive_rows.append(
            "<tr>"
            f"<td>{orig_lang or '(?)'}</td>"
            f"<td>{lang or '(?)'}</td>"
            f"<td>{stage_display}</td>"
            f"<td style='text-align:right'>{cnt}</td>"
            f"<td style='text-align:right'>{retrans or 0}</td>"
            f"<td style='text-align:right'>{retell or 0}</td>"
            "</tr>"
        )

    valency_table_rows = []
    for lang, stage, texts, patterns in valency_rows:
        stage_display = stage or "(unspecified)"
        valency_table_rows.append(
            "<tr>"
            f"<td>{lang or '(?)'}</td>"
            f"<td>{stage_display}</td>"
            f"<td style='text-align:right'>{texts}</td>"
            f"<td style='text-align:right'>{patterns or 0}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Diachronic Corpus – Quality Control Console</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #e5e7eb;
      font-size: 18px;
      line-height: 1.6;
    }}
    h1 {{
      font-size: 2.2rem;
      margin-bottom: 0.25rem;
    }}
    .subtitle {{
      font-size: 0.95rem;
      color: #9ca3af;
      margin-bottom: 1.5rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
      align-items: flex-start;
    }}
    .card {{
      background: #020617;
      border-radius: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.6);
      padding: 18px 20px 20px;
      border: 1px solid #1f2937;
    }}
    .card h2 {{
      font-size: 1.3rem;
      margin: 0 0 0.75rem;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      font-size: 1rem;
      margin: 3px 0;
    }}
    .stat-label {{ color: #9ca3af; }}
    .stat-value {{ font-weight: 600; }}
    .big-number {{
      font-size: 2.0rem;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
      margin-top: 0.5rem;
    }}
    th, td {{
      padding: 4px 6px;
      border-bottom: 1px solid #1f2937;
    }}
    th {{
      text-align: left;
      color: #9ca3af;
      font-weight: 500;
    }}
    .qc-note {{
      margin-top: 8px;
      font-size: 0.9rem;
      color: #9ca3af;
    }}
    .footer {{
      margin-top: 24px;
      font-size: 0.85rem;
      color: #6b7280;
    }}
    .badge-ok {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #16a34a33;
      color: #4ade80;
      font-size: 0.8rem;
    }}
    .badge-warn {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #f9731633;
      color: #fed7aa;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
  <h1>Diachronic Corpus – Quality Control Console</h1>
  <div class="subtitle">Snapshot generated on {now}. Use this view to supervise metadata and treebank quality, layer by layer.</div>

  <div class="grid">
    <div class="card">
      <h2>Corpus overview</h2>
      <div class="stat-row"><span class="stat-label">Total texts</span><span class="stat-value">{overview['total_texts']:,}</span></div>
      <div class="stat-row"><span class="stat-label">Total words</span><span class="stat-value">{overview['total_words']:,}</span></div>
      <div class="stat-row"><span class="stat-label">Avg. metadata quality</span><span class="stat-value">{overview['avg_metadata']:.1f}%</span></div>
      <div class="stat-row"><span class="stat-label">Avg. annotation score</span><span class="stat-value">{overview['avg_annotation']:.1f}%</span></div>
    </div>

    <div class="card">
      <h2>Processing pipeline coverage</h2>
      <div class="stat-row"><span class="stat-label">Sentence segmentation</span><span class="stat-value big-number">{qc_sent}</span></div>
      <div class="stat-row"><span class="stat-label">POS tagging</span><span class="stat-value big-number">{qc_pos}</span></div>
      <div class="stat-row"><span class="stat-label">Lemmatization</span><span class="stat-value big-number">{qc_lemmas}</span></div>
      <div class="stat-row"><span class="stat-label">Dependency parsing</span><span class="stat-value big-number">{qc_deps}</span></div>
      <p class="qc-note">{qc_note}</p>
    </div>

    <div class="card">
      <h2>Texts by language</h2>
      <table>
        <thead><tr><th>Language</th><th>Texts</th><th>Words</th></tr></thead>
        <tbody>
          {''.join(by_lang_rows)}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Treebank quality by language</h2>
      <table>
        <thead>
          <tr>
            <th>Lang</th>
            <th>Excellent</th>
            <th>Good</th>
            <th>Partial</th>
            <th>None</th>
          </tr>
        </thead>
        <tbody>
          {''.join(tb_rows)}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Valency coverage by diachronic stage</h2>
      <p class="qc-note">Texts and total valency patterns per language and diachronic stage, based on PROIEL-derived valency patterns. Use this to see where the Diachronic Contrastive Valency Lexicon is already rich and where more data are needed.</p>
      <table>
        <thead>
          <tr>
            <th>Language</th>
            <th>Diachronic stage</th>
            <th>Texts</th>
            <th>Total valency patterns</th>
          </tr>
        </thead>
        <tbody>
          {''.join(valency_table_rows) or '<tr><td colspan="4">No valency patterns stored yet.</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Contrastive clusters (original → target)</h2>
      <p class="qc-note">Aggregated by original language, target language and diachronic stage. Use this to spot chains such as Greek 
      Homer/NT → Early Modern English → Modern English, or Latin Boethius → Middle/Modern English.</p>
      <table>
        <thead>
          <tr>
            <th>Origin</th>
            <th>Target</th>
            <th>Diachronic stage</th>
            <th>Texts</th>
            <th>Retranslations</th>
            <th>Retellings</th>
          </tr>
        </thead>
        <tbody>
          {''.join(contrastive_rows) or '<tr><td colspan="6">No contrastive clusters yet (original_language not set).</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Texts needing attention</h2>
      <p class="qc-note">Low metadata quality or weak treebank; Greek and Latin are prioritised to appear first.</p>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Lang</th>
            <th>Title</th>
            <th>Words</th>
            <th>Meta %</th>
            <th>Annot %</th>
            <th>Diachronic stage</th>
          </tr>
        </thead>
        <tbody>
          {''.join(att_rows) or '<tr><td colspan="7">No texts currently match the attention criteria.</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    This console is generated from corpus_platform.db and the latest evaluation report
    (if available). It is designed for manual supervision and research decisions,
    not for any one specific AI provider.
  </div>
</body>
</html>
"""


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        overview = corpus_overview(conn)
        tb_summary = treebank_quality_summary(conn)
        attention = texts_needing_attention(conn)
        contrastive = contrastive_clusters(conn)
        valency_rows = valency_by_stage(conn)
    finally:
        conn.close()

    eval_path = latest_evaluation_csv()
    eval_info = evaluation_summary(eval_path)

    html = build_html(overview, tb_summary, eval_info, attention, contrastive, valency_rows)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = VIS_DIR / f"quality_control_dashboard_{now}.html"
    out_path.write_text(html, encoding="utf-8")

    logger.info("Quality control dashboard written to %s", out_path)


if __name__ == "__main__":
    main()
