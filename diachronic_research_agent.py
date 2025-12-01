#!/usr/bin/env python3
"""Rule-based diachronic research agent.

Reads corpus_platform.db (and latest evaluation report, if present) and
produces a Markdown report with:
- Corpus and treebank overview (especially for Greek and Latin).
- Languages / periods / genres that are under-served.
- Concrete lists of texts (IDs) that need better treebanks or metadata.

Run:
    python diachronic_research_agent.py

The report is written to research_exports/agent_reports/.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EVAL_DIR = ROOT / "research_exports" / "evaluation"
AGENT_DIR = ROOT / "research_exports" / "agent_reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("diachronic_research_agent")

CORE_LANGS = ("grc", "lat")


def latest_evaluation_csv() -> Optional[Path]:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(EVAL_DIR.glob("evaluation_report_*.csv"))
    return reports[-1] if reports else None


def load_eval_summary(path: Path) -> Dict[str, float]:
    """Load a very light summary from evaluation CSV (if any)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lang_counts: Dict[str, int] = defaultdict(int)
            lang_has_multi_ai: Dict[str, int] = defaultdict(int)
            for row in reader:
                lang = row.get("language") or "unknown"
                lang_counts[lang] += 1
                if row.get("has_multi_ai") == "1":
                    lang_has_multi_ai[lang] += 1
        return {
            f"multi_ai_coverage_{lang}": (lang_has_multi_ai[lang] / lang_counts[lang])
            for lang in lang_counts
            if lang_counts[lang] > 0
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not summarise evaluation CSV %s: %s", path, exc)
        return {}


def corpus_overview(conn: sqlite3.Connection) -> Dict:
    conn.row_factory = sqlite3.Row
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


def treebank_quality_by_language(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
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


def greek_latin_gaps(conn: sqlite3.Connection) -> Tuple[List[sqlite3.Row], List[sqlite3.Row]]:
    """Return lists of Greek and Latin texts that need attention.

    We prioritise:
    - long texts (higher word_count)
    - with treebank_quality in ('none', 'partial')
    - or low metadata_quality
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, title, language, word_count, metadata_quality, annotation_score,
               COALESCE(treebank_quality, 'none') as treebank_quality
        FROM corpus_items
        WHERE language IN ('grc', 'lat')
        ORDER BY
            CASE COALESCE(treebank_quality, 'none')
                WHEN 'none' THEN 0
                WHEN 'partial' THEN 1
                WHEN 'good' THEN 2
                WHEN 'excellent' THEN 3
                ELSE 0
            END ASC,
            word_count DESC
        LIMIT 40
        """
    )
    rows = cur.fetchall()

    greek = [r for r in rows if r["language"] == "grc"]
    latin = [r for r in rows if r["language"] == "lat"]
    return greek, latin


def underrepresented_periods(conn: sqlite3.Connection) -> List[Tuple[str, str, int]]:
    """Find diachronic stages with relatively few texts for core languages."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT language, diachronic_stage, COUNT(*) as cnt
        FROM corpus_items
        WHERE language IN ('grc', 'lat')
          AND diachronic_stage IS NOT NULL AND diachronic_stage != ''
        GROUP BY language, diachronic_stage
        ORDER BY language, cnt ASC
        """
    )
    rows = cur.fetchall()

    # For each language, keep the 3 stages with the fewest texts
    by_lang: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for lang, stage, cnt in rows:
        by_lang[lang].append((stage, cnt))

    suggestions: List[Tuple[str, str, int]] = []
    for lang, items in by_lang.items():
        for stage, cnt in items[:3]:
            suggestions.append((lang, stage, cnt))
    return suggestions


def contrastive_opportunities(conn: sqlite3.Connection) -> List[Tuple[str, str, str, int, int, int]]:
    """Return aggregated contrastive opportunities across languages and stages.

    We look for texts where original_language is set (e.g. Greek or Latin
    originals) and group their translations, retranslations and retellings
    by target language and diachronic stage. This highlights language contact
    and diachronic contrastive clusters (e.g. Koine Greek vs Early Modern vs
    Modern English).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(original_language, '') AS orig_lang,
            COALESCE(language, '') AS lang,
            COALESCE(diachronic_stage, '') AS stage,
            COUNT(*) AS cnt,
            SUM(CASE WHEN is_retranslation THEN 1 ELSE 0 END) AS retrans,
            SUM(CASE WHEN is_retelling THEN 1 ELSE 0 END) AS retell
        FROM corpus_items
        WHERE original_language IS NOT NULL AND original_language != ''
        GROUP BY orig_lang, lang, stage
        ORDER BY orig_lang, stage, lang
        """
    )
    rows = cur.fetchall()
    return rows


def format_treebank_summary(tb: Dict[str, Dict[str, int]]) -> str:
    lines = ["Language | Excellent | Good | Partial | None", "-------- | --------- | ---- | ------- | ----"]
    for lang in sorted(tb.keys()):
        row = tb[lang]
        lines.append(
            f"{lang} | {row.get('excellent', 0)} | {row.get('good', 0)} | "
            f"{row.get('partial', 0)} | {row.get('none', 0)}"
        )
    return "\n".join(lines)


def ml_source_suggestions(conn: sqlite3.Connection) -> List[str]:
    """Use ML to suggest new sources based on corpus gaps."""
    cur = conn.cursor()

    # Get current corpus features
    cur.execute("""
        SELECT language, diachronic_stage, genre, word_count, metadata_quality, annotation_score
        FROM corpus_items
        WHERE diachronic_stage IS NOT NULL AND diachronic_stage != ''
    """)

    texts = cur.fetchall()

    if len(texts) < 10:
        return ["Need more texts for ML analysis (minimum 10 required)"]

    # Convert to feature matrix
    features = []
    labels = []

    for lang, stage, genre, wc, mq, ascore in texts:
        # Encode categorical features
        lang_code = {'grc': 0, 'lat': 1, 'en': 2}.get(lang, 3)
        stage_code = hash(stage) % 1000  # Simple hash for stages
        genre_code = hash(genre or '') % 100

        features.append([lang_code, stage_code, genre_code, wc, mq, ascore])
        labels.append(f"{lang}_{stage}")

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Cluster existing texts
    n_clusters = min(5, len(features) // 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Find underrepresented clusters
    cluster_counts = Counter(clusters)
    avg_count = sum(cluster_counts.values()) / len(cluster_counts)

    suggestions = []

    # Suggest based on cluster gaps
    for cluster_id, count in cluster_counts.items():
        if count < avg_count * 0.5:  # Underrepresented
            # Find representative texts in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if cluster_indices:
                rep_text = texts[cluster_indices[0]]
                lang, stage = rep_text[0], rep_text[1]
                suggestions.append(f"Expand cluster {cluster_id}: more {lang} texts from {stage} period")

    # Language balance suggestions
    lang_counts = Counter([t[0] for t in texts])
    total_texts = len(texts)
    for lang, count in lang_counts.items():
        pct = (count / total_texts) * 100
        if pct < 20:  # Less than 20% of corpus
            suggestions.append(f"Increase {lang} representation (currently {pct:.1f}%)")

    # Period gaps
    period_counts = Counter([t[1] for t in texts])
    for period, count in period_counts.items():
        if count < 3:  # Very few texts in period
            suggestions.append(f"Add more texts from {period} period")

    return suggestions[:10]  # Top 10 suggestions


def write_report(overview: Dict, tb_summary: Dict[str, Dict[str, int]],
                 greek_rows: List[sqlite3.Row], latin_rows: List[sqlite3.Row],
                 period_suggestions: List[Tuple[str, str, int]],
                 eval_summary: Dict[str, float],
                 contrastive_rows: List[Tuple[str, str, str, int, int, int]],
                 ml_suggestions: List[str]) -> Path:
    lines.append("")
    lines.append(f"- Total texts: **{overview['total_texts']}**")
    lines.append(f"- Total words: **{overview['total_words']:,}**")
    lines.append(f"- Avg. metadata quality: **{overview['avg_metadata']:.1f}%**")
    lines.append(f"- Avg. annotation score: **{overview['avg_annotation']:.1f}%**")
    lines.append("")

    lines.append("### By language (texts, words)")
    lines.append("")
    lines.append("Language | Texts | Words")
    lines.append("-------- | ----- | -----")
    for row in overview["by_language"]:
        lines.append(f"{row['language']} | {row['count']} | {row['words']:,}")
    lines.append("")

    # Treebank quality
    lines.append("## Treebank quality overview")
    lines.append("")
    lines.append("Treebank quality counts per language:")
    lines.append("")
    lines.append(format_treebank_summary(tb_summary))
    lines.append("")

    # Evaluation / multi-AI coverage (if any)
    if eval_summary:
        lines.append("## Multi-AI coverage (from evaluation)")
        lines.append("")
        for key, val in sorted(eval_summary.items()):
            lang = key.replace("multi_ai_coverage_", "")
            lines.append(f"- {lang}: multi-AI coverage on evaluated items ≈ **{val*100:.1f}%**")
        lines.append("")

    # Greek & Latin texts needing attention
    def format_rows(title: str, rows: List[sqlite3.Row]) -> None:
        lines.append(title)
        lines.append("")
        if not rows:
            lines.append("(none)")
            lines.append("")
            return
        lines.append("ID | Title | Words | Meta% | Annot% | Treebank")
        lines.append("-- | ----- | ----- | ----- | ------- | --------")
        for r in rows:
            title_short = (r["title"] or "").strip()
            if len(title_short) > 40:
                title_short = title_short[:37] + "..."
            lines.append(
                f"{r['id']} | {title_short} | {r['word_count'] or 0:,} | "
                f"{(r['metadata_quality'] or 0):.1f} | {(r['annotation_score'] or 0):.1f} | "
                f"{r['treebank_quality']}"
            )
        lines.append("")

    lines.append("## Greek texts needing attention")
    lines.append("")
    format_rows("", greek_rows[:20])

    lines.append("## Latin texts needing attention")
    lines.append("")
    format_rows("", latin_rows[:20])

    # Underrepresented periods
    lines.append("## Underrepresented diachronic stages (Greek & Latin)")
    lines.append("")
    if not period_suggestions:
        lines.append("No clear gaps detected.")
    else:
        lines.append("Language | Diachronic stage | Texts")
        lines.append("-------- | ---------------- | -----")
        for lang, stage, cnt in period_suggestions:
            lines.append(f"{lang} | {stage} | {cnt}")
    lines.append("")

    # Contrastive opportunities (language contact & diachrony)
    lines.append("## Contrastive opportunities (language contact & diachrony)")
    lines.append("")
    if not contrastive_rows:
        lines.append("No contrastive clusters detected yet (original_language not set).")
    else:
        lines.append("Origin | Target | Diachronic stage | Texts | Retranslations | Retellings")
        lines.append("------ | ------ | ---------------- | ----- | -------------- | ----------")
        for orig_lang, lang, stage, cnt, retrans, retell in contrastive_rows:
            stage_display = stage or "(unspecified)"
            lines.append(
                f"{orig_lang or '(?)'} | {lang or '(?)'} | {stage_display} | "
                f"{cnt} | {retrans or 0} | {retell or 0}"
            )
    lines.append("")

    # ML-powered source suggestions
    lines.append("## ML-Powered Source Suggestions")
    lines.append("")
    if not ml_suggestions:
        lines.append("No suggestions available (need more texts for analysis).")
    else:
        lines.append("Priority recommendations for expanding the corpus:")
        for i, suggestion in enumerate(ml_suggestions, 1):
            lines.append(f"{i}. {suggestion}")
    lines.append("")

    # Simple action suggestions
    lines.append("## Suggested actions for tonight")
    lines.append("")
    lines.append("- Prioritise PROIEL/Stanza annotation for the top Greek & Latin texts listed above.")
    lines.append("- Collect more texts in the underrepresented stages (see table) from your configured repositories.")
    lines.append("- Use the 'Texts needing attention' section in the HTML dashboard together with this report.")
    lines.append("- Consider the ML suggestions for strategic corpus expansion.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Research agent report written to %s", out_path)
    return out_path


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        overview = corpus_overview(conn)
        tb_summary = treebank_quality_by_language(conn)
        greek_rows, latin_rows = greek_latin_gaps(conn)
        period_suggestions = underrepresented_periods(conn)
        contrastive_rows = contrastive_opportunities(conn)
        ml_suggestions = ml_source_suggestions(conn)
    finally:
        conn.close()

    eval_summary: Dict[str, float] = {}
    eval_path = latest_evaluation_csv()
    if eval_path is not None:
        eval_summary = load_eval_summary(eval_path)

    write_report(overview, tb_summary, greek_rows, latin_rows, period_suggestions, eval_summary, contrastive_rows, ml_suggestions)


if __name__ == "__main__":
    main()
