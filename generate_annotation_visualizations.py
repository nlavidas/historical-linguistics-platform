#!/usr/bin/env python3
"""Generate an HTML visualization report for metadata and annotation quality.

- Reads aggregate statistics from corpus_platform.db.
- Optionally reads the latest evaluation_report_*.csv if present.
- Produces an HTML file with JavaScript charts (Chart.js via CDN).

Output: research_exports/visual_reports/annotation_dashboard_YYYYMMDD_HHMMSS.html
"""

import csv
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
EVAL_DIR = ROOT / "research_exports" / "evaluation"
VIS_DIR = ROOT / "research_exports" / "visual_reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_annotation_visualizations")


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


def corpus_aggregates(conn: sqlite3.Connection):
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
        SELECT language, COUNT(*), AVG(metadata_quality), AVG(annotation_score)
        FROM corpus_items
        GROUP BY language
        ORDER BY COUNT(*) DESC
        """
    )
    by_lang = [
        {
            "language": lang or "unknown",
            "count": count or 0,
            "avg_metadata": avg_m or 0.0,
            "avg_annotation": avg_a or 0.0,
        }
        for (lang, count, avg_m, avg_a) in cur.fetchall()
    ]

    cur.execute(
        """
        SELECT status, COUNT(*)
        FROM corpus_items
        GROUP BY status
        """
    )
    by_status = {status or "unknown": cnt for (status, cnt) in cur.fetchall()}

    cur.execute(
        """
        SELECT
            SUM(CASE WHEN proiel_xml IS NOT NULL AND proiel_xml != '' THEN 1 ELSE 0 END) as annotated,
            SUM(CASE WHEN proiel_xml IS NULL OR proiel_xml = '' THEN 1 ELSE 0 END) as not_annotated
        FROM corpus_items
        """
    )
    annotated, not_annotated = cur.fetchone()

    # Treebank quality distribution
    cur.execute(
        """
        SELECT COALESCE(treebank_quality, 'none') as tq, COUNT(*)
        FROM corpus_items
        GROUP BY COALESCE(treebank_quality, 'none')
        """
    )
    treebank_counts = {tq: cnt for (tq, cnt) in cur.fetchall()}

    cur.execute(
        """
        SELECT language, COALESCE(treebank_quality, 'none') as tq, COUNT(*)
        FROM corpus_items
        GROUP BY language, COALESCE(treebank_quality, 'none')
        """
    )
    treebank_by_language = {}
    for lang, tq, cnt in cur.fetchall():
        lang_key = lang or "unknown"
        if lang_key not in treebank_by_language:
            treebank_by_language[lang_key] = {}
        treebank_by_language[lang_key][tq] = cnt

    return {
        "total_texts": total_texts or 0,
        "total_words": total_words or 0,
        "avg_metadata": avg_meta or 0.0,
        "avg_annotation": avg_annot or 0.0,
        "by_language": by_lang,
        "by_status": by_status,
        "annotated": annotated or 0,
        "not_annotated": not_annotated or 0,
        "treebank_counts": treebank_counts,
        "treebank_by_language": treebank_by_language,
    }


def trends_and_attention(conn: sqlite3.Connection):
    """Compute timeline trends and a list of texts needing attention."""
    cur = conn.cursor()

    # Timeline: one row per day with average metadata/annotation
    cur.execute(
        """
        SELECT substr(date_added, 1, 10) as day,
               COUNT(*) as total_texts,
               AVG(metadata_quality) as avg_metadata,
               AVG(annotation_score) as avg_annotation
        FROM corpus_items
        WHERE date_added IS NOT NULL
        GROUP BY substr(date_added, 1, 10)
        ORDER BY day
        """
    )
    timeline = [
        {
            "day": day or "",
            "total_texts": total or 0,
            "avg_metadata": avg_m or 0.0,
            "avg_annotation": avg_a or 0.0,
        }
        for (day, total, avg_m, avg_a) in cur.fetchall()
    ]

    # Texts needing attention: weak metadata or weak annotation (with special
    # focus on Greek/Latin for annotation score).
    cur.execute(
        """
        SELECT id, title, language, word_count, metadata_quality, annotation_score, status
        FROM corpus_items
        WHERE
            (metadata_quality IS NOT NULL AND metadata_quality < 80)
         OR (language IN ('grc', 'lat') AND (annotation_score IS NULL OR annotation_score < 60))
        ORDER BY
            COALESCE(annotation_score, 0) ASC,
            COALESCE(metadata_quality, 0) ASC,
            id DESC
        LIMIT 15
        """
    )

    attention = []
    for (id_, title, lang, wc, meta_q, annot_q, status) in cur.fetchall():
        attention.append(
            {
                "id": id_,
                "title": title or "",
                "language": lang or "",
                "word_count": wc or 0,
                "metadata_quality": meta_q or 0.0,
                "annotation_score": annot_q or 0.0,
                "status": status or "",
            }
        )

    return timeline, attention


def eval_aggregates(rows):
    if not rows:
        return {
            "has_eval": False,
            "avg_meta_completeness": 0.0,
            "coverage_counts": {},
        }

    total = len(rows)
    meta_scores = [float(r["metadata_completeness"]) for r in rows]
    avg_meta = sum(meta_scores) / total if total else 0.0

    cov = Counter()
    for r in rows:
        for key in [
            "has_multi_ai",
            "has_tokens",
            "has_sentences",
            "has_pos",
            "has_lemmas",
            "has_dependencies",
        ]:
            if r.get(key) == "1":
                cov[key] += 1

    return {
        "has_eval": True,
        "avg_meta_completeness": avg_meta,
        "coverage_counts": dict(cov),
        "total_eval": total,
    }


def build_html(data: dict) -> str:
    # Avoid closing </script> inside JSON
    json_data = json.dumps(data).replace("</", "<\\/")

    eval_note = "and latest evaluation report" if data.get("eval", {}).get("has_eval") else ""

    eval_note = "and latest evaluation report" if data.get("eval", {}).get("has_eval") else ""

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Annotation & Metadata Quality Dashboard</title>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fb;
            color: #1f2933;
        }}
        h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
        }}
        .card {{
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.08);
            padding: 16px 20px 20px;
        }}
        .card h2 {{
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin: 2px 0;
        }}
        .footer {{
            margin-top: 24px;
            font-size: 0.8rem;
            color: #9ca3af;
        }}
        canvas {{
            max-width: 100%;
        }}
    </style>
</head>
<body>
    <h1>Annotation & Metadata Quality Dashboard</h1>
    <div class=\"subtitle\">Diachronic Multilingual Corpus – generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class=\"grid\">
        <div class=\"card\">
            <h2>Corpus Overview</h2>
            <div class=\"stat-row\"><span>Total texts</span><span id=\"stat-total\"></span></div>
            <div class=\"stat-row\"><span>Total words</span><span id=\"stat-words\"></span></div>
            <div class=\"stat-row\"><span>Annotated (PROIEL)</span><span id=\"stat-annotated\"></span></div>
            <div class=\"stat-row\"><span>Not annotated</span><span id=\"stat-not-annotated\"></span></div>
            <div class=\"stat-row\"><span>Avg. metadata quality</span><span id=\"stat-avg-meta\"></span></div>
            <div class=\"stat-row\"><span>Avg. annotation score</span><span id=\"stat-avg-annot\"></span></div>
        </div>

        <div class=\"card\">
            <h2>Texts by Language</h2>
            <canvas id=\"langChart\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Status Distribution</h2>
            <canvas id=\"statusChart\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Treebank quality (by language)</h2>
            <canvas id=\"treebankChart\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Metadata vs Annotation (by Language)</h2>
            <canvas id=\"qualityChart\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Evaluation Coverage (Multi-AI)</h2>
            <canvas id=\"coverageChart\"></canvas>
            <p style=\"font-size:0.8rem;color:#6b7280;margin-top:8px;\" id=\"coverage-note\"></p>
        </div>

        <div class=\"card\">
            <h2>Trends over time</h2>
            <canvas id=\"trendChart\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Texts needing attention</h2>
            <p style=\"font-size:0.8rem;color:#6b7280;margin-top:4px;\" id=\"attentionEmptyNote\">All tracked texts currently satisfy the thresholds.</p>
            <table style=\"width:100%;border-collapse:collapse;font-size:0.8rem;margin-top:8px;\">
                <thead>
                    <tr>
                        <th style=\"text-align:left;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">ID</th>
                        <th style=\"text-align:left;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">Lang</th>
                        <th style=\"text-align:left;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">Title</th>
                        <th style=\"text-align:right;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">Words</th>
                        <th style=\"text-align:right;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">Meta %</th>
                        <th style=\"text-align:right;padding:4px 6px;border-bottom:1px solid #e5e7eb;\">Annot %</th>
                    </tr>
                </thead>
                <tbody id=\"attentionTableBody\"></tbody>
            </table>
        </div>
    </div>

    <div class=\"footer\">
        Generated from corpus_platform.db {eval_note}.<br>
        Use this dashboard to decide which languages, periods and texts to improve next.
    </div>

    <script>
        const DATA = {json_data};

        function formatInt(x) {{
            return x.toLocaleString('en-US');
        }}

        // Overview stats
        document.getElementById('stat-total').textContent = formatInt(DATA.corpus.total_texts || 0);
        document.getElementById('stat-words').textContent = formatInt(DATA.corpus.total_words || 0);
        document.getElementById('stat-annotated').textContent = formatInt(DATA.corpus.annotated || 0);
        document.getElementById('stat-not-annotated').textContent = formatInt(DATA.corpus.not_annotated || 0);
        document.getElementById('stat-avg-meta').textContent = (DATA.corpus.avg_metadata || 0).toFixed(1) + '%';
        document.getElementById('stat-avg-annot').textContent = (DATA.corpus.avg_annotation || 0).toFixed(1) + '%';

        // Language chart
        const langCtx = document.getElementById('langChart').getContext('2d');
        const langs = DATA.corpus.by_language.map(x => x.language || 'unknown');
        const langCounts = DATA.corpus.by_language.map(x => x.count);

        new Chart(langCtx, {{
            type: 'bar',
            data: {{
                labels: langs,
                datasets: [{{
                    label: 'Texts',
                    data: langCounts,
                    backgroundColor: 'rgba(37, 99, 235, 0.7)'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                }},
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});

        // Status chart
        const statusCtx = document.getElementById('statusChart').getContext('2d');
        const statusLabels = Object.keys(DATA.corpus.by_status || {{}});
        const statusValues = statusLabels.map(k => DATA.corpus.by_status[k]);

        new Chart(statusCtx, {{
            type: 'pie',
            data: {{
                labels: statusLabels,
                datasets: [{{
                    data: statusValues,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(234, 179, 8, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});

        // Quality by language
        const qualCtx = document.getElementById('qualityChart').getContext('2d');
        const avgMetaByLang = DATA.corpus.by_language.map(x => x.avg_metadata);
        const avgAnnotByLang = DATA.corpus.by_language.map(x => x.avg_annotation);

        new Chart(qualCtx, {{
            type: 'bar',
            data: {{
                labels: langs,
                datasets: [
                    {{
                        label: 'Metadata %',
                        data: avgMetaByLang,
                        backgroundColor: 'rgba(16, 185, 129, 0.7)'
                    }},
                    {{
                        label: 'Annotation %',
                        data: avgAnnotByLang,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true, max: 100 }}
                }}
            }}
        }});

        // Treebank quality by language (stacked bar)
        const tbCanvas = document.getElementById('treebankChart');
        const tbData = DATA.corpus.treebank_by_language || {{}};
        const tbLangs = Object.keys(tbData);
        if (tbCanvas && tbLangs.length) {{
            const tbCtx = tbCanvas.getContext('2d');
            const levels = ['excellent', 'good', 'partial', 'none'];
            const colors = {{
                excellent: 'rgba(16, 185, 129, 0.9)',
                good: 'rgba(59, 130, 246, 0.9)',
                partial: 'rgba(234, 179, 8, 0.9)',
                none: 'rgba(148, 163, 184, 0.9)',
            }};

            const datasets = levels.map(level => {{
                return {{
                    label: level.charAt(0).toUpperCase() + level.slice(1),
                    data: tbLangs.map(lang => (tbData[lang] && tbData[lang][level]) || 0),
                    backgroundColor: colors[level],
                    stack: 'treebank',
                }};
            }});

            new Chart(tbCtx, {{
                type: 'bar',
                data: {{
                    labels: tbLangs,
                    datasets: datasets,
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ stacked: true }},
                        y: {{ stacked: true, beginAtZero: true }},
                    }},
                }},
            }});
        }}

        // Trends over time (average metadata/annotation per day)
        const trendCanvas = document.getElementById('trendChart');
        const timeline = DATA.timeline || [];
        if (trendCanvas && timeline.length) {{
            const trendCtx = trendCanvas.getContext('2d');
            const days = timeline.map(x => x.day);
            const metaTrend = timeline.map(x => x.avg_metadata);
            const annotTrend = timeline.map(x => x.avg_annotation);

            new Chart(trendCtx, {{
                type: 'line',
                data: {{
                    labels: days,
                    datasets: [
                        {{
                            label: 'Metadata %',
                            data: metaTrend,
                            borderColor: 'rgba(16, 185, 129, 1)',
                            backgroundColor: 'rgba(16, 185, 129, 0.12)',
                            tension: 0.2
                        }},
                        {{
                            label: 'Annotation %',
                            data: annotTrend,
                            borderColor: 'rgba(59, 130, 246, 1)',
                            backgroundColor: 'rgba(59, 130, 246, 0.12)',
                            tension: 0.2
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, max: 100 }}
                    }}
                }}
            }});
        }}

        // Evaluation coverage chart
        const evalData = DATA.eval || {{}};
        const covCtx = document.getElementById('coverageChart').getContext('2d');

        if (evalData.has_eval && evalData.total_eval > 0) {{
            const rawCounts = evalData.coverage_counts || {{}};
            const covKeyToLabel = {{
                has_multi_ai: 'Multi-AI outputs',
                has_tokens: 'Tokenization',
                has_sentences: 'Sentence segmentation',
                has_pos: 'POS tagging',
                has_lemmas: 'Lemmatization',
                has_dependencies: 'Dependency parsing',
            }};

            const keys = Object.keys(rawCounts);
            const covLabels = keys.map(k => covKeyToLabel[k] || k);
            const covPercents = keys.map(k => {{
                const count = rawCounts[k] || 0;
                const total = evalData.total_eval || 1;
                return (count / total) * 100.0;
            }});

            new Chart(covCtx, {{
                type: 'bar',
                data: {{
                    labels: covLabels,
                    datasets: [{
                        label: '% of evaluated texts with this layer',
                        data: covPercents,
                        backgroundColor: 'rgba(139, 92, 246, 0.8)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, max: 100 }}
                    }}
                }}
            }});

            const note = 'Based on ' + evalData.total_eval +
                ' evaluated texts. Values show the percentage of texts that already have each processing layer ' +
                '(multi-AI outputs, tokenization, sentence segmentation, POS tagging, lemmatization, dependency parsing).';
            document.getElementById('coverage-note').textContent = note;
        }} else {{
            document.getElementById('coverage-note').textContent =
                'No evaluation report found yet. Run evaluate_metadata_and_annotations.py first or let the full professional cycle finish.';
        }}

        // Texts needing attention table
        const attention = DATA.attention_texts || [];
        const attBody = document.getElementById('attentionTableBody');
        const attEmpty = document.getElementById('attentionEmptyNote');
        if (attBody && attention.length) {{
            if (attEmpty) {{
                attEmpty.style.display = 'none';
            }}
            attention.forEach(item => {{
                const tr = document.createElement('tr');

                const tdId = document.createElement('td');
                tdId.textContent = item.id;
                tr.appendChild(tdId);

                const tdLang = document.createElement('td');
                tdLang.textContent = item.language || '';
                tr.appendChild(tdLang);

                const tdTitle = document.createElement('td');
                tdTitle.textContent = item.title || '';
                tr.appendChild(tdTitle);

                const tdWords = document.createElement('td');
                tdWords.style.textAlign = 'right';
                tdWords.textContent = formatInt(item.word_count || 0);
                tr.appendChild(tdWords);

                const tdMeta = document.createElement('td');
                tdMeta.style.textAlign = 'right';
                tdMeta.textContent = (item.metadata_quality || 0).toFixed(1) + '%';
                tr.appendChild(tdMeta);

                const tdAnnot = document.createElement('td');
                tdAnnot.style.textAlign = 'right';
                tdAnnot.textContent = (item.annotation_score || 0).toFixed(1) + '%';
                tr.appendChild(tdAnnot);

                attBody.appendChild(tr);
            }});
        }}
    </script>
</body>
</html>
"""


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        raise SystemExit(1)

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    corpus = corpus_aggregates(conn)
    timeline, attention = trends_and_attention(conn)
    conn.close()

    eval_path = latest_evaluation_csv()
    if eval_path is not None:
        eval_rows = load_evaluation(eval_path)
        eval_data = eval_aggregates(eval_rows)
    else:
        eval_data = {"has_eval": False, "avg_meta_completeness": 0.0, "coverage_counts": {}, "total_eval": 0}

    data = {"corpus": corpus, "eval": eval_data, "timeline": timeline, "attention_texts": attention}

    now = datetime.now()
    out_path = VIS_DIR / f"annotation_dashboard_{now.strftime('%Y%m%d_%H%M%S')}.html"
    html = build_html(data)
    out_path.write_text(html, encoding="utf-8")

    logger.info("Annotation visualization report written to %s", out_path)


if __name__ == "__main__":
    main()
