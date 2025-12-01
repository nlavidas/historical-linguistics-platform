#!/usr/bin/env python3
"""Advanced Valency Analysis for Contrastive Linguistics.

Performs statistical analysis of valency patterns from PROIEL-annotated texts,
using scikit-learn and spaCy for contrastive linguistics studies.

Features:
- Extract valency frames from PROIEL XML
- Statistical modeling of valency patterns
- Contrastive analysis across languages/stages
- Machine learning for pattern classification

Run:
    python advanced_valency_analysis.py

Outputs:
    research_exports/valency_analysis/valency_report_YYYYMMDD_HHMMSS.md
    research_exports/valency_analysis/valency_patterns.json
"""

import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"
VALENCY_DIR = ROOT / "research_exports" / "valency_analysis"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("advanced_valency_analysis")

class AdvancedValencyAnalyzer:
    """Advanced statistical analysis of valency patterns."""

    def __init__(self):
        self.db_path = DB_PATH
        VALENCY_DIR.mkdir(parents=True, exist_ok=True)

    def extract_valency_patterns(self, proiel_xml: str) -> List[Dict]:
        """Extract valency patterns from PROIEL XML."""
        patterns = []

        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(proiel_xml)

            # Find all sentences
            for sentence in root.findall('.//sentence'):
                sent_id = sentence.get('id', '')

                # Find all tokens in sentence
                tokens = sentence.findall('.//token')
                if not tokens:
                    continue

                # Build dependency graph
                token_dict = {}
                for token in tokens:
                    tid = token.get('id', '')
                    form = token.get('form', '')
                    lemma = token.get('lemma', '')
                    pos = token.get('part-of-speech', '')
                    head_id = token.get('head-id', '')

                    token_dict[tid] = {
                        'form': form,
                        'lemma': lemma,
                        'pos': pos,
                        'head_id': head_id,
                        'dependents': []
                    }

                # Build dependents
                for tid, token in token_dict.items():
                    head_id = token['head_id']
                    if head_id and head_id in token_dict:
                        token_dict[head_id]['dependents'].append(tid)

                # Extract valency patterns for verbs
                for tid, token in token_dict.items():
                    if token['pos'].startswith('V'):  # Verb
                        pattern = self.extract_verb_valency(token, token_dict)
                        if pattern:
                            patterns.append({
                                'sentence_id': sent_id,
                                'verb': token['lemma'],
                                'pattern': pattern
                            })

        except Exception as e:
            logger.warning(f"Failed to parse PROIEL XML: {e}")

        return patterns

    def extract_verb_valency(self, verb_token: Dict, token_dict: Dict) -> Optional[str]:
        """Extract valency pattern for a verb."""
        dependents = verb_token['dependents']

        if not dependents:
            return None

        # Classify dependents by grammatical function
        subjects = []
        objects = []
        obliques = []

        for dep_id in dependents:
            if dep_id not in token_dict:
                continue

            dep = token_dict[dep_id]
            pos = dep['pos']

            # Simple classification (in practice, use PROIEL features)
            if pos in ['S-', 'Sb']:  # Subject
                subjects.append(dep_id)
            elif pos in ['O-', 'Ob']:  # Object
                objects.append(dep_id)
            else:  # Oblique
                obliques.append(dep_id)

        # Build pattern string
        pattern_parts = []
        if subjects:
            pattern_parts.append(f"SUBJ({len(subjects)})")
        if objects:
            pattern_parts.append(f"OBJ({len(objects)})")
        if obliques:
            pattern_parts.append(f"OBL({len(obliques)})")

        return '+'.join(pattern_parts) if pattern_parts else None

    def analyze_corpus_valency(self) -> Dict:
        """Analyze valency patterns across the corpus."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all texts with PROIEL XML
        cursor.execute("""
            SELECT id, title, language, proiel_xml, diachronic_stage
            FROM corpus_items
            WHERE proiel_xml IS NOT NULL AND proiel_xml != ''
        """)

        texts = cursor.fetchall()
        conn.close()

        # Extract patterns from all texts
        all_patterns = []
        pattern_stats = defaultdict(lambda: defaultdict(Counter))

        for text_id, title, language, proiel_xml, stage in texts:
            logger.info(f"Analyzing valency for: {title}")

            patterns = self.extract_valency_patterns(proiel_xml)
            all_patterns.extend(patterns)

            # Statistics by language and stage
            for pattern in patterns:
                verb = pattern['verb']
                pat = pattern['pattern']
                pattern_stats[language][stage][f"{verb}:{pat}"] += 1

        return {
            'total_patterns': len(all_patterns),
            'patterns': all_patterns,
            'stats_by_lang_stage': dict(pattern_stats)
        }

    def statistical_modeling(self, patterns: List[Dict]) -> Dict:
        """Apply statistical modeling to valency patterns."""
        if not patterns:
            return {}

        # Convert patterns to feature vectors
        pattern_strings = [f"{p['verb']}:{p['pattern']}" for p in patterns]

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            X = vectorizer.fit_transform(pattern_strings)

            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())

            # K-means clustering
            kmeans = KMeans(n_clusters=min(5, len(patterns)), random_state=42)
            clusters = kmeans.fit_predict(X_pca)

            # Analyze clusters
            cluster_analysis = defaultdict(list)
            for i, cluster in enumerate(clusters):
                cluster_analysis[cluster].append(pattern_strings[i])

            return {
                'feature_names': vectorizer.get_feature_names_out().tolist(),
                'pca_components': pca.components_.tolist(),
                'clusters': dict(cluster_analysis),
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }

        except Exception as e:
            logger.warning(f"Statistical modeling failed: {e}")
            return {}

    def contrastive_analysis(self, stats_by_lang_stage: Dict) -> Dict:
        """Perform contrastive analysis across languages and stages."""
        contrastive = {}

        # Compare pattern distributions between languages
        languages = list(stats_by_lang_stage.keys())

        if len(languages) >= 2:
            for i, lang1 in enumerate(languages):
                for j, lang2 in enumerate(languages):
                    if i < j:
                        key = f"{lang1}_vs_{lang2}"
                        contrastive[key] = self.compare_languages(
                            stats_by_lang_stage[lang1],
                            stats_by_lang_stage[lang2],
                            lang1, lang2
                        )

        return contrastive

    def compare_languages(self, lang1_stats: Dict, lang2_stats: Dict,
                         lang1: str, lang2: str) -> Dict:
        """Compare valency patterns between two languages."""
        # Get all unique patterns
        all_patterns = set()
        for stage_stats in lang1_stats.values():
            all_patterns.update(stage_stats.keys())
        for stage_stats in lang2_stats.values():
            all_patterns.update(stage_stats.keys())

        # Compare frequencies
        comparisons = {}
        for pattern in all_patterns:
            lang1_freq = sum(stage_stats.get(pattern, 0) for stage_stats in lang1_stats.values())
            lang2_freq = sum(stage_stats.get(pattern, 0) for stage_stats in lang2_stats.values())

            if lang1_freq + lang2_freq > 0:
                comparisons[pattern] = {
                    lang1: lang1_freq,
                    lang2: lang2_freq,
                    'ratio': lang1_freq / (lang2_freq + 1),  # Avoid division by zero
                    'difference': lang1_freq - lang2_freq
                }

        return comparisons

    def generate_report(self, analysis: Dict, modeling: Dict, contrastive: Dict) -> str:
        """Generate comprehensive valency analysis report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = []
        report.append(f"# Advanced Valency Analysis Report ({now})")
        report.append("")

        report.append("## Overview")
        report.append(f"- Total valency patterns extracted: **{analysis['total_patterns']}**")
        report.append("")

        report.append("## Patterns by Language and Stage")
        for lang, stage_stats in analysis['stats_by_lang_stage'].items():
            report.append(f"### {lang.upper()}")
            for stage, patterns in stage_stats.items():
                report.append(f"#### {stage}")
                report.append("Pattern | Frequency")
                report.append("------- | ---------")
                for pattern, freq in patterns.most_common(10):
                    report.append(f"{pattern} | {freq}")
                report.append("")

        if modeling:
            report.append("## Statistical Modeling")
            report.append(f"- Number of pattern clusters identified: **{len(modeling.get('clusters', {}))}**")
            report.append("")

            # Show top patterns per cluster
            for cluster_id, patterns in modeling.get('clusters', {}).items():
                report.append(f"### Cluster {cluster_id}")
                pattern_counts = Counter(patterns)
                for pattern, count in pattern_counts.most_common(5):
                    report.append(f"- {pattern}: {count}")
                report.append("")

        if contrastive:
            report.append("## Contrastive Analysis")
            for comparison, data in contrastive.items():
                report.append(f"### {comparison.replace('_', ' ').title()}")
                # Show most distinctive patterns
                sorted_patterns = sorted(data.items(),
                                       key=lambda x: abs(x[1]['difference']), reverse=True)
                for pattern, stats in sorted_patterns[:10]:
                    lang1, lang2 = comparison.split('_vs_')
                    report.append(f"- **{pattern}**: {lang1}={stats[lang1]}, {lang2}={stats[lang2]} "
                                f"(ratio: {stats['ratio']:.2f})")
                report.append("")

        return "\n".join(report)

    def run_analysis(self) -> Tuple[str, Dict]:
        """Run complete valency analysis."""
        logger.info("Starting advanced valency analysis...")

        # Extract and analyze patterns
        analysis = self.analyze_corpus_valency()

        # Statistical modeling
        modeling = self.statistical_modeling(analysis['patterns'])

        # Contrastive analysis
        contrastive = self.contrastive_analysis(analysis['stats_by_lang_stage'])

        # Generate report
        report = self.generate_report(analysis, modeling, contrastive)

        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = VALENCY_DIR / f"valency_report_{timestamp}.md"
        patterns_file = VALENCY_DIR / f"valency_patterns_{timestamp}.json"

        report_file.write_text(report, encoding="utf-8")

        output_data = {
            'analysis': analysis,
            'modeling': modeling,
            'contrastive': contrastive
        }
        patterns_file.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")

        logger.info(f"Valency analysis complete. Report: {report_file}")
        logger.info(f"Patterns data: {patterns_file}")

        return report, output_data

def main():
    analyzer = AdvancedValencyAnalyzer()
    report, data = analyzer.run_analysis()

    # Print summary
    print("Valency Analysis Summary:")
    print(f"- Patterns extracted: {data['analysis']['total_patterns']}")
    print(f"- Languages analyzed: {len(data['analysis']['stats_by_lang_stage'])}")
    if data['modeling']:
        print(f"- Pattern clusters: {len(data['modeling'].get('clusters', {}))}")
    if data['contrastive']:
        print(f"- Contrastive comparisons: {len(data['contrastive'])}")

if __name__ == '__main__':
    main()
