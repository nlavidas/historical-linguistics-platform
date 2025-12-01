#!/usr/bin/env python3
"""
COST-EFFECTIVE VALENCY ANALYSIS SYSTEM
Community-driven tools for verb valency analysis and reporting
Optimized for low-cost servers with historical linguistics focus

Analyzes native vs borrowed verbs and their argument structures
Produces comparative valency lexica and reports
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Iterator, Optional, Tuple, Set
import sqlite3
from datetime import datetime
import time

# Free community-driven analysis tools
try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('valency_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostEffectiveValencyAnalyzer:
    """
    Memory-efficient valency analysis for historical linguistics
    Analyzes verb argument structures and produces comparative reports
    """

    def __init__(self, db_path="corpus_efficient.db", batch_size=25):
        self.db_path = db_path
        self.batch_size = batch_size

        # Valency analysis parameters
        self.min_verb_occurrences = 5  # Minimum occurrences to analyze
        self.max_valency_patterns = 10  # Maximum patterns per verb

        # Language-specific verb morphology
        self.verb_suffixes = {
            'grc': ['ω', 'ομαι', 'μι', 'μαι', 'νυμι', 'αμαι'],  # Ancient Greek
            'la': ['o', 'or', 'i', 'ri', 're'],  # Latin
            'en': ['ate', 'ify', 'ize', 'ise'],  # Borrowed verb markers
            'de': ['ieren', 'isieren'],  # German borrowed verbs
            'fr': ['er', 'ir', 're']  # French infinitives
        }

        # Etymological markers for borrowed verbs
        self.borrowed_markers = {
            'grc': ['λογ', 'γραφ', 'νομ', 'κρατ', 'φιλ', 'σοφ'],  # Greek roots in other languages
            'la': ['dict', 'rupt', 'pend', 'tend', 'ced', 'mit'],   # Latin roots
            'en': ['tele', 'photo', 'bio', 'geo', 'micro', 'macro'], # Greek/Latin in English
        }

    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _batch_load_annotated_texts(self, limit=None, offset=0) -> Iterator[List[Dict]]:
        """Load annotated texts in batches"""

        annotated_dir = Path("annotated")
        if not annotated_dir.exists():
            logger.warning("No annotated directory found")
            return

        json_files = list(annotated_dir.glob("*.jsonl.gz"))
        if not json_files:
            logger.warning("No annotated files found")
            return

        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

        import gzip
        batch = []

        with gzip.open(latest_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and line_num >= limit:
                    break

                try:
                    item = json.loads(line.strip())
                    batch.append(item)

                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        if batch:
            yield batch

    def classify_verb_etymology(self, verb: str, language: str) -> str:
        """Classify verb as native or borrowed based on morphological markers"""

        verb_lower = verb.lower()

        # Language-specific classification
        if language == 'grc':
            # Ancient Greek verbs
            if any(verb_lower.endswith(suffix) for suffix in self.verb_suffixes['grc']):
                return 'native'
            else:
                return 'borrowed'

        elif language == 'la':
            # Latin verbs
            if any(verb_lower.endswith(suffix) for suffix in self.verb_suffixes['la']):
                return 'native'
            else:
                return 'borrowed'

        elif language == 'en':
            # English verbs - check for borrowed markers
            if any(marker in verb_lower for marker in self.borrowed_markers['en']):
                return 'borrowed'
            elif verb_lower.endswith('ate') or verb_lower.endswith('ify') or verb_lower.endswith('ize'):
                # Likely borrowed from Latin/Greek
                return 'borrowed'
            else:
                # Assume native Germanic verb
                return 'native'

        elif language == 'de':
            # German verbs
            if verb_lower.endswith('ieren') or verb_lower.endswith('isieren'):
                return 'borrowed'  # Likely from French/Latin
            else:
                return 'native'

        elif language == 'fr':
            # French verbs - most are native Romance
            return 'native'

        # Default classification
        return 'unknown'

    def extract_verb_valency(self, annotation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract verb valency patterns from annotated data"""

        valency_patterns = []

        try:
            # Get argument structures if available
            if 'annotations' in annotation_data and 'argument_structure' in annotation_data['annotations']:
                arg_structures = annotation_data['annotations']['argument_structure']

                for structure in arg_structures:
                    if 'predicate' in structure and 'arguments' in structure:
                        verb = structure['predicate']
                        arguments = structure['arguments']

                        # Classify arguments by semantic role
                        subjects = []
                        direct_objects = []
                        indirect_objects = []
                        prepositional_objects = []

                        for arg in arguments:
                            role = arg.get('role', '')
                            text = arg.get('text', '')

                            if role in ['nsubj', 'nsubjpass']:
                                subjects.append(text)
                            elif role == 'dobj':
                                direct_objects.append(text)
                            elif role == 'iobj':
                                indirect_objects.append(text)
                            elif role in ['pobj', 'prep']:
                                prepositional_objects.append(text)

                        # Create valency pattern
                        pattern = {
                            'verb': verb,
                            'language': annotation_data.get('language', 'unknown'),
                            'valency_class': self._classify_valency(len(subjects), len(direct_objects), len(indirect_objects)),
                            'subjects': subjects,
                            'direct_objects': direct_objects,
                            'indirect_objects': indirect_objects,
                            'prepositional_objects': prepositional_objects,
                            'total_arguments': len(arguments),
                            'corpus_id': annotation_data.get('corpus_id'),
                            'sentence_context': annotation_data.get('original_text', '')[:200]
                        }

                        valency_patterns.append(pattern)

            # Fallback: Extract from dependency parsing
            elif 'analyses' in annotation_data and 'dependency_parsing' in annotation_data['analyses']:
                dep_data = annotation_data['analyses']['dependency_parsing']

                if 'dependencies' in dep_data:
                    # Group dependencies by verb
                    verb_deps = defaultdict(list)

                    for dep in dep_data['dependencies']:
                        if dep.get('pos') == 'VERB':
                            verb = dep['text']
                            verb_deps[verb].append(dep)

                    # Extract valency for each verb
                    for verb, deps in verb_deps.items():
                        subjects = []
                        objects = []

                        for dep in deps:
                            if dep.get('deprel') in ['nsubj', 'nsubjpass']:
                                subjects.append(dep.get('text', ''))
                            elif dep.get('deprel') in ['dobj', 'iobj']:
                                objects.append(dep.get('text', ''))

                        if subjects or objects:  # Only include verbs with arguments
                            pattern = {
                                'verb': verb,
                                'language': annotation_data.get('language', 'unknown'),
                                'valency_class': self._classify_valency(len(subjects), len(objects), 0),
                                'subjects': subjects,
                                'direct_objects': objects,
                                'indirect_objects': [],
                                'prepositional_objects': [],
                                'total_arguments': len(subjects) + len(objects),
                                'corpus_id': annotation_data.get('corpus_id'),
                                'extraction_method': 'dependency_parsing'
                            }
                            valency_patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Valency extraction failed: {e}")

        return valency_patterns

    def _classify_valency(self, n_subjects: int, n_direct_objects: int, n_indirect_objects: int) -> str:
        """Classify verb valency based on argument structure"""

        # Basic valency classification
        if n_subjects == 1 and n_direct_objects == 0 and n_indirect_objects == 0:
            return 'intransitive'
        elif n_subjects == 1 and n_direct_objects == 1 and n_indirect_objects == 0:
            return 'transitive'
        elif n_subjects == 1 and n_direct_objects == 1 and n_indirect_objects == 1:
            return 'ditransitive'
        elif n_subjects == 1 and n_direct_objects == 0 and n_indirect_objects == 1:
            return 'transitive_indirect'
        elif n_subjects == 0:
            return 'impersonal'
        else:
            return 'complex'

    def analyze_valency_batch(self, batch: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze valency patterns in a batch of annotated texts"""

        valency_data = []

        for item in batch:
            try:
                # Extract valency patterns
                patterns = self.extract_verb_valency(item)

                for pattern in patterns:
                    # Add etymological classification
                    verb = pattern['verb']
                    language = pattern.get('language', 'unknown')
                    etymology = self.classify_verb_etymology(verb, language)

                    pattern['etymology'] = etymology
                    pattern['extraction_timestamp'] = datetime.now().isoformat()

                    valency_data.append(pattern)

            except Exception as e:
                logger.error(f"Failed to analyze valency for item {item.get('corpus_id', 'unknown')}: {e}")
                continue

        return valency_data

    def aggregate_valency_statistics(self, valency_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate valency statistics across the corpus"""

        stats = {
            'total_verbs_analyzed': len(valency_data),
            'languages': defaultdict(int),
            'valency_classes': defaultdict(int),
            'etymology_distribution': defaultdict(int),
            'verb_frequencies': Counter(),
            'valency_patterns': defaultdict(lambda: defaultdict(int)),
            'native_vs_borrowed': {
                'native': defaultdict(lambda: defaultdict(int)),
                'borrowed': defaultdict(lambda: defaultdict(int))
            }
        }

        for item in valency_data:
            # Basic counts
            language = item.get('language', 'unknown')
            stats['languages'][language] += 1

            valency_class = item.get('valency_class', 'unknown')
            stats['valency_classes'][valency_class] += 1

            etymology = item.get('etymology', 'unknown')
            stats['etymology_distribution'][etymology] += 1

            # Verb frequencies
            verb = item.get('verb', '').lower()
            stats['verb_frequencies'][verb] += 1

            # Valency patterns by etymology
            if etymology in ['native', 'borrowed']:
                stats['native_vs_borrowed'][etymology][valency_class] += 1

            # Detailed patterns
            pattern_key = f"{language}_{etymology}_{valency_class}"
            stats['valency_patterns'][pattern_key][verb] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        stats['languages'] = dict(stats['languages'])
        stats['valency_classes'] = dict(stats['valency_classes'])
        stats['etymology_distribution'] = dict(stats['etymology_distribution'])
        stats['verb_frequencies'] = dict(stats['verb_frequencies'])

        stats['valency_patterns'] = {
            pattern: dict(verbs) for pattern, verbs in stats['valency_patterns'].items()
        }

        stats['native_vs_borrowed'] = {
            etymology: dict(classes) for etymology, classes in stats['native_vs_borrowed'].items()
        }

        return stats

    def generate_valency_report(self, stats: Dict[str, Any], language: str = 'all') -> str:
        """Generate a comprehensive valency report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HISTORICAL LINGUISTICS VALENCY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 20)
        report_lines.append(f"Total verbs analyzed: {stats['total_verbs_analyzed']}")
        report_lines.append(f"Languages covered: {', '.join(stats['languages'].keys())}")
        report_lines.append("")

        # Language breakdown
        report_lines.append("LANGUAGE BREAKDOWN")
        report_lines.append("-" * 20)
        for lang, count in sorted(stats['languages'].items()):
            report_lines.append(f"{lang.upper()}: {count} verbs")
        report_lines.append("")

        # Valency class distribution
        report_lines.append("VALENCY CLASS DISTRIBUTION")
        report_lines.append("-" * 30)
        for vclass, count in sorted(stats['valency_classes'].items()):
            percentage = (count / stats['total_verbs_analyzed']) * 100
            report_lines.append(".1f")
        report_lines.append("")

        # Etymology distribution
        report_lines.append("NATIVE VS BORROWED VERBS")
        report_lines.append("-" * 25)
        total_native_borrowed = stats['etymology_distribution'].get('native', 0) + stats['etymology_distribution'].get('borrowed', 0)
        if total_native_borrowed > 0:
            native_pct = (stats['etymology_distribution'].get('native', 0) / total_native_borrowed) * 100
            borrowed_pct = (stats['etymology_distribution'].get('borrowed', 0) / total_native_borrowed) * 100
            report_lines.append(".1f")
            report_lines.append(".1f")
        report_lines.append("")

        # Valency patterns by etymology
        report_lines.append("VALENCY PATTERNS: NATIVE VS BORROWED")
        report_lines.append("-" * 40)

        native_patterns = stats['native_vs_borrowed'].get('native', {})
        borrowed_patterns = stats['native_vs_borrowed'].get('borrowed', {})

        all_classes = set(list(native_patterns.keys()) + list(borrowed_patterns.keys()))

        for vclass in sorted(all_classes):
            native_count = native_patterns.get(vclass, 0)
            borrowed_count = borrowed_patterns.get(vclass, 0)
            total_class = native_count + borrowed_count

            if total_class > 0:
                report_lines.append(f"\n{vclass.upper()} VALENCY:")
                if native_count > 0:
                    native_pct = (native_count / total_class) * 100
                    report_lines.append(".1f")
                if borrowed_count > 0:
                    borrowed_pct = (borrowed_count / total_class) * 100
                    report_lines.append(".1f")
        report_lines.append("")

        # Most frequent verbs
        report_lines.append("MOST FREQUENT VERBS")
        report_lines.append("-" * 20)
        top_verbs = stats['verb_frequencies'].most_common(20)
        for verb, count in top_verbs:
            report_lines.append(f"{verb}: {count} occurrences")
        report_lines.append("")

        # Implications for historical linguistics
        report_lines.append("HISTORICAL LINGUISTICS IMPLICATIONS")
        report_lines.append("-" * 35)
        report_lines.append("This analysis reveals patterns in verb argument structures that can inform")
        report_lines.append("our understanding of language contact, grammaticalization, and semantic change.")
        report_lines.append("")

        # Compare native vs borrowed valency preferences
        native_transitive = native_patterns.get('transitive', 0)
        borrowed_transitive = borrowed_patterns.get('transitive', 0)

        if native_transitive > 0 and borrowed_transitive > 0:
            native_total = sum(native_patterns.values())
            borrowed_total = sum(borrowed_patterns.values())

            native_trans_pct = (native_transitive / native_total) * 100
            borrowed_trans_pct = (borrowed_transitive / borrowed_total) * 100

            report_lines.append("VALENCY PREFERENCE ANALYSIS:")
            report_lines.append(".1f")
            report_lines.append(".1f")

            if abs(native_trans_pct - borrowed_trans_pct) > 10:
                report_lines.append("SIGNIFICANT DIFFERENCE: Borrowed verbs show different valency preferences,")
                report_lines.append("potentially indicating substrate influence or calquing effects.")
            else:
                report_lines.append("SIMILAR PATTERNS: Native and borrowed verbs show comparable valency preferences.")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def create_comparative_lexica(self, valency_data: List[Dict[str, Any]], output_dir: str = "lexica"):
        """Create comparative valency lexica for native vs borrowed verbs"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Group by language and etymology
        lexica = defaultdict(lambda: defaultdict(list))

        for item in valency_data:
            language = item.get('language', 'unknown')
            etymology = item.get('etymology', 'unknown')
            verb = item.get('verb', '').lower()

            if verb and etymology in ['native', 'borrowed']:
                lexica[language][etymology].append(item)

        # Create lexicon files
        for language, etymology_groups in lexica.items():
            for etymology, verbs in etymology_groups.items():
                if len(verbs) < self.min_verb_occurrences:
                    continue

                # Aggregate valency patterns for this verb group
                verb_patterns = defaultdict(list)

                for verb_item in verbs:
                    verb = verb_item['verb'].lower()
                    valency_class = verb_item.get('valency_class', 'unknown')
                    verb_patterns[verb].append({
                        'valency_class': valency_class,
                        'arguments': verb_item.get('total_arguments', 0),
                        'frequency': 1
                    })

                # Create lexicon entries
                lexicon_entries = []

                for verb, patterns in verb_patterns.items():
                    if len(patterns) >= 3:  # At least 3 occurrences
                        # Aggregate patterns
                        pattern_counts = Counter(p['valency_class'] for p in patterns)
                        total_freq = len(patterns)

                        # Most common valency class
                        primary_valency = pattern_counts.most_common(1)[0][0]

                        # Average arguments
                        avg_args = sum(p['arguments'] for p in patterns) / len(patterns)

                        entry = {
                            'verb': verb,
                            'primary_valency': primary_valency,
                            'average_arguments': round(avg_args, 2),
                            'total_occurrences': total_freq,
                            'valency_distribution': dict(pattern_counts),
                            'language': language,
                            'etymology': etymology
                        }

                        lexicon_entries.append(entry)

                # Sort by frequency
                lexicon_entries.sort(key=lambda x: x['total_occurrences'], reverse=True)

                # Save lexicon
                if lexicon_entries:
                    filename = f"valency_lexicon_{language}_{etymology}_{datetime.now().strftime('%Y%m%d')}.json"
                    filepath = output_path / filename

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump({
                            'metadata': {
                                'language': language,
                                'etymology': etymology,
                                'total_entries': len(lexicon_entries),
                                'generation_date': datetime.now().isoformat(),
                                'min_occurrences': self.min_verb_occurrences
                            },
                            'entries': lexicon_entries
                        }, f, indent=2, ensure_ascii=False)

                    logger.info(f"Created comparative lexicon: {filename} ({len(lexicon_entries)} entries)")

    def run_valency_analysis_pipeline(self, limit=None, batch_size=None, output_dir="valency_reports"):
        """Run the complete valency analysis pipeline"""

        if batch_size:
            self.batch_size = batch_size

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        all_valency_data = []
        batch_count = 0

        logger.info(f"Starting valency analysis pipeline (batch_size={self.batch_size})")

        for batch in self._batch_load_annotated_texts(limit=limit):
            logger.info(f"Analyzing valency in batch {batch_count + 1} with {len(batch)} items")

            # Analyze valency in batch
            valency_batch = self.analyze_valency_batch(batch)
            all_valency_data.extend(valency_batch)

            batch_count += 1

        # Aggregate statistics
        logger.info(f"Aggregating statistics from {len(all_valency_data)} valency patterns")
        stats = self.aggregate_valency_statistics(all_valency_data)

        # Generate report
        report = self.generate_valency_report(stats)

        # Save report
        report_file = output_path / f"valency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Create comparative lexica
        self.create_comparative_lexica(all_valency_data, output_dir)

        # Save raw statistics
        stats_file = output_path / f"valency_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Valency analysis completed. Processed {len(all_valency_data)} patterns")
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Statistics saved to: {stats_file}")

        return {
            'total_patterns': len(all_valency_data),
            'report_file': str(report_file),
            'stats_file': str(stats_file),
            'stats': stats
        }

    def get_valency_stats(self) -> Dict[str, Any]:
        """Get current valency analysis statistics"""

        stats = {
            'min_verb_occurrences': self.min_verb_occurrences,
            'max_valency_patterns': self.max_valency_patterns,
            'supported_languages': list(self.verb_suffixes.keys())
        }

        # Check for existing reports
        reports_dir = Path("valency_reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("valency_report_*.txt"))
            stats['report_files'] = len(report_files)

            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                stats['latest_report'] = str(latest_report)

        return stats

def main():
    """Main valency analysis workflow"""
    analyzer = CostEffectiveValencyAnalyzer()

    print("Cost-Effective Valency Analysis System")
    print("=====================================")
    print(f"Minimum verb occurrences: {analyzer.min_verb_occurrences}")
    print(f"Supported languages: {', '.join(analyzer.verb_suffixes.keys())}")

    # Run valency analysis
    results = analyzer.run_valency_analysis_pipeline(limit=100)  # Analyze first 100 items

    print("
Valency analysis completed!")
    print(f"Total patterns analyzed: {results['total_patterns']}")
    print(f"Report saved to: {results['report_file']}")

if __name__ == "__main__":
    main()
