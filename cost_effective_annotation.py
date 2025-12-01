#!/usr/bin/env python3
"""
COST-EFFECTIVE TEXT ANNOTATION SYSTEM
Community-driven tools for linguistic annotation and markup
Optimized for low-cost servers with community-driven AIs

Uses free tools: spaCy, Stanza, TextBlob, NLTK, Polyglot
Focuses on semantic, syntactic, and morphological annotation
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Iterator, Optional, Tuple
import sqlite3
from datetime import datetime
import time

# Free community-driven annotation tools
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import stanza
    HAS_STANZA = True
except ImportError:
    HAS_STANZA = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import textblob
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import polyglot
    from polyglot.text import Text as PolyglotText
    HAS_POLYGLOT = True
except ImportError:
    HAS_POLYGLOT = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('annotation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostEffectiveAnnotator:
    """
    Memory-efficient annotation system using free community tools
    Provides comprehensive linguistic annotation for research
    """

    def __init__(self, db_path="corpus_efficient.db", batch_size=30):
        self.db_path = db_path
        self.batch_size = batch_size

        # Initialize annotation tools lazily
        self.annotation_tools = {}
        self._init_tools()

        # Annotation types
        self.annotation_types = [
            'semantic_roles',
            'sentiment_analysis',
            'semantic_similarity',
            'temporal_expressions',
            'coreference_resolution',
            'discourse_markers',
            'argument_structure'
        ]

        # Language support
        self.language_support = {
            'en': ['spacy', 'stanza', 'nltk', 'textblob', 'polyglot'],
            'de': ['spacy', 'stanza', 'polyglot'],
            'fr': ['spacy', 'stanza', 'polyglot'],
            'es': ['spacy', 'stanza', 'polyglot'],
            'grc': ['stanza', 'polyglot'],
            'la': ['stanza', 'polyglot']
        }

    def _init_tools(self):
        """Initialize annotation tools with memory-efficient configurations"""

        # spaCy - comprehensive annotation
        if HAS_SPACY:
            try:
                models_to_try = [
                    ('en_core_web_sm', 'en'),
                    ('de_core_news_sm', 'de'),
                    ('fr_core_news_sm', 'fr'),
                    ('es_core_news_sm', 'es')
                ]

                for model_name, lang in models_to_try:
                    try:
                        nlp = spacy.load(model_name, disable=['textcat'])
                        self.annotation_tools[f'spacy_{lang}'] = nlp
                        logger.info(f"spaCy annotation model loaded: {model_name}")
                    except OSError:
                        continue

            except Exception as e:
                logger.warning(f"spaCy annotation initialization failed: {e}")

        # Stanza - neural annotation
        if HAS_STANZA:
            try:
                supported_langs = ['en', 'de', 'fr', 'es', 'grc', 'la']
                for lang in supported_langs:
                    try:
                        pipeline = stanza.Pipeline(
                            lang=lang,
                            processors='tokenize,pos,lemma,depparse',
                            use_gpu=False,
                            verbose=False
                        )
                        self.annotation_tools[f'stanza_{lang}'] = pipeline
                        logger.info(f"Stanza annotation pipeline loaded: {lang}")
                    except Exception as e:
                        logger.warning(f"Stanza annotation {lang} failed: {e}")

            except Exception as e:
                logger.warning(f"Stanza annotation initialization failed: {e}")

        # NLTK - sentiment and semantic analysis
        if HAS_NLTK:
            try:
                # Download required data
                required_data = ['vader_lexicon', 'punkt', 'wordnet']
                for data in required_data:
                    try:
                        nltk.data.find(f'sentiment/{data}')
                    except LookupError:
                        logger.info(f"Downloading NLTK {data}...")
                        nltk.download(data, quiet=True)

                # Initialize sentiment analyzer
                self.annotation_tools['nltk_sentiment'] = SentimentIntensityAnalyzer()
                logger.info("NLTK sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"NLTK annotation initialization failed: {e}")

        # TextBlob - sentiment and semantic analysis
        if HAS_TEXTBLOB:
            self.annotation_tools['textblob'] = True
            logger.info("TextBlob annotation tools loaded")

        # Polyglot - multilingual semantic analysis
        if HAS_POLYGLOT:
            self.annotation_tools['polyglot'] = True
            logger.info("Polyglot annotation tools loaded")

    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _batch_load_parsed_texts(self, limit=None, offset=0) -> Iterator[List[Dict]]:
        """Load parsed texts in batches"""

        parsed_dir = Path("parsed")
        if not parsed_dir.exists():
            logger.warning("No parsed directory found")
            return

        json_files = list(parsed_dir.glob("*.jsonl.gz"))
        if not json_files:
            logger.warning("No parsed files found")
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

    def _select_best_annotator(self, language: str, task: str) -> str:
        """Select best available annotator for language and task"""

        if language not in self.language_support:
            language = 'en'

        available_annotators = self.language_support[language]
        annotator_priority = {
            'stanza': 1,      # Best for comprehensive annotation
            'spacy': 2,       # Fast and reliable
            'polyglot': 3,    # Good multilingual support
            'nltk': 4,        # Good for sentiment
            'textblob': 5     # Simple but effective
        }

        available_for_lang = [a for a in available_annotators if f"{a}_{language}" in self.annotation_tools or a in self.annotation_tools]

        if not available_for_lang:
            return 'textblob' if 'textblob' in self.annotation_tools else None

        return min(available_for_lang, key=lambda x: annotator_priority.get(x, 99))

    def semantic_role_labeling(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Perform semantic role labeling"""

        annotator = self._select_best_annotator(language, 'semantic_roles')
        if not annotator:
            return []

        try:
            if annotator == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.annotation_tools:
                    nlp = self.annotation_tools[spacy_key]
                    doc = nlp(text)

                    # Basic SRL using dependency parsing
                    roles = []
                    for token in doc:
                        if token.pos_ == 'VERB':
                            # Find subject and objects
                            subject = None
                            objects = []

                            for child in token.children:
                                if child.dep_ in ['nsubj', 'nsubjpass']:
                                    subject = child.text
                                elif child.dep_ in ['dobj', 'iobj', 'pobj']:
                                    objects.append(child.text)

                            if subject:
                                roles.append({
                                    'predicate': token.text,
                                    'agent': subject,
                                    'patients': objects,
                                    'type': 'basic_srl'
                                })

                    return roles

        except Exception as e:
            logger.warning(f"Semantic role labeling failed with {annotator}: {e}")

        return []

    def sentiment_analysis(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Perform sentiment analysis"""

        annotator = self._select_best_annotator(language, 'sentiment_analysis')

        try:
            if annotator == 'nltk' and language == 'en':
                analyzer = self.annotation_tools['nltk_sentiment']
                scores = analyzer.polarity_scores(text)

                return {
                    'method': 'nltk_vader',
                    'compound': scores['compound'],
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'sentiment': 'positive' if scores['compound'] > 0.05 else 'negative' if scores['compound'] < -0.05 else 'neutral'
                }

            elif annotator == 'textblob':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                return {
                    'method': 'textblob',
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment': 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
                }

            elif annotator == 'polyglot':
                if language in ['en', 'de', 'fr', 'es']:
                    ptext = PolyglotText(text, hint_language_code=language)
                    polarity = ptext.polarity

                    return {
                        'method': 'polyglot',
                        'polarity': polarity,
                        'sentiment': 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
                    }

        except Exception as e:
            logger.warning(f"Sentiment analysis failed with {annotator}: {e}")

        # Fallback
        return {
            'method': 'fallback',
            'sentiment': 'neutral',
            'confidence': 0.0
        }

    def temporal_expression_extraction(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Extract temporal expressions"""

        # Simple rule-based temporal expression extraction
        temporal_patterns = [
            # Dates
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',

            # Times
            r'\b\d{1,2}:\d{2}(?:\s?[ap]m)?\b',

            # Relative expressions
            r'\b(?:yesterday|today|tomorrow|last\s+week|next\s+month|this\s+year)\b',
            r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from\s+now)\b',

            # Ancient time periods (for historical texts)
            r'\b\d+(?:st|nd|rd|th)\s+century\b',
            r'\b\d+(?:st|nd|rd|th)\s+cent\.\s*bce?\b',
            r'\b\d+(?:st|nd|rd|th)\s+cent\.\s*ce\b'
        ]

        temporal_expressions = []

        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_expressions.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'temporal_expression',
                    'confidence': 0.8
                })

        # Remove duplicates based on position
        seen_positions = set()
        unique_expressions = []
        for expr in temporal_expressions:
            pos_key = (expr['start'], expr['end'])
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_expressions.append(expr)

        return unique_expressions

    def discourse_marker_identification(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Identify discourse markers and connectives"""

        # Language-specific discourse markers
        discourse_markers = {
            'en': [
                'however', 'therefore', 'moreover', 'furthermore', 'consequently',
                'nevertheless', 'although', 'whereas', 'because', 'since', 'while',
                'but', 'and', 'or', 'so', 'then', 'after', 'before', 'when', 'if'
            ],
            'de': [
                'aber', 'und', 'oder', 'denn', 'weil', 'da', 'obwohl', 'während',
                'also', 'deshalb', 'trotzdem', 'jedoch', 'außerdem', 'folglich'
            ],
            'fr': [
                'mais', 'et', 'ou', 'car', 'parce que', 'puisque', 'bien que',
                'pendant que', 'donc', 'par conséquent', 'cependant', 'toutefois'
            ]
        }

        markers = discourse_markers.get(language, discourse_markers['en'])
        identified_markers = []

        words = re.findall(r'\b\w+\b', text.lower())

        for i, word in enumerate(words):
            if word in markers:
                # Get original casing from text
                start_pos = text.lower().find(word, sum(len(w) + 1 for w in words[:i]))
                if start_pos != -1:
                    identified_markers.append({
                        'text': text[start_pos:start_pos + len(word)],
                        'lemma': word,
                        'position': i,
                        'type': 'discourse_marker',
                        'function': self._classify_discourse_marker(word, language)
                    })

        return identified_markers

    def _classify_discourse_marker(self, marker: str, language: str) -> str:
        """Classify the function of a discourse marker"""

        contrastive = {
            'en': ['but', 'however', 'although', 'whereas', 'nevertheless', 'yet'],
            'de': ['aber', 'jedoch', 'trotzdem', 'dennoch', 'hingegen'],
            'fr': ['mais', 'cependant', 'toutefois', 'néanmoins']
        }

        causal = {
            'en': ['because', 'since', 'therefore', 'consequently', 'so'],
            'de': ['weil', 'da', 'deshalb', 'also', 'folglich'],
            'fr': ['parce que', 'puisque', 'donc', 'par conséquent']
        }

        additive = {
            'en': ['and', 'also', 'moreover', 'furthermore', 'besides'],
            'de': ['und', 'außerdem', 'ebenso', 'ferner'],
            'fr': ['et', 'aussi', 'de plus', 'en outre']
        }

        lang_markers = contrastive.get(language, contrastive['en'])
        if marker in lang_markers:
            return 'contrastive'

        lang_markers = causal.get(language, causal['en'])
        if marker in lang_markers:
            return 'causal'

        lang_markers = additive.get(language, additive['en'])
        if marker in lang_markers:
            return 'additive'

        return 'other'

    def coreference_resolution(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Perform basic coreference resolution"""

        # Simple rule-based coreference for pronouns
        # This is a simplified implementation - real coreference is complex

        annotator = self._select_best_annotator(language, 'coreference_resolution')
        if not annotator:
            return []

        try:
            if annotator == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.annotation_tools:
                    nlp = self.annotation_tools[spacy_key]
                    doc = nlp(text)

                    # Simple pronoun resolution based on proximity
                    entities = []
                    pronouns = []

                    # Find named entities and pronouns
                    for token in doc:
                        if token.pos_ == 'PROPN' or (token.pos_ == 'NOUN' and token.ent_type_):
                            entities.append({
                                'text': token.text,
                                'type': 'entity',
                                'position': token.i
                            })
                        elif token.pos_ == 'PRON' and token.text.lower() in ['he', 'she', 'it', 'they', 'we', 'us']:
                            pronouns.append({
                                'text': token.text,
                                'type': 'pronoun',
                                'position': token.i
                            })

                    # Simple resolution: link pronouns to nearest preceding entity
                    resolutions = []
                    for pronoun in pronouns:
                        # Find closest preceding entity
                        candidates = [e for e in entities if e['position'] < pronoun['position']]
                        if candidates:
                            closest_entity = max(candidates, key=lambda x: x['position'])
                            resolutions.append({
                                'pronoun': pronoun['text'],
                                'pronoun_position': pronoun['position'],
                                'referent': closest_entity['text'],
                                'referent_position': closest_entity['position'],
                                'confidence': 0.6
                            })

                    return resolutions

        except Exception as e:
            logger.warning(f"Coreference resolution failed with {annotator}: {e}")

        return []

    def argument_structure_annotation(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Annotate argument structure based on parsed data"""

        arguments = []

        try:
            # Extract from dependency parsing if available
            if 'analyses' in parsed_data and 'dependency_parsing' in parsed_data['analyses']:
                dep_data = parsed_data['analyses']['dependency_parsing']

                if 'dependencies' in dep_data:
                    # Group by verbs and their arguments
                    verb_arguments = defaultdict(list)

                    for dep in dep_data['dependencies']:
                        if dep.get('pos') == 'VERB':
                            verb = dep['text']
                            # Find arguments based on dependency relations
                            if dep.get('dep') in ['nsubj', 'dobj', 'iobj']:
                                verb_arguments[verb].append({
                                    'role': dep['dep'],
                                    'text': dep['text'],
                                    'relation': dep.get('deprel', 'unknown')
                                })

                    # Convert to argument structures
                    for verb, args in verb_arguments.items():
                        arguments.append({
                            'predicate': verb,
                            'arguments': args,
                            'type': 'predicate_argument_structure'
                        })

        except Exception as e:
            logger.warning(f"Argument structure annotation failed: {e}")

        return arguments

    def annotate_text_comprehensive(self, text: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive annotation"""

        language = parsed_data.get('language', 'en')

        annotation = {
            'language': language,
            'original_text': text,
            'annotation_timestamp': datetime.now().isoformat(),
            'annotations': {}
        }

        # Semantic Role Labeling
        try:
            srl = self.semantic_role_labeling(text, language)
            annotation['annotations']['semantic_roles'] = srl
        except Exception as e:
            annotation['annotations']['semantic_roles'] = {'error': str(e)}

        # Sentiment Analysis
        try:
            sentiment = self.sentiment_analysis(text, language)
            annotation['annotations']['sentiment'] = sentiment
        except Exception as e:
            annotation['annotations']['sentiment'] = {'error': str(e)}

        # Temporal Expression Extraction
        try:
            temporal = self.temporal_expression_extraction(text, language)
            annotation['annotations']['temporal_expressions'] = temporal
        except Exception as e:
            annotation['annotations']['temporal_expressions'] = {'error': str(e)}

        # Discourse Marker Identification
        try:
            discourse = self.discourse_marker_identification(text, language)
            annotation['annotations']['discourse_markers'] = discourse
        except Exception as e:
            annotation['annotations']['discourse_markers'] = {'error': str(e)}

        # Coreference Resolution
        try:
            coreference = self.coreference_resolution(text, language)
            annotation['annotations']['coreference'] = coreference
        except Exception as e:
            annotation['annotations']['coreference'] = {'error': str(e)}

        # Argument Structure (based on parsed data)
        try:
            arguments = self.argument_structure_annotation(parsed_data)
            annotation['annotations']['argument_structure'] = arguments
        except Exception as e:
            annotation['annotations']['argument_structure'] = {'error': str(e)}

        return annotation

    def annotate_corpus_batch(self, batch: List[Dict]) -> List[Dict]:
        """Annotate a batch of parsed texts"""

        annotated_batch = []

        for item in batch:
            try:
                # Get original text (reconstruct from tokens if needed)
                if 'original_text' in item:
                    text = item['original_text']
                elif 'tokens' in item:
                    text = ' '.join(item['tokens'])
                else:
                    continue

                # Perform comprehensive annotation
                annotated = self.annotate_text_comprehensive(text, item)

                # Add corpus metadata
                annotated.update({
                    'corpus_id': item.get('corpus_id'),
                    'original_language': item.get('original_language'),
                    'quality_score': item.get('quality_score'),
                    'processing_timestamp': datetime.now().isoformat()
                })

                annotated_batch.append(annotated)

            except Exception as e:
                logger.error(f"Failed to annotate corpus item {item.get('corpus_id', 'unknown')}: {e}")
                continue

        return annotated_batch

    def save_annotated_batch(self, annotated_batch: List[Dict], output_dir: str = "annotated"):
        """Save annotated batch to disk"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        import gzip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotated_batch_{timestamp}.jsonl.gz"

        with gzip.open(output_path / filename, 'wt', encoding='utf-8') as f:
            for item in annotated_batch:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(annotated_batch)} annotated items to {filename}")
        return str(output_path / filename)

    def run_annotation_pipeline(self, limit=None, batch_size=None, output_dir="annotated"):
        """Run the complete annotation pipeline"""

        if batch_size:
            self.batch_size = batch_size

        total_annotated = 0
        batch_count = 0

        logger.info(f"Starting annotation pipeline (batch_size={self.batch_size})")

        for batch in self._batch_load_parsed_texts(limit=limit):
            logger.info(f"Annotating batch {batch_count + 1} with {len(batch)} items")

            # Annotate batch
            annotated_batch = self.annotate_corpus_batch(batch)

            if annotated_batch:
                # Save results
                self.save_annotated_batch(annotated_batch, output_dir)
                total_annotated += len(annotated_batch)

            batch_count += 1

            # Progress logging
            if batch_count % 5 == 0:
                logger.info(f"Annotated {total_annotated} items in {batch_count} batches")

        logger.info(f"Annotation pipeline completed. Total annotated: {total_annotated}")
        return total_annotated

    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get annotation statistics"""

        stats = {
            'available_tools': list(self.annotation_tools.keys()),
            'supported_languages': list(self.language_support.keys()),
            'annotation_types': self.annotation_types,
            'batch_size': self.batch_size
        }

        # Count annotated files
        annotated_dir = Path("annotated")
        if annotated_dir.exists():
            annotated_files = list(annotated_dir.glob("*.jsonl.gz"))
            stats['annotated_files'] = len(annotated_files)
            stats['total_annotated_items'] = 0

            import gzip
            for file_path in annotated_files:
                try:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        stats['total_annotated_items'] += sum(1 for _ in f)
                except:
                    continue

        return stats

def main():
    """Main annotation workflow"""
    annotator = CostEffectiveAnnotator()

    print("Available Annotation Tools:")
    for tool in annotator.annotation_tools.keys():
        print(f"  - {tool}")

    print(f"\nSupported Languages: {', '.join(annotator.language_support.keys())}")

    # Run annotation pipeline
    stats = annotator.run_annotation_pipeline(limit=50)  # Annotate first 50 items

    print(f"\nAnnotation completed: {stats} items processed")

if __name__ == "__main__":
    main()
