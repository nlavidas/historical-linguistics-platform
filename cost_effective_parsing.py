#!/usr/bin/env python3
"""
COST-EFFECTIVE TEXT PARSING SYSTEM
Community-driven tools for linguistic parsing and analysis
Optimized for low-cost servers with community-driven AIs

Uses free tools: spaCy, Stanza, UDPipe, Trankit, NLTK
Focuses on dependency parsing, POS tagging, morphological analysis
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

# Free community-driven parsing tools
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
    from nltk import pos_tag, ne_chunk
    from nltk.tree import Tree
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import trankit
    HAS_TRANKIT = True
except ImportError:
    HAS_TRANKIT = False

try:
    import udpipe
    HAS_UDPIPE = True
except ImportError:
    HAS_UDPIPE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parsing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostEffectiveParser:
    """
    Memory-efficient parsing system using free community tools
    Focuses on syntactic and morphological analysis for linguistic research
    """

    def __init__(self, db_path="corpus_efficient.db", batch_size=50):
        self.db_path = db_path
        self.batch_size = batch_size

        # Initialize parsing tools lazily
        self.parsing_tools = {}
        self._init_tools()

        # Supported parsing tasks
        self.parsing_tasks = [
            'pos_tagging',
            'dependency_parsing',
            'morphological_analysis',
            'constituent_parsing',
            'named_entity_recognition'
        ]

        # Language support matrix
        self.language_support = {
            'en': ['spacy', 'stanza', 'nltk', 'trankit'],
            'de': ['spacy', 'stanza', 'trankit'],
            'fr': ['spacy', 'stanza', 'trankit'],
            'es': ['spacy', 'stanza', 'trankit'],
            'grc': ['stanza', 'trankit', 'udpipe'],
            'la': ['stanza', 'trankit', 'udpipe']
        }

    def _init_tools(self):
        """Initialize parsing tools with memory-efficient configurations"""

        # spaCy - industrial-strength parsing
        if HAS_SPACY:
            try:
                # Load small models to save memory
                models_to_try = [
                    ('en_core_web_sm', 'en'),
                    ('de_core_news_sm', 'de'),
                    ('fr_core_news_sm', 'fr'),
                    ('es_core_news_sm', 'es')
                ]

                for model_name, lang in models_to_try:
                    try:
                        # Load with minimal components to save memory
                        nlp = spacy.load(model_name, disable=['lemmatizer', 'textcat'])
                        self.parsing_tools[f'spacy_{lang}'] = nlp
                        logger.info(f"spaCy model loaded: {model_name}")
                    except OSError:
                        continue

            except Exception as e:
                logger.warning(f"spaCy initialization failed: {e}")

        # Stanza - Stanford's neural parsing (good for ancient languages)
        if HAS_STANZA:
            try:
                # Initialize pipelines for supported languages
                supported_langs = ['en', 'de', 'fr', 'es', 'grc', 'la']
                for lang in supported_langs:
                    try:
                        # Use CPU and minimal processors to save memory
                        pipeline = stanza.Pipeline(
                            lang=lang,
                            processors='tokenize,pos,lemma,depparse',
                            use_gpu=False,
                            verbose=False
                        )
                        self.parsing_tools[f'stanza_{lang}'] = pipeline
                        logger.info(f"Stanza pipeline loaded: {lang}")
                    except Exception as e:
                        logger.warning(f"Stanza {lang} failed: {e}")

            except Exception as e:
                logger.warning(f"Stanza initialization failed: {e}")

        # Trankit - multilingual neural parsing
        if HAS_TRANKIT:
            try:
                # Initialize with supported languages
                trankit_langs = ['english', 'german', 'french', 'spanish', 'classical-greek', 'latin']
                pipeline = trankit.Pipeline(lang='auto')  # Auto-detect language
                self.parsing_tools['trankit'] = pipeline
                logger.info("Trankit pipeline loaded")
            except Exception as e:
                logger.warning(f"Trankit initialization failed: {e}")

        # UDPipe - neural parsing (good for low-resource languages)
        if HAS_UDPIPE:
            try:
                # UDPipe models would need to be downloaded separately
                # For now, mark as available for future use
                self.parsing_tools['udpipe'] = True
                logger.info("UDPipe marked as available")
            except Exception as e:
                logger.warning(f"UDPipe initialization failed: {e}")

        # NLTK - traditional parsing (English-focused)
        if HAS_NLTK:
            try:
                # Ensure required data is available
                required_data = ['averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
                for data in required_data:
                    try:
                        nltk.data.find(f'taggers/{data}')
                    except LookupError:
                        logger.info(f"Downloading NLTK {data}...")
                        nltk.download(data, quiet=True)

                self.parsing_tools['nltk'] = True
                logger.info("NLTK parsing tools loaded")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")

    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _select_best_parser(self, language: str, task: str) -> str:
        """Select the best available parser for language and task"""

        if language not in self.language_support:
            language = 'en'  # Default fallback

        available_parsers = self.language_support[language]
        parser_priority = {
            'stanza': 1,  # Best for accuracy and ancient languages
            'spacy': 2,   # Fast and reliable
            'trankit': 3, # Good multilingual support
            'udpipe': 4,  # Good for low-resource languages
            'nltk': 5     # Basic but always available
        }

        # Find available parsers for this language
        available_for_lang = [p for p in available_parsers if f"{p}_{language}" in self.parsing_tools or p in self.parsing_tools]

        if not available_for_lang:
            return 'nltk' if 'nltk' in self.parsing_tools else None

        # Return highest priority available parser
        return min(available_for_lang, key=lambda x: parser_priority.get(x, 99))

    def _batch_load_preprocessed_texts(self, limit=None, offset=0) -> Iterator[List[Dict]]:
        """Load preprocessed texts in batches"""

        # Look for preprocessed files
        preprocessed_dir = Path("preprocessed")
        if not preprocessed_dir.exists():
            logger.warning("No preprocessed directory found")
            return

        # Find most recent preprocessed file
        json_files = list(preprocessed_dir.glob("*.jsonl.gz"))
        if not json_files:
            logger.warning("No preprocessed files found")
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

    def pos_tag_text(self, tokens: List[str], language: str = 'en') -> List[Tuple[str, str]]:
        """POS tag tokens using best available tool"""

        parser = self._select_best_parser(language, 'pos_tagging')
        if not parser:
            return [(token, 'UNK') for token in tokens]

        try:
            if parser == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.parsing_tools:
                    nlp = self.parsing_tools[spacy_key]
                    doc = nlp(' '.join(tokens))
                    return [(token.text, token.pos_) for token in doc]

            elif parser == 'stanza':
                stanza_key = f"stanza_{language}"
                if stanza_key in self.parsing_tools:
                    pipeline = self.parsing_tools[stanza_key]
                    doc = pipeline(' '.join(tokens))
                    return [(word.text, word.pos) for sent in doc.sentences for word in sent.words]

            elif parser == 'nltk':
                # Basic POS tagging
                return pos_tag(tokens)

            elif parser == 'trankit':
                pipeline = self.parsing_tools['trankit']
                result = pipeline.pos(' '.join(tokens), lang=language)
                return [(token['text'], token.get('pos', 'UNK')) for token in result[0]]

        except Exception as e:
            logger.warning(f"POS tagging failed with {parser}: {e}")

        # Fallback
        return [(token, 'UNK') for token in tokens]

    def dependency_parse(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Perform dependency parsing"""

        parser = self._select_best_parser(language, 'dependency_parsing')
        if not parser:
            return {'error': 'No suitable parser available'}

        try:
            if parser == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.parsing_tools:
                    nlp = self.parsing_tools[spacy_key]
                    doc = nlp(text)

                    dependencies = []
                    for token in doc:
                        dependencies.append({
                            'text': token.text,
                            'pos': token.pos_,
                            'head': token.head.text if token.head != token else 'ROOT',
                            'dep': token.dep_,
                            'children': [child.text for child in token.children]
                        })

                    return {
                        'parser': 'spacy',
                        'language': language,
                        'dependencies': dependencies
                    }

            elif parser == 'stanza':
                stanza_key = f"stanza_{language}"
                if stanza_key in self.parsing_tools:
                    pipeline = self.parsing_tools[stanza_key]
                    doc = pipeline(text)

                    dependencies = []
                    for sentence in doc.sentences:
                        for word in sentence.words:
                            dependencies.append({
                                'text': word.text,
                                'pos': word.pos,
                                'head': word.head,
                                'deprel': word.deprel,
                                'lemma': word.lemma
                            })

                    return {
                        'parser': 'stanza',
                        'language': language,
                        'dependencies': dependencies,
                        'sentences': len(doc.sentences)
                    }

            elif parser == 'trankit':
                pipeline = self.parsing_tools['trankit']
                result = pipeline.parse(text, lang=language)

                return {
                    'parser': 'trankit',
                    'language': language,
                    'result': result
                }

        except Exception as e:
            logger.warning(f"Dependency parsing failed with {parser}: {e}")
            return {'error': str(e)}

    def morphological_analysis(self, tokens: List[str], language: str = 'en') -> List[Dict[str, Any]]:
        """Perform morphological analysis"""

        parser = self._select_best_parser(language, 'morphological_analysis')
        if not parser:
            return [{'token': token, 'features': {}} for token in tokens]

        try:
            if parser == 'stanza':
                stanza_key = f"stanza_{language}"
                if stanza_key in self.parsing_tools:
                    pipeline = self.parsing_tools[stanza_key]
                    doc = pipeline(' '.join(tokens))

                    morphology = []
                    for sentence in doc.sentences:
                        for word in sentence.words:
                            morph_features = {}
                            if hasattr(word, 'feats') and word.feats:
                                # Parse morphological features
                                for feat in word.feats.split('|'):
                                    if '=' in feat:
                                        key, value = feat.split('=', 1)
                                        morph_features[key] = value

                            morphology.append({
                                'token': word.text,
                                'lemma': word.lemma,
                                'pos': word.pos,
                                'morphological_features': morph_features
                            })

                    return morphology

            elif parser == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.parsing_tools:
                    nlp = self.parsing_tools[spacy_key]
                    doc = nlp(' '.join(tokens))

                    morphology = []
                    for token in doc:
                        morph_features = {}
                        # spaCy morphological features (limited for free models)
                        if token.pos_ in ['VERB', 'NOUN', 'ADJ']:
                            morph_features['pos'] = token.pos_

                        morphology.append({
                            'token': token.text,
                            'lemma': token.lemma_,
                            'pos': token.pos_,
                            'morphological_features': morph_features
                        })

                    return morphology

        except Exception as e:
            logger.warning(f"Morphological analysis failed with {parser}: {e}")

        # Fallback
        return [{'token': token, 'features': {}} for token in tokens]

    def named_entity_recognition(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Extract named entities"""

        parser = self._select_best_parser(language, 'named_entity_recognition')
        if not parser:
            return []

        try:
            if parser == 'spacy':
                spacy_key = f"spacy_{language}"
                if spacy_key in self.parsing_tools:
                    nlp = self.parsing_tools[spacy_key]
                    doc = nlp(text)

                    entities = []
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': getattr(ent, '_.confidence', 1.0)
                        })

                    return entities

            elif parser == 'nltk' and language == 'en':
                # Basic NER using NLTK
                tokens = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)
                tree = ne_chunk(pos_tags)

                entities = []
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity_text = ' '.join([token for token, pos in subtree.leaves()])
                        entities.append({
                            'text': entity_text,
                            'label': subtree.label(),
                            'start': text.find(entity_text),
                            'end': text.find(entity_text) + len(entity_text),
                            'confidence': 0.8
                        })

                return entities

        except Exception as e:
            logger.warning(f"NER failed with {parser}: {e}")

        return []

    def parse_text_comprehensive(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Perform comprehensive parsing analysis"""

        result = {
            'language': language,
            'original_text': text,
            'parsing_timestamp': datetime.now().isoformat(),
            'analyses': {}
        }

        # Tokenization (basic preprocessing step)
        try:
            if HAS_NLTK:
                tokens = nltk.word_tokenize(text)
            else:
                tokens = text.split()
        except:
            tokens = text.split()

        result['tokens'] = tokens
        result['token_count'] = len(tokens)

        # POS Tagging
        try:
            pos_tags = self.pos_tag_text(tokens, language)
            result['analyses']['pos_tagging'] = {
                'method': self._select_best_parser(language, 'pos_tagging'),
                'tags': pos_tags
            }
        except Exception as e:
            result['analyses']['pos_tagging'] = {'error': str(e)}

        # Dependency Parsing
        try:
            dep_parse = self.dependency_parse(text, language)
            result['analyses']['dependency_parsing'] = dep_parse
        except Exception as e:
            result['analyses']['dependency_parsing'] = {'error': str(e)}

        # Morphological Analysis
        try:
            morphology = self.morphological_analysis(tokens, language)
            result['analyses']['morphological_analysis'] = morphology
        except Exception as e:
            result['analyses']['morphological_analysis'] = {'error': str(e)}

        # Named Entity Recognition
        try:
            entities = self.named_entity_recognition(text, language)
            result['analyses']['named_entities'] = entities
        except Exception as e:
            result['analyses']['named_entities'] = {'error': str(e)}

        return result

    def parse_corpus_batch(self, batch: List[Dict]) -> List[Dict]:
        """Parse a batch of preprocessed texts"""

        parsed_batch = []

        for item in batch:
            try:
                # Skip if no tokens available
                if 'tokens' not in item or not item['tokens']:
                    continue

                # Perform comprehensive parsing
                parsed = self.parse_text_comprehensive(
                    ' '.join(item['tokens']),
                    item.get('language', 'en')
                )

                # Add corpus metadata
                parsed.update({
                    'corpus_id': item.get('corpus_id'),
                    'original_language': item.get('original_language'),
                    'quality_score': item.get('quality_score'),
                    'processing_timestamp': datetime.now().isoformat()
                })

                parsed_batch.append(parsed)

            except Exception as e:
                logger.error(f"Failed to parse corpus item {item.get('corpus_id', 'unknown')}: {e}")
                continue

        return parsed_batch

    def save_parsed_batch(self, parsed_batch: List[Dict], output_dir: str = "parsed"):
        """Save parsed batch to disk"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save as compressed JSON lines
        import gzip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parsed_batch_{timestamp}.jsonl.gz"

        with gzip.open(output_path / filename, 'wt', encoding='utf-8') as f:
            for item in parsed_batch:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(parsed_batch)} parsed items to {filename}")
        return str(output_path / filename)

    def run_parsing_pipeline(self, limit=None, batch_size=None, output_dir="parsed"):
        """Run the complete parsing pipeline"""

        if batch_size:
            self.batch_size = batch_size

        total_parsed = 0
        batch_count = 0

        logger.info(f"Starting parsing pipeline (batch_size={self.batch_size})")

        for batch in self._batch_load_preprocessed_texts(limit=limit):
            logger.info(f"Parsing batch {batch_count + 1} with {len(batch)} items")

            # Parse batch
            parsed_batch = self.parse_corpus_batch(batch)

            if parsed_batch:
                # Save results
                self.save_parsed_batch(parsed_batch, output_dir)
                total_parsed += len(parsed_batch)

            batch_count += 1

            # Progress logging
            if batch_count % 5 == 0:
                logger.info(f"Parsed {total_parsed} items in {batch_count} batches")

        logger.info(f"Parsing pipeline completed. Total parsed: {total_parsed}")
        return total_parsed

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get parsing statistics and available tools"""

        stats = {
            'available_tools': list(self.parsing_tools.keys()),
            'supported_languages': list(self.language_support.keys()),
            'parsing_tasks': self.parsing_tasks,
            'batch_size': self.batch_size
        }

        # Count parsed files if they exist
        parsed_dir = Path("parsed")
        if parsed_dir.exists():
            parsed_files = list(parsed_dir.glob("*.jsonl.gz"))
            stats['parsed_files'] = len(parsed_files)
            stats['total_parsed_items'] = 0

            import gzip
            for file_path in parsed_files:
                try:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        stats['total_parsed_items'] += sum(1 for _ in f)
                except:
                    continue

        return stats

def main():
    """Main parsing workflow"""
    parser = CostEffectiveParser()

    # Print available tools
    print("Available Parsing Tools:")
    for tool in parser.parsing_tools.keys():
        print(f"  - {tool}")

    print(f"\nSupported Languages: {', '.join(parser.language_support.keys())}")

    # Run parsing pipeline
    stats = parser.run_parsing_pipeline(limit=100)  # Parse first 100 items

    print(f"\nParsing completed: {stats} items processed")

if __name__ == "__main__":
    main()
