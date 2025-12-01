#!/usr/bin/env python3
"""
COST-EFFECTIVE TEXT PREPROCESSING SYSTEM
Community-driven tools for linguistic text preparation
Optimized for low-cost servers with community-driven AIs

Uses free tools: NLTK, spaCy (free models), TextBlob, Polyglot
Memory-efficient batch processing, streaming for large corpora
"""

import os
import sys
import json
import logging
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Iterator, Optional, Tuple
import sqlite3
from datetime import datetime
import time

# Free community-driven NLP tools
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

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
        logging.FileHandler('preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostEffectivePreprocessor:
    """
    Memory-efficient text preprocessing using free community tools
    Designed for low-cost servers with limited RAM
    """

    def __init__(self, db_path="corpus_efficient.db", batch_size=100, use_gpu=False):
        self.db_path = db_path
        self.batch_size = batch_size
        self.use_gpu = use_gpu and self._check_gpu_availability()

        # Initialize tools lazily to save memory
        self.nlp_tools = {}
        self._init_tools()

        # Language-specific configurations
        self.lang_configs = {
            'grc': {  # Ancient Greek
                'script': 'greek',
                'has_spaces': True,
                'direction': 'ltr'
            },
            'la': {   # Latin
                'script': 'latin',
                'has_spaces': True,
                'direction': 'ltr'
            },
            'en': {   # English
                'script': 'latin',
                'has_spaces': True,
                'direction': 'ltr'
            },
            'de': {   # German
                'script': 'latin',
                'has_spaces': True,
                'direction': 'ltr'
            },
            'fr': {   # French
                'script': 'latin',
                'has_spaces': True,
                'direction': 'ltr'
            }
        }

        # Preprocessing pipeline configuration
        self.pipeline_steps = [
            'normalize_unicode',
            'remove_markup',
            'normalize_whitespace',
            'language_detection',
            'sentence_segmentation',
            'tokenization',
            'lowercasing',
            'remove_punctuation',
            'remove_numbers',
            'remove_stopwords',
            'lemmatization',
            'filter_short_tokens'
        ]

    def _check_gpu_availability(self):
        """Check if GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _init_tools(self):
        """Initialize NLP tools lazily"""

        # NLTK - most comprehensive free toolkit
        if HAS_NLTK:
            try:
                # Download required NLTK data
                required_nltk = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
                for package in required_nltk:
                    try:
                        nltk.data.find(f'tokenizers/{package}')
                    except LookupError:
                        logger.info(f"Downloading NLTK {package}...")
                        nltk.download(package, quiet=True)

                self.nlp_tools['nltk'] = {
                    'lemmatizer': WordNetLemmatizer(),
                    'stopwords': set(stopwords.words('english'))
                }
                logger.info("NLTK initialized")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")

        # spaCy - industrial-strength NLP (free models)
        if HAS_SPACY:
            try:
                # Use small free models to save memory
                models_to_try = ['en_core_web_sm', 'de_core_news_sm']

                for model_name in models_to_try:
                    try:
                        self.nlp_tools[f'spacy_{model_name}'] = spacy.load(model_name, disable=['parser', 'ner'])
                        logger.info(f"spaCy model loaded: {model_name}")
                        break
                    except OSError:
                        continue

                if not any(k.startswith('spacy_') for k in self.nlp_tools.keys()):
                    logger.warning("No spaCy models available")

            except Exception as e:
                logger.warning(f"spaCy initialization failed: {e}")

        # TextBlob - simple but effective
        if HAS_TEXTBLOB:
            self.nlp_tools['textblob'] = True
            logger.info("TextBlob initialized")

        # Polyglot - multilingual support
        if HAS_POLYGLOT:
            self.nlp_tools['polyglot'] = True
            logger.info("Polyglot initialized")

    def _get_db_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _batch_process_texts(self, limit=None, offset=0) -> Iterator[List[Dict]]:
        """Stream texts from database in batches to save memory"""

        conn = self._get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT id, content_compressed, language, quality_score
            FROM corpus_items
            WHERE duplicate_of IS NULL
            AND quality_score > 0.4
            ORDER BY id
        """

        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        cursor.execute(query)

        batch = []
        for row in cursor:
            batch.append(dict(row))

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

        conn.close()

    def _decompress_text(self, compressed_data):
        """Decompress text data"""
        import gzip
        return gzip.decompress(compressed_data).decode('utf-8')

    def normalize_unicode(self, text: str, lang: str = 'en') -> str:
        """Normalize unicode characters for consistency"""
        try:
            # Normalize to NFC form (recommended for most languages)
            text = unicodedata.normalize('NFC', text)

            # Language-specific normalizations
            if lang in ['grc', 'el']:
                # Greek-specific normalizations
                text = re.sub(r'᾽', "'", text)  # Normalize apostrophes
                text = re.sub(r'᾿', "'", text)

            elif lang == 'la':
                # Latin-specific normalizations
                text = re.sub(r'ā', 'a', text)  # Remove macrons for basic processing
                text = re.sub(r'ē', 'e', text)
                text = re.sub(r'ī', 'i', text)
                text = re.sub(r'ō', 'o', text)
                text = re.sub(r'ū', 'u', text)

            return text

        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text

    def remove_markup(self, text: str) -> str:
        """Remove HTML/XML markup and formatting"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove XML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace various whitespace with single spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def detect_language(self, text: str) -> str:
        """Detect language using free tools"""
        # Try polyglot first (good for many languages)
        if HAS_POLYGLOT and 'polyglot' in self.nlp_tools:
            try:
                detector = PolyglotText(text[:1000]).language
                return detector.code if hasattr(detector, 'code') else 'unknown'
            except:
                pass

        # Fallback to simple heuristics
        text_lower = text.lower()

        # Greek detection
        greek_chars = sum(1 for char in text_lower if '\u0370' <= char <= '\u03ff' or '\u1f00' <= char <= '\u1fff')
        if greek_chars > len(text) * 0.1:
            return 'grc' if any(word in text_lower for word in ['εἰμί', 'λέγω', 'φημί']) else 'el'

        # Latin detection
        latin_chars = sum(1 for char in text_lower if '\u0041' <= char <= '\u007a')
        if latin_chars > len(text) * 0.8:
            # Check for Latin-specific patterns
            if any(char in text_lower for char in ['ā', 'ē', 'ī', 'ō', 'ū']):
                return 'la'
            return 'en'

        return 'unknown'

    def segment_sentences(self, text: str, lang: str = 'en') -> List[str]:
        """Segment text into sentences"""
        try:
            if HAS_NLTK and 'nltk' in self.nlp_tools:
                return sent_tokenize(text, language=lang if lang in ['english', 'portuguese', 'spanish', 'german'] else 'english')
            else:
                # Simple sentence segmentation
                sentences = re.split(r'[.!?]+\s+', text)
                return [s.strip() for s in sentences if s.strip()]

        except Exception as e:
            logger.warning(f"Sentence segmentation failed: {e}")
            # Very basic fallback
            return text.split('\n')

    def tokenize_text(self, text: str, lang: str = 'en') -> List[str]:
        """Tokenize text into words"""
        try:
            if HAS_NLTK and 'nltk' in self.nlp_tools:
                # Map language codes to NLTK language names
                nltk_lang_map = {
                    'en': 'english',
                    'de': 'german',
                    'fr': 'french',
                    'es': 'spanish',
                    'pt': 'portuguese'
                }
                nltk_lang = nltk_lang_map.get(lang, 'english')
                return word_tokenize(text, language=nltk_lang)

            elif HAS_SPACY and any(k.startswith('spacy_') for k in self.nlp_tools.keys()):
                # Use available spaCy model
                spacy_key = next(k for k in self.nlp_tools.keys() if k.startswith('spacy_'))
                nlp = self.nlp_tools[spacy_key]
                doc = nlp(text)
                return [token.text for token in doc]

            else:
                # Basic whitespace tokenization
                return text.split()

        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return text.split()

    def lowercase_text(self, tokens: List[str], lang: str = 'en') -> List[str]:
        """Convert tokens to lowercase"""
        # Be careful with languages that use casing for meaning
        if lang in ['de', 'en']:  # Languages where casing is important
            return [token.lower() for token in tokens]
        else:
            # For other languages, preserve original casing
            return tokens

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation from tokens"""
        # Keep only alphanumeric characters and apostrophes
        cleaned = []
        for token in tokens:
            # Remove leading/trailing punctuation
            token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            if token and len(token) > 1:  # Keep single chars that are alphanumeric
                cleaned.append(token)
        return cleaned

    def remove_numbers(self, tokens: List[str]) -> List[str]:
        """Remove numeric tokens"""
        return [token for token in tokens if not re.match(r'^\d+(\.\d+)?$', token)]

    def remove_stopwords(self, tokens: List[str], lang: str = 'en') -> List[str]:
        """Remove stopwords using available tools"""
        try:
            if HAS_NLTK and 'nltk' in self.nlp_tools and lang == 'en':
                stopwords_set = self.nlp_tools['nltk']['stopwords']
                return [token for token in tokens if token.lower() not in stopwords_set]

            elif HAS_SPACY and any(k.startswith('spacy_') for k in self.nlp_tools.keys()):
                spacy_key = next(k for k in self.nlp_tools.keys() if k.startswith('spacy_'))
                nlp = self.nlp_tools[spacy_key]
                return [token for token in tokens if not nlp.vocab[token].is_stop]

            else:
                # Basic stopword removal (English only)
                basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                return [token for token in tokens if token.lower() not in basic_stopwords]

        except Exception as e:
            logger.warning(f"Stopword removal failed: {e}")
            return tokens

    def lemmatize_tokens(self, tokens: List[str], lang: str = 'en') -> List[str]:
        """Lemmatize tokens to base forms"""
        try:
            if HAS_NLTK and 'nltk' in self.nlp_tools and lang == 'en':
                lemmatizer = self.nlp_tools['nltk']['lemmatizer']
                return [lemmatizer.lemmatize(token) for token in tokens]

            elif HAS_SPACY and any(k.startswith('spacy_') for k in self.nlp_tools.keys()):
                spacy_key = next(k for k in self.nlp_tools.keys() if k.startswith('spacy_'))
                nlp = self.nlp_tools[spacy_key]
                doc = nlp(' '.join(tokens))
                return [token.lemma_ for token in doc]

            elif HAS_TEXTBLOB and 'textblob' in self.nlp_tools and lang == 'en':
                # TextBlob lemmatization is limited but free
                blob = TextBlob(' '.join(tokens))
                return [word.lemmatize() for word in blob.words]

            else:
                # No lemmatization available
                return tokens

        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return tokens

    def filter_short_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """Filter out very short tokens"""
        return [token for token in tokens if len(token) >= min_length]

    def preprocess_text(self, text: str, lang: str = None, pipeline: List[str] = None) -> Dict[str, Any]:
        """Run full preprocessing pipeline on a single text"""

        if pipeline is None:
            pipeline = self.pipeline_steps

        result = {
            'original_text': text,
            'processed_text': text,
            'tokens': [],
            'sentences': [],
            'language': lang or self.detect_language(text),
            'processing_steps': []
        }

        current_text = text
        current_tokens = []

        for step in pipeline:
            try:
                if step == 'normalize_unicode':
                    current_text = self.normalize_unicode(current_text, result['language'])
                elif step == 'remove_markup':
                    current_text = self.remove_markup(current_text)
                elif step == 'normalize_whitespace':
                    current_text = self.normalize_whitespace(current_text)
                elif step == 'language_detection':
                    result['language'] = self.detect_language(current_text)
                elif step == 'sentence_segmentation':
                    result['sentences'] = self.segment_sentences(current_text, result['language'])
                elif step == 'tokenization':
                    current_tokens = self.tokenize_text(current_text, result['language'])
                elif step == 'lowercasing':
                    current_tokens = self.lowercase_text(current_tokens, result['language'])
                elif step == 'remove_punctuation':
                    current_tokens = self.remove_punctuation(current_tokens)
                elif step == 'remove_numbers':
                    current_tokens = self.remove_numbers(current_tokens)
                elif step == 'remove_stopwords':
                    current_tokens = self.remove_stopwords(current_tokens, result['language'])
                elif step == 'lemmatization':
                    current_tokens = self.lemmatize_tokens(current_tokens, result['language'])
                elif step == 'filter_short_tokens':
                    current_tokens = self.filter_short_tokens(current_tokens)

                result['processing_steps'].append(f"{step}: completed")

            except Exception as e:
                logger.warning(f"Preprocessing step {step} failed: {e}")
                result['processing_steps'].append(f"{step}: failed ({e})")

        result['processed_text'] = current_text
        result['tokens'] = current_tokens
        result['token_count'] = len(current_tokens)

        return result

    def preprocess_corpus_batch(self, batch: List[Dict]) -> List[Dict]:
        """Preprocess a batch of texts from the corpus"""
        processed_batch = []

        for item in batch:
            try:
                # Decompress text
                text = self._decompress_text(item['content_compressed'])

                # Skip if too short or low quality
                if len(text) < 100 or item.get('quality_score', 0) < 0.3:
                    continue

                # Preprocess
                processed = self.preprocess_text(text, item.get('language', 'en'))

                # Add metadata
                processed.update({
                    'corpus_id': item['id'],
                    'original_language': item.get('language'),
                    'quality_score': item.get('quality_score'),
                    'processing_timestamp': datetime.now().isoformat()
                })

                processed_batch.append(processed)

            except Exception as e:
                logger.error(f"Failed to preprocess corpus item {item['id']}: {e}")
                continue

        return processed_batch

    def save_preprocessed_batch(self, processed_batch: List[Dict], output_dir: str = "preprocessed"):
        """Save preprocessed batch to disk efficiently"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save as compressed JSON lines
        import gzip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"preprocessed_batch_{timestamp}.jsonl.gz"

        with gzip.open(output_path / filename, 'wt', encoding='utf-8') as f:
            for item in processed_batch:
                # Convert to JSON and write
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(processed_batch)} preprocessed items to {filename}")
        return str(output_path / filename)

    def run_preprocessing_pipeline(self, limit=None, batch_size=None, output_dir="preprocessed"):
        """Run the complete preprocessing pipeline on the corpus"""

        if batch_size:
            self.batch_size = batch_size

        total_processed = 0
        batch_count = 0

        logger.info(f"Starting preprocessing pipeline (batch_size={self.batch_size})")

        for batch in self._batch_process_texts(limit=limit):
            logger.info(f"Processing batch {batch_count + 1} with {len(batch)} items")

            # Preprocess batch
            processed_batch = self.preprocess_corpus_batch(batch)

            if processed_batch:
                # Save results
                self.save_preprocessed_batch(processed_batch, output_dir)
                total_processed += len(processed_batch)

            batch_count += 1

            # Progress logging
            if batch_count % 10 == 0:
                logger.info(f"Processed {total_processed} items in {batch_count} batches")

        logger.info(f"Preprocessing pipeline completed. Total processed: {total_processed}")
        return total_processed

    def get_preprocessing_stats(self):
        """Get statistics about preprocessing results"""
        # This would analyze the preprocessed files
        # For now, return basic info
        return {
            'pipeline_steps': self.pipeline_steps,
            'available_tools': list(self.nlp_tools.keys()),
            'gpu_accelerated': self.use_gpu
        }

def main():
    """Main preprocessing workflow"""
    preprocessor = CostEffectivePreprocessor()

    # Print available tools
    print("Available NLP Tools:")
    for tool, status in preprocessor.nlp_tools.items():
        print(f"  - {tool}: {'Available' if status else 'Not available'}")

    # Run preprocessing
    stats = preprocessor.run_preprocessing_pipeline(limit=1000)  # Process first 1000 items

    print(f"\nPreprocessing completed: {stats} items processed")

if __name__ == "__main__":
    main()
