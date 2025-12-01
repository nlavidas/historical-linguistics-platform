#!/usr/bin/env python3
"""
COST-EFFECTIVE TEXT COLLECTION SYSTEM
Community-driven tools for linguistic corpus building
Optimized for low-cost servers and storage

Features:
- Free/public data sources
- Efficient storage compression
- Incremental collection
- Quality filtering
- Metadata enrichment
- Duplicate detection
- Language identification
- Basic preprocessing
"""

import os
import sys
import json
import sqlite3
import logging
import requests
import time
import hashlib
import gzip
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from collections import defaultdict, Counter
import re
import unicodedata

# Free NLP tools
try:
    import langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostEffectiveTextCollector:
    """
    Text collection system optimized for low-cost infrastructure
    Uses free APIs, public datasets, and efficient storage
    """

    def __init__(self, db_path="corpus_efficient.db", storage_path="corpus_storage"):
        self.db_path = db_path
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Compression settings for storage efficiency
        self.compression_level = 6  # Balanced speed/size

        # Rate limiting to avoid blocking
        self.request_delay = 1.0  # 1 second between requests
        self.last_request = 0

        # Quality thresholds
        self.min_text_length = 50
        self.max_text_length = 10000
        self.min_words = 10

        # Language settings for historical linguistics
        self.target_languages = ['grc', 'la', 'en', 'de', 'fr', 'es']

        self._init_database()

    def _init_database(self):
        """Create efficient database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main corpus table with compression-ready design
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corpus_items (
                id INTEGER PRIMARY KEY,
                source_url TEXT,
                title TEXT,
                content_hash TEXT UNIQUE,
                content_compressed BLOB,
                content_length INTEGER,
                language TEXT,
                period TEXT,
                genre TEXT,
                quality_score REAL,
                collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duplicate_of INTEGER,
                word_count INTEGER,
                sentence_count INTEGER,
                metadata TEXT
            )
        ''')

        # Indexes for efficient queries (space-optimized)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_lang_period ON corpus_items(language, period)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON corpus_items(quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection_date ON corpus_items(collection_date)')

        # Sources tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                name TEXT,
                type TEXT,
                last_crawled TIMESTAMP,
                crawl_frequency_hours INTEGER DEFAULT 24,
                status TEXT DEFAULT 'active'
            )
        ''')

        conn.commit()
        conn.close()

    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()

    def _compress_text(self, text):
        """Compress text for efficient storage"""
        return gzip.compress(text.encode('utf-8'), compresslevel=self.compression_level)

    def _decompress_text(self, compressed_data):
        """Decompress stored text"""
        return gzip.decompress(compressed_data).decode('utf-8')

    def _calculate_content_hash(self, text):
        """Calculate content hash for duplicate detection"""
        # Normalize text for better duplicate detection
        normalized = unicodedata.normalize('NFKC', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _detect_language(self, text):
        """Detect language using free tools"""
        if HAS_LANGDETECT:
            try:
                return langdetect.detect(text)
            except:
                pass

        # Fallback: simple heuristics for historical languages
        text_lower = text.lower()

        # Ancient Greek indicators
        greek_indicators = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
        if any(char in text_lower for char in greek_indicators):
            # Distinguish ancient vs modern Greek
            if any(word in text_lower for word in ['εἰμί', 'εἶναι', 'λέγω', 'φημί']):
                return 'grc'  # Ancient Greek
            return 'el'   # Modern Greek

        # Latin indicators
        latin_indicators = ['ā', 'ē', 'ī', 'ō', 'ū', 'æ', 'œ']
        if any(char in text_lower for char in latin_indicators):
            return 'la'

        return 'unknown'

    def _assess_text_quality(self, text):
        """Assess text quality for corpus inclusion"""
        if not text or len(text) < self.min_text_length:
            return 0.0

        score = 1.0

        # Length check
        if len(text) > self.max_text_length:
            score *= 0.8

        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            score *= 0.5

        # Character diversity (avoid repetitive text)
        char_counts = Counter(text.lower())
        unique_chars = len(char_counts)
        total_chars = len(text)
        diversity_ratio = unique_chars / total_chars

        if diversity_ratio < 0.1:  # Very repetitive
            score *= 0.3
        elif diversity_ratio < 0.3:  # Somewhat repetitive
            score *= 0.7

        # Check for markup/HTML (penalty)
        if '<' in text and '>' in text:
            score *= 0.6

        # Check for excessive numbers (might be data, not text)
        number_ratio = len(re.findall(r'\d', text)) / len(text)
        if number_ratio > 0.1:
            score *= 0.7

        return min(score, 1.0)

    def _extract_metadata(self, text, url=""):
        """Extract basic metadata from text"""
        metadata = {}

        # Word and sentence counts
        words = text.split()
        metadata['word_count'] = len(words)
        metadata['sentence_count'] = len(re.split(r'[.!?]+', text))

        # Average word length
        if words:
            metadata['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            metadata['avg_word_length'] = 0

        # Character statistics
        metadata['char_count'] = len(text)
        metadata['unique_chars'] = len(set(text.lower()))

        # Source information
        if url:
            parsed = urlparse(url)
            metadata['domain'] = parsed.netloc
            metadata['path'] = parsed.path

        return json.dumps(metadata)

    def add_source(self, url, name, source_type="web", crawl_frequency_hours=24):
        """Add a data source for collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO sources (url, name, type, crawl_frequency_hours, last_crawled)
            VALUES (?, ?, ?, ?, ?)
        ''', (url, name, source_type, crawl_frequency_hours, datetime.now()))

        conn.commit()
        conn.close()
        logger.info(f"Added source: {name} ({url})")

    def collect_from_perseus(self, start_page=1, max_pages=10):
        """Collect texts from Perseus Digital Library (free, high-quality)"""
        base_url = "https://www.perseus.tufts.edu/hopper/text"

        for page in range(start_page, start_page + max_pages):
            try:
                self._rate_limit()
                url = f"{base_url}?page={page}"

                response = requests.get(url, timeout=30)
                response.raise_for_status()

                if HAS_BEAUTIFULSOUP:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract text content (adjust selectors based on Perseus structure)
                    text_elements = soup.find_all(['p', 'div'], class_=re.compile(r'text|content'))

                    for element in text_elements:
                        text = element.get_text().strip()

                        if self._assess_text_quality(text) > 0.6:
                            self.store_text(
                                text=text,
                                source_url=url,
                                title=f"Perseus Text Page {page}",
                                language="grc",  # Assume Greek, could be detected
                                period="ancient"
                            )

                logger.info(f"Processed Perseus page {page}")

            except Exception as e:
                logger.error(f"Error collecting from Perseus page {page}: {e}")

    def collect_from_gutenberg(self, languages=None):
        """Collect public domain texts from Project Gutenberg"""
        if languages is None:
            languages = ['grc', 'la', 'en']

        base_url = "https://www.gutenberg.org/ebooks/search/"

        for lang in languages:
            try:
                self._rate_limit()

                # Search for texts in specific language
                search_url = f"{base_url}?query=language:{lang}&submit_search=Go"
                response = requests.get(search_url, timeout=30)

                if HAS_BEAUTIFULSOUP:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract book links (simplified - adjust based on Gutenberg structure)
                    book_links = soup.find_all('a', href=re.compile(r'/ebooks/\d+'))

                    for link in book_links[:5]:  # Limit to avoid overwhelming
                        book_url = urljoin("https://www.gutenberg.org", link['href'])

                        # Get actual text content
                        text_url = book_url.replace('/ebooks/', '/files/') + '/plain_text.txt'
                        try:
                            self._rate_limit()
                            text_response = requests.get(text_url, timeout=30)

                            if text_response.status_code == 200:
                                text = text_response.text

                                if self._assess_text_quality(text) > 0.5:
                                    title = link.get_text().strip() or f"Gutenberg {lang.upper()} Text"
                                    self.store_text(
                                        text=text,
                                        source_url=book_url,
                                        title=title,
                                        language=lang,
                                        period="various",
                                        genre="literature"
                                    )
                        except:
                            continue

                logger.info(f"Processed Gutenberg texts for language: {lang}")

            except Exception as e:
                logger.error(f"Error collecting from Gutenberg for {lang}: {e}")

    def collect_from_wikimedia(self, language="grc", categories=None):
        """Collect texts from Wikimedia dumps (free, comprehensive)"""
        if categories is None:
            categories = ["Ancient_Greek_texts", "Latin_texts"]

        # Wikimedia API for category members
        api_url = f"https://{language}.wikisource.org/w/api.php"

        for category in categories:
            try:
                self._rate_limit()

                params = {
                    'action': 'query',
                    'list': 'categorymembers',
                    'cmtitle': f'Category:{category}',
                    'cmlimit': '50',
                    'format': 'json'
                }

                response = requests.get(api_url, params=params, timeout=30)
                data = response.json()

                for member in data.get('query', {}).get('categorymembers', []):
                    page_title = member['title']

                    # Get page content
                    content_params = {
                        'action': 'query',
                        'prop': 'extracts',
                        'titles': page_title,
                        'explaintext': True,
                        'format': 'json'
                    }

                    try:
                        self._rate_limit()
                        content_response = requests.get(api_url, params=content_params, timeout=30)
                        content_data = content_response.json()

                        pages = content_data.get('query', {}).get('pages', {})
                        for page_id, page_data in pages.items():
                            text = page_data.get('extract', '')

                            if self._assess_text_quality(text) > 0.4:
                                self.store_text(
                                    text=text,
                                    source_url=f"https://{language}.wikisource.org/wiki/{page_title}",
                                    title=page_title,
                                    language=language,
                                    period="various",
                                    genre="reference"
                                )

                    except Exception as e:
                        logger.error(f"Error getting content for {page_title}: {e}")
                        continue

                logger.info(f"Processed Wikimedia category: {category}")

            except Exception as e:
                logger.error(f"Error collecting from Wikimedia category {category}: {e}")

    def collect_from_arxiv(self, query="linguistics", max_results=20):
        """Collect academic papers from arXiv (free, high-quality)"""
        api_url = "http://export.arxiv.org/api/query"

        try:
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            self._rate_limit()
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            if HAS_BEAUTIFULSOUP:
                soup = BeautifulSoup(response.text, 'xml')

                for entry in soup.find_all('entry'):
                    title = entry.find('title').get_text().strip()
                    summary = entry.find('summary').get_text().strip()
                    published = entry.find('published').get_text().strip()

                    # Combine title and abstract
                    full_text = f"{title}\n\n{summary}"

                    if self._assess_text_quality(full_text) > 0.7:
                        arxiv_id = entry.find('id').get_text().split('/')[-1]

                        self.store_text(
                            text=full_text,
                            source_url=f"https://arxiv.org/abs/{arxiv_id}",
                            title=title,
                            language="en",
                            period="modern",
                            genre="academic"
                        )

            logger.info(f"Processed {max_results} arXiv papers for query: {query}")

        except Exception as e:
            logger.error(f"Error collecting from arXiv: {e}")

    def store_text(self, text, source_url="", title="", language=None,
                   period=None, genre=None, metadata=None):
        """Store text efficiently with compression and deduplication"""

        # Quality check
        quality_score = self._assess_text_quality(text)
        if quality_score < 0.3:
            return False

        # Detect language if not provided
        if not language:
            language = self._detect_language(text)

        # Calculate content hash for deduplication
        content_hash = self._calculate_content_hash(text)

        # Check for duplicates
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM corpus_items WHERE content_hash = ?", (content_hash,))
        existing = cursor.fetchone()

        if existing:
            logger.info(f"Duplicate text detected (hash: {content_hash[:8]}...)")
            conn.close()
            return False

        # Compress text
        compressed_content = self._compress_text(text)

        # Prepare metadata
        if metadata is None:
            metadata = self._extract_metadata(text, source_url)

        # Word and sentence counts
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        # Store in database
        try:
            cursor.execute('''
                INSERT INTO corpus_items
                (source_url, title, content_hash, content_compressed, content_length,
                 language, period, genre, quality_score, word_count, sentence_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                source_url,
                title,
                content_hash,
                compressed_content,
                len(text),
                language,
                period,
                genre,
                quality_score,
                len(words),
                len(sentences),
                metadata
            ))

            conn.commit()
            item_id = cursor.lastrowid

            logger.info(f"Stored text: {title[:50]}... ({len(text)} chars, quality: {quality_score:.2f})")
            return item_id

        except Exception as e:
            logger.error(f"Error storing text: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_text(self, item_id):
        """Retrieve and decompress text"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT content_compressed FROM corpus_items WHERE id = ?", (item_id,))
        result = cursor.fetchone()

        conn.close()

        if result:
            return self._decompress_text(result[0])
        return None

    def get_corpus_stats(self):
        """Get comprehensive corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total items and storage
        cursor.execute("SELECT COUNT(*), SUM(LENGTH(content_compressed)) FROM corpus_items")
        total_items, compressed_size = cursor.fetchone()
        stats['total_items'] = total_items or 0
        stats['compressed_size_mb'] = (compressed_size or 0) / (1024 * 1024)

        # Language breakdown
        cursor.execute("SELECT language, COUNT(*) FROM corpus_items GROUP BY language")
        stats['language_breakdown'] = dict(cursor.fetchall())

        # Quality distribution
        cursor.execute("SELECT quality_score, COUNT(*) FROM corpus_items GROUP BY ROUND(quality_score, 1)")
        quality_dist = dict(cursor.fetchall())
        stats['quality_distribution'] = quality_dist

        # Period breakdown
        cursor.execute("SELECT period, COUNT(*) FROM corpus_items WHERE period IS NOT NULL GROUP BY period")
        stats['period_breakdown'] = dict(cursor.fetchall())

        # Recent additions (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM corpus_items WHERE collection_date > ?", (week_ago,))
        stats['recent_additions'] = cursor.fetchone()[0]

        conn.close()
        return stats

    def deduplicate_corpus(self):
        """Find and mark duplicates (conservative approach)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Group by content hash and mark duplicates
        cursor.execute('''
            SELECT content_hash, GROUP_CONCAT(id) as ids
            FROM corpus_items
            GROUP BY content_hash
            HAVING COUNT(*) > 1
        ''')

        duplicates_found = 0
        for row in cursor.fetchall():
            content_hash, ids_str = row
            ids = ids_str.split(',')

            # Keep the first one, mark others as duplicates
            for duplicate_id in ids[1:]:
                cursor.execute(
                    "UPDATE corpus_items SET duplicate_of = ? WHERE id = ?",
                    (ids[0], duplicate_id)
                )
                duplicates_found += 1

        conn.commit()
        conn.close()

        logger.info(f"Marked {duplicates_found} duplicate texts")
        return duplicates_found

    def optimize_storage(self):
        """Optimize database and storage"""
        logger.info("Optimizing storage...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Vacuum database to reclaim space
        cursor.execute("VACUUM")

        # Analyze for query optimization
        cursor.execute("ANALYZE")

        conn.commit()
        conn.close()

        logger.info("Storage optimization completed")

    def run_collection_cycle(self, sources=None):
        """Run a complete collection cycle"""
        if sources is None:
            sources = ['perseus', 'gutenberg', 'wikimedia', 'arxiv']

        logger.info(f"Starting collection cycle with sources: {sources}")

        for source in sources:
            try:
                if source == 'perseus':
                    self.collect_from_perseus(max_pages=5)
                elif source == 'gutenberg':
                    self.collect_from_gutenberg()
                elif source == 'wikimedia':
                    self.collect_from_wikimedia()
                elif source == 'arxiv':
                    self.collect_from_arxiv(max_results=10)

                # Small delay between sources
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error collecting from {source}: {e}")

        # Post-collection maintenance
        self.deduplicate_corpus()
        self.optimize_storage()

        stats = self.get_corpus_stats()
        logger.info(f"Collection cycle completed. Corpus now has {stats['total_items']} items")

        return stats

# Usage examples and setup
def setup_collection_sources(collector):
    """Set up common collection sources"""

    # Free academic and historical text sources
    sources = [
        ("https://www.perseus.tufts.edu/hopper/", "Perseus Digital Library", "academic"),
        ("https://www.gutenberg.org/", "Project Gutenberg", "public_domain"),
        ("https://en.wikisource.org/", "Wikisource English", "wiki"),
        ("https://grc.wikisource.org/", "Wikisource Greek", "wiki"),
        ("https://la.wikisource.org/", "Wikisource Latin", "wiki"),
        ("https://arxiv.org/", "arXiv", "academic"),
        ("https://www.texts.com/", "Internet Sacred Text Archive", "religious"),
    ]

    for url, name, source_type in sources:
        collector.add_source(url, name, source_type)

def main():
    """Main collection workflow"""
    collector = CostEffectiveTextCollector()

    # Setup sources
    setup_collection_sources(collector)

    # Run collection
    stats = collector.run_collection_cycle()

    # Print results
    print("Collection Results:")
    print(f"Total texts: {stats['total_items']}")
    print(f"Storage used: {stats['compressed_size_mb']:.2f} MB")
    print(f"Language breakdown: {stats['language_breakdown']}")
    print(f"Quality distribution: {stats.get('quality_distribution', {})}")

if __name__ == "__main__":
    main()
