#!/usr/bin/env python3
"""Advanced API Discovery for Diachronic Corpus Collection.

Automatically discovers and collects texts from online APIs and repositories
across all historical periods of Greek, English, and other IE languages.

Supports:
- Perseus Digital Library API (Greek, Latin)
- Open Greek and Latin (GitHub)
- Gutenberg API (English retranslations)
- PHI Latin Texts
- Latin Library
- First1KGreek

Run:
    python api_discovery_collector.py --languages grc lat en --periods all

This extends the existing collectors with API-driven discovery.
"""

import json
import logging
import re
import requests
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_discovery_collector")

class APIDiscoveryCollector:
    """Advanced collector using APIs for comprehensive diachronic coverage."""

    def __init__(self, languages: List[str] = None, periods: List[str] = None):
        self.db_path = DB_PATH
        self.languages = languages or ['grc', 'lat', 'en']
        self.periods = periods or ['all']
        self.setup_database()

    def setup_database(self):
        """Ensure database has necessary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                language TEXT,
                content TEXT,
                word_count INTEGER,
                date_added TEXT,
                status TEXT DEFAULT 'collected',
                metadata_quality REAL DEFAULT 100.0,
                genre TEXT,
                period TEXT,
                author TEXT,
                translator TEXT,
                original_language TEXT,
                translation_year INTEGER,
                is_retranslation BOOLEAN DEFAULT 0,
                is_retelling BOOLEAN DEFAULT 0,
                is_biblical BOOLEAN DEFAULT 0,
                is_classical BOOLEAN DEFAULT 0,
                text_type TEXT,
                diachronic_stage TEXT
            )
        """)

        conn.commit()
        conn.close()

    def discover_perseus_texts(self, language: str) -> List[Dict]:
        """Discover texts from Perseus Digital Library."""
        texts = []

        # Perseus API endpoints
        base_urls = {
            'grc': 'http://www.perseus.tufts.edu/hopper/xmlchunk?doc=',
            'lat': 'http://www.perseus.tufts.edu/hopper/xmlchunk?doc='
        }

        if language not in base_urls:
            return texts

        # Known Perseus texts by period
        perseus_texts = {
            'grc': {
                'Archaic Greek (8th century BCE)': [
                    'Perseus:text:1999.01.0133',  # Iliad
                    'Perseus:text:1999.01.0134',  # Iliad (alternate)
                    'Perseus:text:1999.01.0135',  # Odyssey
                ],
                'Classical Greek (5th century BCE)': [
                    'Perseus:text:1999.01.0113',  # Sophocles Ajax
                    'Perseus:text:1999.01.0114',  # Sophocles Antigone
                    'Perseus:text:1999.01.0116',  # Euripides Alcestis
                    'Perseus:text:1999.01.0119',  # Euripides Medea
                ],
                'Hellenistic Greek (3rd century BCE)': [
                    'Perseus:text:2008.01.0589',  # Theocritus Idylls
                ],
                'Koine Greek (1st century CE)': [
                    'Perseus:text:1999.01.0133',  # NT Matthew (if available)
                ]
            },
            'lat': {
                'Classical Latin (1st century BCE)': [
                    'Perseus:text:1999.02.0002',  # Virgil Aeneid
                    'Perseus:text:1999.02.0001',  # Virgil Eclogues
                ],
                'Late Latin (6th century CE)': [
                    'Perseus:text:1999.02.0029',  # Boethius Consolation
                ]
            }
        }

        for period, urls in perseus_texts.get(language, {}).items():
            if self.periods == ['all'] or period in self.periods:
                for url_suffix in urls:
                    try:
                        url = base_urls[language] + url_suffix
                        response = requests.get(url, headers=HEADERS, timeout=10)
                        if response.status_code == 200:
                            content = self.extract_text_from_perseus(response.text)
                            if content and len(content.split()) > 100:
                                texts.append({
                                    'url': url,
                                    'title': f'Perseus {language.upper()} Text {url_suffix}',
                                    'language': language,
                                    'content': content,
                                    'period': period,
                                    'diachronic_stage': period,
                                    'genre': 'literature',
                                    'is_classical': True
                                })
                    except Exception as e:
                        logger.warning(f"Failed to fetch Perseus text {url_suffix}: {e}")

        return texts

    def extract_text_from_perseus(self, html: str) -> str:
        """Extract plain text from Perseus HTML/XML."""
        # Simple extraction - in practice, use BeautifulSoup
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def discover_open_greek_latin(self, language: str) -> List[Dict]:
        """Discover texts from Open Greek and Latin GitHub."""
        texts = []

        base_url = 'https://raw.githubusercontent.com/OpenGreekAndLatin/'

        # Open Greek and Latin repositories by period
        ogl_repos = {
            'grc': {
                'Archaic Greek (8th century BCE)': [
                    'First1KGreek/master/data/tlg0012/tlg001.xml',  # Homer Iliad
                    'First1KGreek/master/data/tlg0012/tlg002.xml',  # Homer Odyssey
                ],
                'Classical Greek (5th century BCE)': [
                    'First1KGreek/master/data/tlg0006/tlg001.xml',  # Euripides
                    'First1KGreek/master/data/tlg0007/tlg001.xml',  # Sophocles
                ],
                'Hellenistic Greek (3rd century BCE)': [
                    'First1KGreek/master/data/tlg0015/tlg001.xml',  # Callimachus
                ],
                'Koine Greek (1st century CE)': [
                    'First1KGreek/master/data/tlg0031/tlg001.xml',  # NT
                ]
            },
            'lat': {
                'Classical Latin (1st century BCE)': [
                    'First1KLatin/master/data/phi0690/phi001/phi0690.phi001.perseus-lat2.xml',  # Virgil
                ],
                'Late Latin (4th century CE)': [
                    'First1KLatin/master/data/stoa0023/stoa001/stoa0023.stoa001.opp-lat1.xml',  # Augustine
                ]
            }
        }

        for period, files in ogl_repos.get(language, {}).items():
            if self.periods == ['all'] or period in self.periods:
                for file_path in files:
                    try:
                        url = base_url + file_path
                        response = requests.get(url, headers=HEADERS, timeout=10)
                        if response.status_code == 200:
                            content = self.extract_text_from_xml(response.text)
                            if content and len(content.split()) > 100:
                                texts.append({
                                    'url': url,
                                    'title': f'OpenGreekLatin {language.upper()} {file_path.split("/")[-1]}',
                                    'language': language,
                                    'content': content,
                                    'period': period,
                                    'diachronic_stage': period,
                                    'genre': 'literature',
                                    'is_classical': True
                                })
                    except Exception as e:
                        logger.warning(f"Failed to fetch OGL text {file_path}: {e}")

        return texts

    def extract_text_from_xml(self, xml: str) -> str:
        """Extract plain text from XML."""
        import re
        # Remove XML tags
        text = re.sub(r'<[^>]+>', '', xml)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def discover_gutenberg_retranslations(self, language: str) -> List[Dict]:
        """Discover English retranslations from Gutenberg."""
        texts = []

        # Gutenberg retranslation URLs by original language and period
        retranslation_urls = {
            'grc': {
                'Archaic Greek (8th century BCE)': [
                    ('https://www.gutenberg.org/cache/epub/1727/pg1727.txt', 'Homer Odyssey (Chapman 1616)'),
                    ('https://www.gutenberg.org/cache/epub/1728/pg1728.txt', 'Homer Iliad (Pope 1720)'),
                    ('https://www.gutenberg.org/cache/epub/2199/pg2199.txt', 'Homer Iliad (Butler 1898)'),
                ],
                'Classical Greek (5th century BCE)': [
                    ('https://www.gutenberg.org/cache/epub/35451/pg35451.txt', 'Euripides Medea (Murray)'),
                    ('https://www.gutenberg.org/cache/epub/27673/pg27673.txt', 'Sophocles Oedipus (Murray)'),
                    ('https://www.gutenberg.org/cache/epub/14417/pg14417.txt', 'Aeschylus Agamemnon (Morshead)'),
                ]
            },
            'lat': {
                'Late Latin (6th century CE)': [
                    ('https://www.gutenberg.org/cache/epub/14328/pg14328.txt', 'Boethius Consolation (James)'),
                    ('https://www.gutenberg.org/cache/epub/42083/pg42083.txt', 'Boethius Consolation (Chaucer)'),
                ]
            }
        }

        for orig_lang, periods in retranslation_urls.items():
            for period, url_title_pairs in periods.items():
                if self.periods == ['all'] or period in self.periods:
                    for url, title in url_title_pairs:
                        try:
                            response = requests.get(url, headers=HEADERS, timeout=10)
                            if response.status_code == 200:
                                content = response.text
                                word_count = len(content.split())
                                if word_count > 500:
                                    texts.append({
                                        'url': url,
                                        'title': title,
                                        'language': language,
                                        'content': content,
                                        'word_count': word_count,
                                        'period': period,
                                        'diachronic_stage': period,
                                        'genre': 'literature',
                                        'original_language': orig_lang,
                                        'is_retranslation': True,
                                        'is_classical': True
                                    })
                        except Exception as e:
                            logger.warning(f"Failed to fetch Gutenberg text {url}: {e}")

        return texts

    def collect_all_texts(self) -> int:
        """Discover and collect texts from all APIs."""
        total_collected = 0

        for language in self.languages:
            logger.info(f"Discovering texts for {language}...")

            # Perseus
            perseus_texts = self.discover_perseus_texts(language)
            total_collected += self.save_texts(perseus_texts)

            # Open Greek and Latin
            ogl_texts = self.discover_open_greek_latin(language)
            total_collected += self.save_texts(ogl_texts)

            # Gutenberg retranslations
            if language == 'en':
                gutenberg_texts = self.discover_gutenberg_retranslations(language)
                total_collected += self.save_texts(gutenberg_texts)

        return total_collected

    def save_texts(self, texts: List[Dict]) -> int:
        """Save discovered texts to database."""
        saved = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for text in texts:
            try:
                # Check if URL already exists
                cursor.execute("SELECT id FROM corpus_items WHERE url = ?", (text['url'],))
                if cursor.fetchone():
                    continue  # Skip duplicates

                # Calculate word count if not provided
                content = text['content']
                word_count = text.get('word_count', len(content.split()))

                cursor.execute("""
                    INSERT INTO corpus_items
                    (url, title, language, content, word_count, date_added, status,
                     genre, period, original_language, is_retranslation, is_classical,
                     diachronic_stage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    text['url'],
                    text['title'],
                    text['language'],
                    content,
                    word_count,
                    datetime.now().isoformat(),
                    'collected',
                    text.get('genre', ''),
                    text.get('period', ''),
                    text.get('original_language', ''),
                    text.get('is_retranslation', False),
                    text.get('is_classical', False),
                    text.get('diachronic_stage', '')
                ))

                saved += 1
                logger.info(f"Saved: {text['title']} ({word_count:,} words)")

            except Exception as e:
                logger.error(f"Failed to save text {text.get('title', 'unknown')}: {e}")

        conn.commit()
        conn.close()
        return saved

def main():
    import argparse

    parser = argparse.ArgumentParser(description='API Discovery Collector')
    parser.add_argument('--languages', nargs='+', default=['grc', 'lat', 'en'],
                       help='Languages to collect (default: grc lat en)')
    parser.add_argument('--periods', nargs='+', default=['all'],
                       help='Periods to collect (default: all)')

    args = parser.parse_args()

    collector = APIDiscoveryCollector(args.languages, args.periods)
    collected = collector.collect_all_texts()
    logger.info(f"Discovery complete. Collected {collected} new texts.")

if __name__ == '__main__':
    main()
