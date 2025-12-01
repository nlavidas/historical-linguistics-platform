#!/usr/bin/env python3
"""
DIACHRONIC MULTILINGUAL CORPUS COLLECTOR
Collects retranslations, retellings, biblical, classical texts
Multiple languages, multiple periods, automatic treebanks
"""

import sys
import os
import sqlite3
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
import traceback
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Setup environment
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))

# Logging
log_file = Path(__file__).parent / 'diachronic_collection.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DiachronicMultilingualCollector:
    """Collect texts across languages, periods, genres"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.cycle = 0
        self.texts_collected = 0
        
        logger.info("="*80)
        logger.info("DIACHRONIC MULTILINGUAL COLLECTOR STARTING")
        logger.info("="*80)
        logger.info("Features:")
        logger.info("  - Multiple languages (Greek, Latin, English, German, French, etc.)")
        logger.info("  - Biblical texts (multiple translations)")
        logger.info("  - Classical texts (multiple editions)")
        logger.info("  - Retellings and adaptations")
        logger.info("  - Diachronic stages (Ancient, Medieval, Renaissance, Modern)")
        logger.info("  - Automatic treebank generation")
        logger.info("="*80)
        
        self.setup_database()
        self.define_sources()
    
    def setup_database(self):
        """Setup enhanced database for diachronic corpus"""
        try:
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
                    
                    -- Enhanced fields for diachronic corpus
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
                    diachronic_stage TEXT,
                    
                    -- Treebank fields
                    has_treebank BOOLEAN DEFAULT 0,
                    treebank_format TEXT,
                    annotation_date TEXT,
                    annotation_quality REAL DEFAULT 0,
                    treebank_quality TEXT,
                    contact_language TEXT,
                    valency_patterns_count INTEGER DEFAULT 0
                )
            """)
            
            # Add columns if they don't exist
            existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(corpus_items)")]
            
            new_columns = [
                ('genre', 'TEXT'),
                ('period', 'TEXT'),
                ('author', 'TEXT'),
                ('translator', 'TEXT'),
                ('original_language', 'TEXT'),
                ('translation_year', 'INTEGER'),
                ('is_retranslation', 'BOOLEAN DEFAULT 0'),
                ('is_retelling', 'BOOLEAN DEFAULT 0'),
                ('is_biblical', 'BOOLEAN DEFAULT 0'),
                ('is_classical', 'BOOLEAN DEFAULT 0'),
                ('text_type', 'TEXT'),
                ('diachronic_stage', 'TEXT'),
                ('has_treebank', 'BOOLEAN DEFAULT 0'),
                ('treebank_format', 'TEXT'),
                ('annotation_date', 'TEXT'),
                ('annotation_quality', 'REAL DEFAULT 0'),
                ('treebank_quality', 'TEXT'),
                ('contact_language', 'TEXT'),
                ('valency_patterns_count', 'INTEGER DEFAULT 0'),
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE corpus_items ADD COLUMN {col_name} {col_type}")
            
            conn.commit()
            conn.close()
            
            logger.info("Database ready with diachronic fields")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def define_sources(self):
        """Define comprehensive multilingual diachronic sources"""
        
        self.sources = {
            # ANCIENT GREEK TEXTS
            'ancient_greek': [
                {
                    'url': 'http://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133',
                    'title': 'Homer - Iliad (Greek)',
                    'language': 'grc',
                    'genre': 'epic',
                    'period': 'Ancient',
                    'author': 'Homer',
                    'diachronic_stage': 'Archaic Greek (8th century BCE)',
                    'is_classical': True
                },
                {
                    'url': 'http://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0135',
                    'title': 'Homer - Odyssey (Greek)',
                    'language': 'grc',
                    'genre': 'epic',
                    'period': 'Ancient',
                    'author': 'Homer',
                    'diachronic_stage': 'Archaic Greek (8th century BCE)',
                    'is_classical': True
                },
            ],
            
            # BIBLICAL TEXTS - MULTIPLE TRANSLATIONS
            'biblical_greek': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/8300/pg8300.txt',
                    'title': 'New Testament - Greek (Westcott-Hort)',
                    'language': 'grc',
                    'genre': 'biblical',
                    'period': 'Ancient',
                    'diachronic_stage': 'Koine Greek (1st century CE)',
                    'is_biblical': True,
                    'original_language': 'grc'
                },
            ],
            
            'biblical_english': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/10/pg10.txt',
                    'title': 'Bible - King James Version (1611)',
                    'language': 'en',
                    'genre': 'biblical',
                    'period': 'Early Modern',
                    'translator': 'King James translators',
                    'translation_year': 1611,
                    'is_biblical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Early Modern English (17th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/8001/pg8001.txt',
                    'title': 'Bible - World English Bible (Modern)',
                    'language': 'en',
                    'genre': 'biblical',
                    'period': 'Contemporary',
                    'translation_year': 2000,
                    'is_biblical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Contemporary English (21st century)'
                },
            ],
            
            # CLASSICAL LATIN
            'latin_classical': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/232/pg232.txt',
                    'title': 'Virgil - Aeneid (Latin)',
                    'language': 'lat',
                    'genre': 'epic',
                    'period': 'Ancient',
                    'author': 'Virgil',
                    'diachronic_stage': 'Classical Latin (1st century BCE)',
                    'is_classical': True
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/13316/pg13316.txt',
                    'title': 'Boethius - Consolatio Philosophiae (Latin)',
                    'language': 'lat',
                    'genre': 'philosophy',
                    'period': 'Late Antique',
                    'author': 'Boethius',
                    'diachronic_stage': 'Late Latin (6th century CE)',
                    'is_classical': True
                },
            ],
            
            # CLASSICAL TEXTS - ENGLISH TRANSLATIONS (Multiple periods)
            'classical_english_translations': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/1727/pg1727.txt',
                    'title': 'Homer - Odyssey (Chapman 1616)',
                    'language': 'en',
                    'genre': 'epic',
                    'period': 'Early Modern',
                    'author': 'Homer',
                    'translator': 'George Chapman',
                    'translation_year': 1616,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Early Modern English (17th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/1728/pg1728.txt',
                    'title': 'Homer - Iliad (Pope 1720)',
                    'language': 'en',
                    'genre': 'epic',
                    'period': 'Early Modern',
                    'author': 'Homer',
                    'translator': 'Alexander Pope',
                    'translation_year': 1720,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Early Modern English (18th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/2199/pg2199.txt',
                    'title': 'Homer - Iliad (Butler 1898)',
                    'language': 'en',
                    'genre': 'epic',
                    'period': 'Modern',
                    'author': 'Homer',
                    'translator': 'Samuel Butler',
                    'translation_year': 1898,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern English (19th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/14417/pg14417.txt',
                    'title': 'Aeschylus - Agamemnon (English translation)',
                    'language': 'en',
                    'genre': 'tragedy',
                    'period': 'Modern',
                    'author': 'Aeschylus',
                    'translator': 'E. D. A. Morshead',
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern English (19th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/27673/pg27673.txt',
                    'title': 'Sophocles - Oedipus King of Thebes (English translation)',
                    'language': 'en',
                    'genre': 'tragedy',
                    'period': 'Modern',
                    'author': 'Sophocles',
                    'translator': 'Gilbert Murray',
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern English (20th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/35451/pg35451.txt',
                    'title': 'Euripides - Medea (English translation)',
                    'language': 'en',
                    'genre': 'tragedy',
                    'period': 'Modern',
                    'author': 'Euripides',
                    'translator': 'Gilbert Murray',
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern English (20th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/14328/pg14328.txt',
                    'title': 'Boethius - The Consolation of Philosophy (English)',
                    'language': 'en',
                    'genre': 'philosophy',
                    'period': 'Modern',
                    'author': 'Boethius',
                    'translator': 'H. R. James',
                    'translation_year': 1897,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'lat',
                    'diachronic_stage': 'Modern English (19th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/42083/pg42083.txt',
                    'title': "Chaucer - Translation of Boethius's 'De Consolatione Philosophiae'",
                    'language': 'enm',
                    'genre': 'philosophy',
                    'period': 'Medieval',
                    'author': 'Boethius',
                    'translator': 'Geoffrey Chaucer',
                    'translation_year': 1380,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'lat',
                    'diachronic_stage': 'Middle English (14th century)'
                },
            ],
            
            # RETELLINGS
            'retellings': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/1656/pg1656.txt',
                    'title': 'Tales from Shakespeare - Lamb (Retelling)',
                    'language': 'en',
                    'genre': 'retelling',
                    'period': 'Modern',
                    'author': 'Charles and Mary Lamb',
                    'is_retelling': True,
                    'original_language': 'en',
                    'diachronic_stage': 'Modern English (19th century)'
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/22382/pg22382.txt',
                    'title': 'Greek Myths - Bulfinch (Retelling)',
                    'language': 'en',
                    'genre': 'mythology',
                    'period': 'Modern',
                    'author': 'Thomas Bulfinch',
                    'is_retelling': True,
                    'is_classical': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern English (19th century)'
                },
            ],
            
            # GERMAN TRANSLATIONS
            'german_translations': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/2030/pg2030.txt',
                    'title': 'Homer - Ilias (Voss German)',
                    'language': 'de',
                    'genre': 'epic',
                    'period': 'Modern',
                    'author': 'Homer',
                    'translator': 'Johann Heinrich Voss',
                    'translation_year': 1793,
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern German (18th century)'
                },
            ],
            
            # FRENCH TRANSLATIONS
            'french_translations': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/16294/pg16294.txt',
                    'title': 'Iliade (French)',
                    'language': 'fr',
                    'genre': 'epic',
                    'period': 'Modern',
                    'author': 'Homer',
                    'is_classical': True,
                    'is_retranslation': True,
                    'original_language': 'grc',
                    'diachronic_stage': 'Modern French'
                },
            ],
            
            # MEDIEVAL TEXTS
            'medieval': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/2160/pg2160.txt',
                    'title': 'Beowulf (Old English)',
                    'language': 'ang',
                    'genre': 'epic',
                    'period': 'Medieval',
                    'diachronic_stage': 'Old English (8th-11th century)',
                    'is_classical': True
                },
                {
                    'url': 'https://www.gutenberg.org/cache/epub/14/pg14.txt',
                    'title': 'Canterbury Tales - Chaucer',
                    'language': 'enm',
                    'genre': 'poetry',
                    'period': 'Medieval',
                    'author': 'Geoffrey Chaucer',
                    'diachronic_stage': 'Middle English (14th century)'
                },
            ],
            
            # RENAISSANCE
            'renaissance': [
                {
                    'url': 'https://www.gutenberg.org/cache/epub/1524/pg1524.txt',
                    'title': 'Hamlet - Shakespeare',
                    'language': 'en',
                    'genre': 'drama',
                    'period': 'Renaissance',
                    'author': 'William Shakespeare',
                    'diachronic_stage': 'Early Modern English (16th century)'
                },
            ],
        }
        
        logger.info(f"Defined {sum(len(v) for v in self.sources.values())} sources across {len(self.sources)} categories")
    
    def collect_text(self, source_info):
        """Collect one text with full metadata"""
        try:
            url = source_info['url']
            title = source_info['title']
            
            logger.info(f"Collecting: {title}")
            logger.info(f"  Language: {source_info['language']}")
            logger.info(f"  Period: {source_info.get('diachronic_stage', 'Unknown')}")
            
            content = self.download_text(url)
            if not content:
                logger.error(f"No textual content downloaded for {title}. Skipping.")
                return False
            
            # Clean content
            content = self.clean_content(content)
            word_count = len(content.split())
            if word_count < 200:
                logger.warning(f"Content for {title} looks too small ({word_count} words); skipping insert to avoid zero-length texts.")
                return False
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO corpus_items
                (url, title, language, content, word_count, date_added, status,
                 genre, period, author, translator, original_language, translation_year,
                 is_retranslation, is_retelling, is_biblical, is_classical,
                 text_type, diachronic_stage, metadata_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                url,
                title,
                source_info['language'],
                content,
                word_count,
                datetime.now().isoformat(),
                'collected',
                source_info.get('genre', ''),
                source_info.get('period', ''),
                source_info.get('author', ''),
                source_info.get('translator', ''),
                source_info.get('original_language', ''),
                source_info.get('translation_year', None),
                source_info.get('is_retranslation', False),
                source_info.get('is_retelling', False),
                source_info.get('is_biblical', False),
                source_info.get('is_classical', False),
                source_info.get('text_type', ''),
                source_info.get('diachronic_stage', ''),
                100.0
            ))
            
            conn.commit()
            conn.close()
            
            self.texts_collected += 1
            
            logger.info(f"Saved: {title} ({word_count:,} words)")
            return True
        
        except Exception as e:
            logger.error(f"Collection error for {source_info['title']}: {e}")
            logger.error(traceback.format_exc())
            return False

    def download_text(self, url):
        """Download text with Gutenberg fallbacks and sanity checks."""
        try:
            response = requests.get(url, timeout=30, headers=HEADERS)
            if response.status_code == 200:
                text = response.text
                if self.looks_like_text(text):
                    return text
            alt_url = self.build_gutenberg_fallback(url)
            if alt_url:
                logger.info(f"  Fallback download: {alt_url}")
                alt_resp = requests.get(alt_url, timeout=30, headers=HEADERS)
                if alt_resp.status_code == 200 and self.looks_like_text(alt_resp.text):
                    return alt_resp.text
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
        return ""

    @staticmethod
    def looks_like_text(sample):
        sample_lower = sample[:2000].lower()
        if '<html' in sample_lower and 'project gutenberg' not in sample_lower:
            return False
        if '*** start of' in sample_lower or len(sample.strip()) > 1000:
            return True
        return False

    @staticmethod
    def build_gutenberg_fallback(url):
        match = re.search(r'/(?:cache/epub|files)/(\d+)/', url)
        if not match:
            return None
        book_id = match.group(1)
        candidates = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-8.txt"
        ]
        for candidate in candidates:
            if candidate != url:
                return candidate
        return None
    
    def clean_content(self, text):
        """Clean Gutenberg text"""
        # Remove Gutenberg header/footer
        start_markers = [
            '*** START OF THIS PROJECT GUTENBERG',
            '*** START OF THE PROJECT GUTENBERG',
            '*END*THE SMALL PRINT',
        ]
        
        end_markers = [
            '*** END OF THIS PROJECT GUTENBERG',
            '*** END OF THE PROJECT GUTENBERG',
            'End of Project Gutenberg',
            'End of the Project Gutenberg',
        ]
        
        # Find start
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        
        # Find end
        for marker in end_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        
        return text.strip()
    
    def run_cycle(self):
        """Run one collection cycle"""
        self.cycle += 1
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"CYCLE {self.cycle} - DIACHRONIC MULTILINGUAL COLLECTION")
        logger.info("="*80)
        
        # Collect from each category
        categories = list(self.sources.keys())
        
        for category in categories[:3]:  # Collect from 3 categories per cycle
            if self.sources[category]:
                source = self.sources[category].pop(0)
                self.collect_text(source)
                time.sleep(2)
        
        # Report status
        self.report_status()
        
        logger.info(f"Cycle {self.cycle} complete")
    
    def report_status(self):
        """Report current corpus status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(word_count) as words,
                    COUNT(DISTINCT language) as languages,
                    COUNT(DISTINCT diachronic_stage) as stages
                FROM corpus_items
            """)
            
            total, words, languages, stages = cursor.fetchone()
            
            logger.info("")
            logger.info("CORPUS STATUS:")
            logger.info(f"  Total texts: {total or 0}")
            logger.info(f"  Total words: {words or 0:,}")
            logger.info(f"  Languages: {languages or 0}")
            logger.info(f"  Diachronic stages: {stages or 0}")
            
            # By language
            cursor.execute("""
                SELECT language, COUNT(*), SUM(word_count)
                FROM corpus_items
                GROUP BY language
                ORDER BY COUNT(*) DESC
            """)
            
            logger.info("")
            logger.info("BY LANGUAGE:")
            for lang, count, wc in cursor.fetchall():
                logger.info(f"  {lang}: {count} texts, {wc:,} words")
            
            # By period
            cursor.execute("""
                SELECT diachronic_stage, COUNT(*)
                FROM corpus_items
                WHERE diachronic_stage IS NOT NULL
                GROUP BY diachronic_stage
                ORDER BY COUNT(*) DESC
            """)
            
            logger.info("")
            logger.info("BY DIACHRONIC STAGE:")
            for stage, count in cursor.fetchall():
                if stage:
                    logger.info(f"  {stage}: {count} texts")
            
            # Special categories
            cursor.execute("""
                SELECT 
                    SUM(is_biblical) as biblical,
                    SUM(is_classical) as classical,
                    SUM(is_retranslation) as retranslations,
                    SUM(is_retelling) as retellings
                FROM corpus_items
            """)
            
            biblical, classical, retrans, retell = cursor.fetchone()
            
            logger.info("")
            logger.info("SPECIAL CATEGORIES:")
            logger.info(f"  Biblical texts: {biblical or 0}")
            logger.info(f"  Classical texts: {classical or 0}")
            logger.info(f"  Retranslations: {retrans or 0}")
            logger.info(f"  Retellings: {retell or 0}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Status report error: {e}")
    
    def run_until_morning(self):
        """Run until 8:00 AM"""
        import pytz
        
        end_time = datetime.now(pytz.timezone('Europe/Athens')).replace(
            hour=8, minute=0, second=0
        )
        
        if end_time <= datetime.now(pytz.timezone('Europe/Athens')):
            end_time += timedelta(days=1)
        
        logger.info(f"Will run until: {end_time.strftime('%Y-%m-%d %H:%M:%S EET')}")
        logger.info("")
        
        while datetime.now(pytz.timezone('Europe/Athens')) < end_time:
            try:
                self.run_cycle()
                
                # Wait 20 minutes
                logger.info("Waiting 20 minutes...")
                
                for i in range(20):
                    if datetime.now(pytz.timezone('Europe/Athens')) >= end_time:
                        break
                    time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
                
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)
        
        # Final report
        logger.info("")
        logger.info("="*80)
        logger.info("MORNING REACHED - FINAL DIACHRONIC CORPUS REPORT")
        logger.info("="*80)
        
        self.report_status()
        
        logger.info("")
        logger.info(f"Total Cycles: {self.cycle}")
        logger.info(f"Texts Collected: {self.texts_collected}")
        logger.info("="*80)


def main():
    """Main entry"""
    try:
        collector = DiachronicMultilingualCollector()
        collector.run_until_morning()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
