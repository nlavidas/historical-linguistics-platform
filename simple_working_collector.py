#!/usr/bin/env python3
"""
Simple Working Collector - Alternative Strategy
Collects texts immediately without complex dependencies
"""

import sys
import os
import sqlite3
import requests
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'simple_collector.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class SimpleCollector:
    """Simple, reliable text collector"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.stats = {
            'collected': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Simple, reliable text sources
        self.sources = {
            'gutenberg': [
                ('http://www.gutenberg.org/files/1342/1342-0.txt', 'Pride and Prejudice', 'en'),
                ('http://www.gutenberg.org/files/84/84-0.txt', 'Frankenstein', 'en'),
                ('http://www.gutenberg.org/files/1661/1661-0.txt', 'Sherlock Holmes', 'en'),
                ('http://www.gutenberg.org/files/2701/2701-0.txt', 'Moby Dick', 'en'),
                ('http://www.gutenberg.org/files/11/11-0.txt', 'Alice in Wonderland', 'en'),
                ('http://www.gutenberg.org/files/1952/1952-0.txt', 'The Yellow Wallpaper', 'en'),
                ('http://www.gutenberg.org/files/98/98-0.txt', 'A Tale of Two Cities', 'en'),
                ('http://www.gutenberg.org/files/1080/1080-0.txt', 'A Modest Proposal', 'en'),
                ('http://www.gutenberg.org/files/174/174-0.txt', 'The Picture of Dorian Gray', 'en'),
                ('http://www.gutenberg.org/files/2554/2554-0.txt', 'Crime and Punishment', 'en'),
            ],
            'perseus_latin': [
                ('http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.02.0055', 'Caesar - Gallic War', 'lat'),
                ('http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.02.0054', 'Cicero - De Officiis', 'lat'),
            ],
            'perseus_greek': [
                ('http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0133', 'Homer - Iliad', 'grc'),
                ('http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0135', 'Homer - Odyssey', 'grc'),
            ]
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Ensure database exists with correct structure"""
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
                    proiel_xml TEXT,
                    word_count INTEGER,
                    date_added TEXT,
                    status TEXT,
                    metadata_quality REAL DEFAULT 0,
                    annotation_score REAL DEFAULT 0,
                    tokens_count INTEGER DEFAULT 0,
                    lemmas_count INTEGER DEFAULT 0,
                    pos_tags_count INTEGER DEFAULT 0,
                    dependencies_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✓ Database ready")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def download_text(self, url: str, title: str, language: str) -> Dict:
        """Download a text from URL"""
        try:
            logger.info(f"Downloading: {title}")
            logger.info(f"  URL: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                word_count = len(content.split())
                
                logger.info(f"✓ Downloaded: {word_count:,} words")
                
                return {
                    'url': url,
                    'title': title,
                    'language': language,
                    'content': content,
                    'word_count': word_count,
                    'date_added': datetime.now().isoformat(),
                    'status': 'collected'
                }
            else:
                logger.warning(f"✗ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            return None
    
    def save_to_database(self, text_data: Dict):
        """Save text to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate metadata quality
            quality = 0
            if text_data.get('title'): quality += 25
            if text_data.get('language'): quality += 25
            if text_data.get('word_count', 0) > 0: quality += 25
            if text_data.get('date_added'): quality += 25
            
            cursor.execute("""
                INSERT OR REPLACE INTO corpus_items
                (url, title, language, content, word_count, date_added, status, metadata_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_data['url'],
                text_data['title'],
                text_data['language'],
                text_data['content'],
                text_data['word_count'],
                text_data['date_added'],
                text_data['status'],
                quality
            ))
            
            conn.commit()
            text_id = cursor.lastrowid
            conn.close()
            
            logger.info("="*80)
            logger.info(f"✓ TEXT SAVED TO DATABASE - ID: {text_id}")
            logger.info("="*80)
            logger.info(f"Title:              {text_data['title']}")
            logger.info(f"URL:                {text_data['url']}")
            logger.info(f"Language:           {text_data['language']}")
            logger.info(f"Word Count:         {text_data['word_count']:,}")
            logger.info(f"Metadata Quality:   {quality:.0f}%")
            logger.info(f"Date Added:         {text_data['date_added']}")
            logger.info("="*80)
            
            self.stats['collected'] += 1
            
        except Exception as e:
            logger.error(f"✗ Database save failed: {e}")
            self.stats['errors'] += 1
    
    def collect_batch(self, batch_size: int = 5):
        """Collect a batch of texts"""
        logger.info("\n" + "="*80)
        logger.info("STARTING COLLECTION BATCH")
        logger.info("="*80)
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")
        
        collected = 0
        
        # Collect from each source
        for source_name, texts in self.sources.items():
            if collected >= batch_size:
                break
            
            logger.info(f"\n>>> SOURCE: {source_name}")
            logger.info("-"*80)
            
            for url, title, language in texts:
                if collected >= batch_size:
                    break
                
                # Download text
                text_data = self.download_text(url, title, language)
                
                if text_data:
                    # Save to database
                    self.save_to_database(text_data)
                    collected += 1
                    
                    # Print stats
                    self.print_stats()
                
                # Be respectful
                time.sleep(2)
        
        logger.info("\n" + "="*80)
        logger.info(f"BATCH COMPLETE: {collected} texts collected")
        logger.info("="*80 + "\n")
        
        return collected
    
    def print_stats(self):
        """Print current statistics"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = elapsed / 3600
        
        logger.info("\n>>> SESSION STATISTICS:")
        logger.info(f"  Texts Collected: {self.stats['collected']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Elapsed Time: {elapsed/60:.1f} minutes")
        if hours > 0:
            logger.info(f"  Rate: {self.stats['collected']/hours:.1f} texts/hour")
    
    def run_continuous(self, texts_per_batch: int = 5, delay_minutes: int = 10):
        """Run continuous collection"""
        logger.info("="*80)
        logger.info("CONTINUOUS COLLECTION MODE")
        logger.info("="*80)
        logger.info(f"Texts per batch: {texts_per_batch}")
        logger.info(f"Delay between batches: {delay_minutes} minutes")
        logger.info("="*80 + "\n")
        
        batch_num = 0
        
        while True:
            batch_num += 1
            
            logger.info(f"\n{'█'*80}")
            logger.info(f"BATCH {batch_num}")
            logger.info(f"{'█'*80}\n")
            
            collected = self.collect_batch(texts_per_batch)
            
            if collected > 0:
                logger.info(f"\n✓ Batch {batch_num} complete: {collected} texts")
            else:
                logger.warning(f"\n⚠ Batch {batch_num}: No texts collected")
            
            logger.info(f"\nWaiting {delay_minutes} minutes until next batch...")
            logger.info(f"Next batch at: {(datetime.now().timestamp() + delay_minutes*60)}")
            
            time.sleep(delay_minutes * 60)


def main():
    """Main entry point"""
    print("="*80)
    print("SIMPLE WORKING COLLECTOR")
    print("="*80)
    print("Alternative strategy: Direct downloads from reliable sources")
    print("="*80)
    print()
    
    collector = SimpleCollector()
    
    # Run continuous collection
    # 5 texts every 10 minutes = 30 texts/hour
    collector.run_continuous(texts_per_batch=5, delay_minutes=10)


if __name__ == "__main__":
    main()
