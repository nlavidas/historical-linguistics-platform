#!/usr/bin/env python3
"""
STABLE CRASH-PROOF SYSTEM
Simple, reliable, no crashes
"""

import sys
import os
import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Setup
sys.path.insert(0, str(Path(__file__).parent))

# Simple logging
log_file = Path(__file__).parent / 'stable_system.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StableSystem:
    """Crash-proof stable system"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.cycle = 0
        self.texts_collected = 0
        
        logger.info("="*80)
        logger.info("STABLE CRASH-PROOF SYSTEM STARTING")
        logger.info("="*80)
        
        # Ensure database exists
        self.setup_database()
    
    def setup_database(self):
        """Setup database safely"""
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
                    metadata_quality REAL DEFAULT 100.0
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✓ Database ready")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def collect_one_text(self):
        """Collect one text safely"""
        try:
            import requests
            
            # Simple, reliable sources
            texts = [
                ("http://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice", "en"),
                ("http://www.gutenberg.org/files/84/84-0.txt", "Frankenstein", "en"),
                ("http://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland", "en"),
                ("http://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes", "en"),
                ("http://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick", "en"),
            ]
            
            # Pick one
            url, title, lang = texts[self.texts_collected % len(texts)]
            
            logger.info(f"Collecting: {title}")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                word_count = len(content.split())
                
                # Save to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO corpus_items
                    (url, title, language, content, word_count, date_added, status, metadata_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (url, title, lang, content, word_count, 
                      datetime.now().isoformat(), 'collected', 100.0))
                
                conn.commit()
                conn.close()
                
                self.texts_collected += 1
                
                logger.info(f"✓ Saved: {title} ({word_count:,} words)")
                return True
            
        except Exception as e:
            logger.error(f"Collection error: {e}")
            return False
    
    def check_status(self):
        """Check system status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*), SUM(word_count) FROM corpus_items")
            count, words = cursor.fetchone()
            
            conn.close()
            
            logger.info(f"Status: {count or 0} texts, {words or 0:,} words")
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
    
    def run_cycle(self):
        """Run one cycle safely"""
        self.cycle += 1
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"CYCLE {self.cycle}")
        logger.info("="*80)
        
        # Collect 2 texts per cycle
        for i in range(2):
            success = self.collect_one_text()
            if success:
                time.sleep(2)  # Brief pause
        
        # Check status
        self.check_status()
        
        logger.info(f"Cycle {self.cycle} complete")
    
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
                logger.info("Continuing despite error...")
                time.sleep(60)
        
        # Final report
        logger.info("")
        logger.info("="*80)
        logger.info("MORNING REACHED - FINAL REPORT")
        logger.info("="*80)
        
        self.check_status()
        
        logger.info(f"Total Cycles: {self.cycle}")
        logger.info(f"Texts Collected: {self.texts_collected}")
        logger.info("="*80)


def main():
    """Main entry"""
    try:
        system = StableSystem()
        system.run_until_morning()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
