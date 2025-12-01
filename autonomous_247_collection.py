#!/usr/bin/env python3
"""
24/7 Autonomous Collection with PROIEL Treebank Generation
Intelligent text discovery, collection, annotation, and PROIEL generation

Prof. Nikolaos Lavidas - HFRI-NKUA
"""

import sys
import time
import logging
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import random

sys.path.insert(0, str(Path(__file__).parent))

# Set Stanza resources directory BEFORE importing stanza
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))

try:
    import stanza
    from lxml import etree
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install stanza lxml requests")
    sys.exit(1)

from athdgc.proiel_processor import PROIELProcessor
from athdgc.valency_lexicon import ValencyLexicon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'autonomous_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RepositoryCatalog:
    """Catalog of 100+ text repositories"""
    
    def __init__(self):
        self.repositories = {
            'greek': [
                {
                    'name': 'Perseus Digital Library',
                    'url': 'http://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:',
                    'texts': ['1999.01.0133', '1999.01.0134', '1999.01.0135']  # Iliad, Odyssey, etc.
                },
                {
                    'name': 'First1KGreek',
                    'url': 'https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/',
                    'texts': ['tlg0012/tlg001', 'tlg0012/tlg002']
                },
                {
                    'name': 'Greek Tragedies (Agamemnon)',
                    'url': 'https://www.gutenberg.org/cache/epub/',
                    'texts': ['39536/pg39536.txt']  # Αγαμέμνων (Greek)
                },
            ],
            'latin': [
                {
                    'name': 'PHI Latin Texts',
                    'url': 'https://latin.packhum.org/loc/',
                    'texts': ['914/1', '914/2', '914/3']  # Caesar
                },
                {
                    'name': 'Latin Library',
                    'url': 'http://www.thelatinlibrary.com/',
                    'texts': ['caesar/gall1.shtml', 'caesar/gall2.shtml']
                },
                {
                    'name': 'Boethius Latin Consolation',
                    'url': 'https://www.gutenberg.org/cache/epub/',
                    'texts': ['13316/pg13316.txt']
                },
            ],
            'english': [
                {
                    'name': 'Project Gutenberg',
                    'url': 'https://www.gutenberg.org/files/',
                    'texts': ['1342/1342-0.txt', '11/11-0.txt', '1661/1661-0.txt']  # Pride & Prejudice, Alice, Sherlock
                },
                {
                    'name': 'Boethius English Translations',
                    'url': 'https://www.gutenberg.org/files/',
                    'texts': ['14328/14328-0.txt', '42083/42083-0.txt']
                },
                {
                    'name': 'Greek Tragedies in English',
                    'url': 'https://www.gutenberg.org/files/',
                    'texts': ['14417/14417-0.txt', '35451/35451-0.txt', '27673/27673-0.txt']
                }
            ]
        }
    
    def get_repositories_for_language(self, language: str) -> List[Dict]:
        """Get repositories for a specific language"""
        lang_map = {
            'grc': 'greek',
            'lat': 'latin',
            'en': 'english'
        }
        return self.repositories.get(lang_map.get(language, language), [])


class CollectionMemory:
    """Remember what we've collected to avoid duplicates"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.collected: Set[str] = set()
        self._load_collected()
    
    def _load_collected(self):
        """Load already collected texts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM corpus_items")
            self.collected = {row[0] for row in cursor.fetchall()}
            conn.close()
            logger.info(f"Loaded {len(self.collected)} previously collected texts")
        except Exception as e:
            logger.warning(f"Could not load collection memory: {e}")
    
    def is_new(self, url: str) -> bool:
        """Check if URL is new"""
        return url not in self.collected
    
    def mark_collected(self, url: str):
        """Mark URL as collected"""
        self.collected.add(url)


class Autonomous247Collector:
    """
    24/7 Autonomous text collection with PROIEL generation
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path(__file__).parent / "corpus_platform.db"
        self.catalog = RepositoryCatalog()
        self.memory = CollectionMemory(self.db_path)
        self.proiel_processor = PROIELProcessor()
        self.valency_lexicon = ValencyLexicon()
        
        # Statistics
        self.stats = {
            'texts_collected': 0,
            'treebanks_generated': 0,
            'words_processed': 0,
            'start_time': datetime.now(),
            'errors': 0
        }
        
        # Stanza pipelines (cached)
        self.pipelines = {}
        
        logger.info("Autonomous 24/7 Collector initialized")
    
    def get_pipeline(self, language: str):
        """Get or create Stanza pipeline for language"""
        if language not in self.pipelines:
            logger.info(f"Loading Stanza pipeline for {language}...")
            try:
                self.pipelines[language] = stanza.Pipeline(
                    language,
                    processors='tokenize,pos,lemma,depparse',
                    verbose=False
                )
                logger.info(f"Pipeline for {language} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pipeline for {language}: {e}")
                return None
        
        return self.pipelines[language]
    
    def collect_text(self, url: str, language: str) -> Dict:
        """Collect a single text"""
        try:
            logger.info(f"Collecting: {url}")
            
            # Download text
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text
            
            # Basic cleaning
            text = text[:100000]  # Limit to 100K chars
            
            if len(text) < 100:
                logger.warning(f"Text too short: {len(text)} chars")
                return None
            
            # Extract title (simple heuristic)
            title = url.split('/')[-1][:100]
            
            # Count words
            word_count = len(text.split())
            
            logger.info(f"Downloaded: {word_count} words")
            
            return {
                'url': url,
                'title': title,
                'language': language,
                'content': text,
                'word_count': word_count,
                'date_collected': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect {url}: {e}")
            self.stats['errors'] += 1
            return None
    
    def generate_proiel(self, text_data: Dict) -> Dict:
        """Generate PROIEL treebank for text"""
        try:
            logger.info(f"Generating PROIEL for: {text_data['title']}")
            
            # Get pipeline
            nlp = self.get_pipeline(text_data['language'])
            if not nlp:
                logger.error(f"No pipeline available for {text_data['language']}")
                return None
            
            # Annotate
            doc = nlp(text_data['content'])
            
            # Generate PROIEL XML
            proiel_result = self.proiel_processor.annotate_proiel(
                text_data['content'],
                text_data['language']
            )
            
            if proiel_result and 'proiel_xml' in proiel_result:
                logger.info("PROIEL treebank generated successfully")
                
                # Extract valency patterns
                valency_patterns = proiel_result.get('valency_patterns', [])
                
                # Add to valency lexicon
                for pattern in valency_patterns:
                    self.valency_lexicon.add_entry(
                        verb=pattern.get('verb', ''),
                        lemma=pattern.get('lemma', ''),
                        pattern=pattern.get('pattern', ''),
                        language=text_data['language'],
                        period='modern',  # TODO: Detect period
                        frequency=pattern.get('frequency', 1),
                        source=text_data['title']
                    )
                
                self.stats['treebanks_generated'] += 1
                self.stats['words_processed'] += text_data['word_count']
                
                return {
                    'proiel_xml': proiel_result['proiel_xml'],
                    'valency_patterns': valency_patterns,
                    'statistics': proiel_result.get('statistics', {})
                }
            else:
                logger.warning("PROIEL generation returned no XML")
                return None
                
        except Exception as e:
            logger.error(f"PROIEL generation failed: {e}")
            self.stats['errors'] += 1
            return None
    
    def save_to_database(self, text_data: Dict, proiel_data: Dict):
        """Save text and treebank to database with detailed metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table with extended fields
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
                    dependencies_count INTEGER DEFAULT 0,
                    treebank_quality TEXT,
                    valency_patterns_count INTEGER DEFAULT 0
                )
            """)
            
            # Calculate metrics from PROIEL data
            tokens_count = 0
            lemmas_count = 0
            pos_tags_count = 0
            dependencies_count = 0
            annotation_score = 0
            metadata_quality = 0
            valency_patterns_count = 0
            
            if proiel_data:
                stats = proiel_data.get('statistics', {})
                tokens_count = stats.get('tokens', 0)
                lemmas_count = stats.get('lemmas', 0)
                pos_tags_count = stats.get('pos_tags', 0)
                dependencies_count = stats.get('dependencies', 0)
                valency_patterns_count = len(proiel_data.get('valency_patterns', []))
                
                # Calculate annotation score
                if tokens_count > 0:
                    lemma_coverage = (lemmas_count / tokens_count) * 100
                    pos_coverage = (pos_tags_count / tokens_count) * 100
                    dep_coverage = (dependencies_count / tokens_count) * 100
                    annotation_score = (lemma_coverage + pos_coverage + dep_coverage) / 3
                
                # Calculate metadata quality
                quality_checks = 0
                if text_data.get('title'): quality_checks += 1
                if text_data.get('language'): quality_checks += 1
                if text_data.get('word_count', 0) > 0: quality_checks += 1
                if text_data.get('date_collected'): quality_checks += 1
                metadata_quality = (quality_checks / 4) * 100

            treebank_quality = 'none'
            if tokens_count > 0 and annotation_score > 0:
                if tokens_count >= 10000 and annotation_score >= 90.0:
                    treebank_quality = 'excellent'
                elif tokens_count >= 2000 and annotation_score >= 75.0:
                    treebank_quality = 'good'
                else:
                    treebank_quality = 'partial'

            # Insert with all metrics
            cursor.execute("""
                INSERT OR REPLACE INTO corpus_items
                (url, title, language, content, proiel_xml, word_count, date_added, status,
                 metadata_quality, annotation_score, tokens_count, lemmas_count, 
                 pos_tags_count, dependencies_count, treebank_quality, valency_patterns_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_data['url'],
                text_data['title'],
                text_data['language'],
                text_data['content'],
                proiel_data['proiel_xml'] if proiel_data else None,
                text_data['word_count'],
                text_data['date_collected'],
                'completed' if proiel_data else 'pending',
                metadata_quality,
                annotation_score,
                tokens_count,
                lemmas_count,
                pos_tags_count,
                dependencies_count,
                treebank_quality,
                valency_patterns_count
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("="*80)
            logger.info(f"✓ TEXT SAVED TO DATABASE - ID: {cursor.lastrowid}")
            logger.info("="*80)
            logger.info(f"Title:              {text_data['title']}")
            logger.info(f"URL:                {text_data['url']}")
            logger.info(f"Language:           {text_data['language']}")
            logger.info(f"Status:             {('completed' if proiel_data else 'pending')}")
            logger.info(f"Word Count:         {text_data['word_count']:,}")
            logger.info(f"Date Collected:     {text_data['date_collected']}")
            logger.info("-"*80)
            logger.info("ANNOTATION METRICS:")
            logger.info("-"*80)
            logger.info(f"Tokens Count:       {tokens_count:,}")
            logger.info(f"Lemmas Count:       {lemmas_count:,}")
            logger.info(f"POS Tags Count:     {pos_tags_count:,}")
            logger.info(f"Dependencies Count: {dependencies_count:,}")
            logger.info(f"Annotation Score:   {annotation_score:.2f}%")
            logger.info(f"Metadata Quality:   {metadata_quality:.2f}%")
            logger.info("-"*80)
            if proiel_data:
                logger.info("PROIEL TREEBANK:")
                logger.info("-"*80)
                logger.info(f"XML Generated:      YES")
                logger.info(f"XML Size:           {len(proiel_data['proiel_xml'])} characters")
                logger.info(f"Valency Patterns:   {len(proiel_data.get('valency_patterns', []))} verbs")
                if proiel_data.get('valency_patterns'):
                    logger.info("Top Valency Patterns:")
                    for i, pattern in enumerate(proiel_data['valency_patterns'][:5], 1):
                        logger.info(f"  {i}. {pattern.get('lemma', 'unknown')}: {pattern.get('pattern', 'N/A')}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            self.stats['errors'] += 1
    
    def collection_cycle(self, languages: List[str], texts_per_cycle: int = 10):
        """Single collection cycle"""
        logger.info("\n" + "="*80)
        logger.info("NEW COLLECTION CYCLE STARTING")
        logger.info("="*80)
        logger.info(f"Target Languages:   {', '.join(languages)}")
        logger.info(f"Texts per Cycle:    {texts_per_cycle}")
        logger.info(f"Cycle Start Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Collected:    {self.stats['texts_collected']}")
        logger.info(f"Total Errors:       {self.stats['errors']}")
        logger.info("="*80 + "\n")
        
        collected_this_cycle = 0
        
        for language in languages:
            # Get repositories for this language
            repos = self.catalog.get_repositories_for_language(language)
            
            if not repos:
                logger.warning(f"No repositories configured for {language}")
                continue
            
            # Select random repository
            repo = random.choice(repos)
            
            logger.info("-"*80)
            logger.info(f"REPOSITORY SELECTED: {repo['name']}")
            logger.info(f"Language:           {language}")
            logger.info(f"URL:                {repo['url']}")
            logger.info(f"Available Texts:    {len(repo['texts'])}")
            logger.info("-"*80)
            
            # Select random texts from repository
            available_texts = [
                f"{repo['url']}{text}"
                for text in repo['texts']
                if self.memory.is_new(f"{repo['url']}{text}")
            ]
            
            logger.info(f"New (uncollected) texts available: {len(available_texts)}")
            
            if not available_texts:
                logger.warning(f"No new texts available from {repo['name']}")
                continue
            
            # Collect texts
            texts_to_collect = random.sample(available_texts, min(texts_per_cycle, len(available_texts)))
            logger.info(f"Selected {len(texts_to_collect)} texts for collection")
            logger.info("")
            
            for idx, url in enumerate(texts_to_collect, 1):
                logger.info("\n" + "█"*80)
                logger.info(f"COLLECTING TEXT {idx}/{len(texts_to_collect)}")
                logger.info("█"*80)
                logger.info(f"URL: {url}")
                logger.info(f"Repository: {repo['name']}")
                logger.info(f"Language: {language}")
                
                # Collect text
                logger.info("\n>>> STEP 1: DOWNLOADING TEXT...")
                text_data = self.collect_text(url, language)
                
                if not text_data:
                    logger.error(f"✗ Failed to collect text from {url}")
                    logger.info("█"*80 + "\n")
                    continue
                
                logger.info(f"✓ Downloaded successfully")
                logger.info(f"  Title: {text_data.get('title', 'Unknown')}")
                logger.info(f"  Words: {text_data.get('word_count', 0):,}")
                
                # Generate PROIEL
                logger.info("\n>>> STEP 2: GENERATING PROIEL TREEBANK...")
                proiel_data = self.generate_proiel(text_data)
                
                if proiel_data and proiel_data.get('proiel_xml'):
                    logger.info(f"✓ PROIEL treebank generated")
                    stats = proiel_data.get('statistics', {})
                    logger.info(f"  Sentences: {stats.get('sentences', 0)}")
                    logger.info(f"  Tokens: {stats.get('tokens', 0):,}")
                    logger.info(f"  Lemmas: {stats.get('lemmas', 0):,}")
                    logger.info(f"  POS Tags: {stats.get('pos_tags', 0):,}")
                    logger.info(f"  Dependencies: {stats.get('dependencies', 0):,}")
                    logger.info(f"  Verbs: {stats.get('verbs', 0)}")
                else:
                    logger.warning(f"✗ PROIEL generation failed or incomplete")
                
                # Save to database
                logger.info("\n>>> STEP 3: SAVING TO DATABASE...")
                self.save_to_database(text_data, proiel_data)
                
                # Update memory
                self.memory.mark_collected(url)
                
                # Update stats
                self.stats['texts_collected'] += 1
                collected_this_cycle += 1
                
                # Log progress
                logger.info("\n>>> CURRENT SESSION STATISTICS:")
                self.print_stats()
                
                logger.info("█"*80 + "\n")
                
                # Small delay to be respectful
                time.sleep(2)
        
        logger.info(f"Cycle complete: {collected_this_cycle} texts collected")
        
        return collected_this_cycle
    
    def print_stats(self):
        """Print current statistics"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = elapsed / 3600
        
        rate = self.stats['texts_collected'] / hours if hours > 0 else 0
        
        logger.info("="*70)
        logger.info("COLLECTION STATISTICS")
        logger.info("="*70)
        logger.info(f"Texts Collected: {self.stats['texts_collected']}")
        logger.info(f"Treebanks Generated: {self.stats['treebanks_generated']}")
        logger.info(f"Words Processed: {self.stats['words_processed']:,}")
        logger.info(f"Collection Rate: {rate:.2f} texts/hour")
        logger.info(f"Uptime: {hours:.2f} hours")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*70)
    
    def run_247(self, languages: List[str] = None, texts_per_cycle: int = 10, cycle_delay: int = 300):
        """Run 24/7 autonomous collection"""
        if languages is None:
            languages = ['grc', 'lat']
        
        logger.info("="*70)
        logger.info("STARTING 24/7 AUTONOMOUS COLLECTION")
        logger.info("="*70)
        logger.info(f"Languages: {', '.join(languages)}")
        logger.info(f"Texts per cycle: {texts_per_cycle}")
        logger.info(f"Cycle delay: {cycle_delay} seconds")
        logger.info("="*70)
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"\n>>> CYCLE {cycle_count} <<<\n")
                
                # Run collection cycle
                collected = self.collection_cycle(languages, texts_per_cycle)
                
                # Print stats
                self.print_stats()
                
                # Wait before next cycle
                logger.info(f"Waiting {cycle_delay} seconds before next cycle...")
                time.sleep(cycle_delay)
                
        except KeyboardInterrupt:
            logger.info("\n\nCollection stopped by user")
            self.print_stats()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.print_stats()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="24/7 Autonomous Collection with PROIEL")
    parser.add_argument('--languages', nargs='+', default=['grc', 'lat'],
                        help='Languages to collect (default: grc lat en)')
    parser.add_argument('--texts-per-cycle', type=int, default=10,
                        help='Texts to collect per cycle (default: 10)')
    parser.add_argument('--cycle-delay', type=int, default=300,
                        help='Delay between cycles in seconds (default: 300)')
    
    args = parser.parse_args()
    
    collector = Autonomous247Collector()
    collector.run_247(
        languages=args.languages,
        texts_per_cycle=args.texts_per_cycle,
        cycle_delay=args.cycle_delay
    )


if __name__ == "__main__":
    main()
