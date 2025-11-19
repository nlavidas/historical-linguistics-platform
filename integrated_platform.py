"""
═══════════════════════════════════════════════════════════════════════════
INTEGRATED HFRI-NKUA CORPUS PLATFORM
Combines unified platform with multi-AI annotation
═══════════════════════════════════════════════════════════════════════════

Author: Nikolaos Lavidas
Institution: National and Kapodistrian University of Athens (NKUA)
Funding: Hellenic Foundation for Research and Innovation (HFRI)
Version: 2.0.0
Date: November 9, 2025
═══════════════════════════════════════════════════════════════════════════
"""

# Load local models configuration (use Z:\models\ - no re-downloads)
try:
    import local_models_config
except ImportError:
    pass  # Fall back to default model locations if config not available

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from unified_corpus_platform import (
    UnifiedCorpusPlatform,
    UnifiedCorpusDatabase,
    AutomaticScraper,
    AutomaticParser
)
from multi_ai_annotator import MultiAIAnnotator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedAnnotator:
    """Enhanced annotator using multi-AI system"""
    
    def __init__(self, db: UnifiedCorpusDatabase):
        self.db = db
        self.multi_ai = MultiAIAnnotator()
    
    def annotate_text(self, file_path: str, item_id: int, 
                     language: str = "en") -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Annotate using multi-AI ensemble"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Use multi-AI ensemble
            logger.info(f"Running multi-AI ensemble for item {item_id}...")
            result = self.multi_ai.annotate_ensemble(text, language)
            
            if not result['success']:
                return False, None, result.get('error', 'Unknown error')
            
            # Save annotations
            annotation_path = Path(f"data/annotated/{item_id}_multi_ai.json")
            self.multi_ai.save_annotations(result, str(annotation_path))
            
            logger.info(f"✓ Multi-AI annotation complete: {result['models_used']} models used")
            
            return True, result, None
            
        except Exception as e:
            logger.error(f"Multi-AI annotation error for item {item_id}: {e}")
            return False, None, str(e)
    
    def process_queue(self, batch_size: int = 3) -> int:
        """Process annotation queue with multi-AI"""
        items = self.db.get_items_by_status('parsed', limit=batch_size)
        
        if not items:
            return 0
        
        logger.info(f"Processing {len(items)} items with multi-AI annotation")
        
        processed = 0
        for item in items:
            self.db.update_status(item['id'], 'queued')
            
            language = item.get('language') or 'en'
            success, annotations, error = self.annotate_text(
                item['parsed_path'], item['id'], language
            )
            
            if success:
                # Update database
                import sqlite3
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE corpus_items 
                    SET status = 'annotated', annotated_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), item['id']))
                conn.commit()
                conn.close()
                
                # Store annotation
                self.db.add_annotation(
                    item['id'],
                    'multi_ai_ensemble',
                    annotations,
                    f"data/annotated/{item['id']}_multi_ai.json",
                    annotations['total_processing_time'],
                    f"ensemble_{annotations['models_used']}_models"
                )
                processed += 1
            else:
                self.db.update_status(item['id'], 'failed', error)
        
        return processed


class IntegratedPlatform(UnifiedCorpusPlatform):
    """Enhanced platform with multi-AI annotation"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        super().__init__(db_path)
        # Replace standard annotator with multi-AI annotator
        self.annotator = IntegratedAnnotator(self.db)
        logger.info("=" * 70)
        logger.info("INTEGRATED HFRI-NKUA CORPUS PLATFORM INITIALIZED")
        logger.info("Multi-AI annotation system active")
        logger.info("=" * 70)


async def main():
    """Main entry point for integrated platform"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HFRI-NKUA Integrated AI Corpus Platform"
    )
    parser.add_argument('--db', default='corpus_platform.db', help='Database file')
    parser.add_argument('--add-urls', nargs='+', help='Add URLs to process')
    parser.add_argument('--source-type', default='custom_url', help='Source type')
    parser.add_argument('--language', help='Language code (grc, en, etc.)')
    parser.add_argument('--priority', type=int, default=5, help='Priority (1-10)')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--cycles', type=int, help='Processing cycles (0=infinite)')
    parser.add_argument('--delay', type=int, default=30, help='Seconds between cycles')
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = IntegratedPlatform(args.db)
    
    # Add URLs if provided
    if args.add_urls:
        platform.add_source_urls(
            args.add_urls,
            source_type=args.source_type,
            priority=args.priority,
            language=args.language
        )
    
    # Show status if requested
    if args.status:
        platform.show_status()
        return
    
    # Run pipeline
    await platform.run_pipeline(cycles=args.cycles, cycle_delay=args.delay)


if __name__ == "__main__":
    print("=" * 70)
    print("HFRI-NKUA INTEGRATED CORPUS PLATFORM")
    print("Multi-AI Annotation System")
    print("=" * 70)
    print()
    asyncio.run(main())
