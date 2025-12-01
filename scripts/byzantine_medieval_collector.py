#!/usr/bin/env python3
"""
Byzantine and Medieval Greek Text Collector
Comprehensive collection of understudied Greek periods

Sources:
- Thesaurus Linguae Graecae (TLG)
- Perseus Digital Library
- Open Greek and Latin
- Dumbarton Oaks Medieval Library
- Byzantine texts from various archives

Periods covered:
- Late Antique Greek (300-600 CE)
- Byzantine Greek (600-1453 CE)
- Medieval Greek (1100-1453 CE)
- Early Modern Greek (1453-1800 CE)
"""

import os
import re
import json
import sqlite3
import logging
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# BYZANTINE AND MEDIEVAL GREEK CORPUS CATALOG
# ============================================================================

BYZANTINE_CORPUS = {
    # Historical Works
    "anna_comnena": {
        "title": "Alexiad",
        "author": "Anna Comnena",
        "period": "byzantine",
        "dates": "1148 CE",
        "genre": "history",
        "language": "learned_byzantine",
        "tokens_estimate": 120000,
        "description": "History of the reign of Alexios I Komnenos",
        "sources": ["perseus", "tlg"]
    },
    "michael_psellus": {
        "title": "Chronographia",
        "author": "Michael Psellus",
        "period": "byzantine",
        "dates": "1078 CE",
        "genre": "history",
        "language": "learned_byzantine",
        "tokens_estimate": 80000,
        "description": "Byzantine history 976-1078",
        "sources": ["tlg"]
    },
    "procopius": {
        "title": "Secret History (Anecdota)",
        "author": "Procopius",
        "period": "late_antique",
        "dates": "550 CE",
        "genre": "history",
        "language": "late_antique",
        "tokens_estimate": 45000,
        "sources": ["perseus", "tlg"]
    },
    "constantine_porphyrogennetos": {
        "title": "De Administrando Imperio",
        "author": "Constantine VII Porphyrogennetos",
        "period": "byzantine",
        "dates": "950 CE",
        "genre": "administrative",
        "language": "learned_byzantine",
        "tokens_estimate": 60000,
        "sources": ["tlg"]
    },
    
    # Chronicle of Morea - Vernacular
    "chronicle_morea": {
        "title": "Chronicle of Morea",
        "author": "Anonymous",
        "period": "medieval",
        "dates": "1300-1350 CE",
        "genre": "chronicle",
        "language": "vernacular_medieval",
        "tokens_estimate": 45000,
        "description": "Frankish conquest of Peloponnese in vernacular Greek",
        "sources": ["open_greek_latin"]
    },
    
    # Digenes Akritas - Epic
    "digenes_akritas": {
        "title": "Digenes Akritas",
        "author": "Anonymous",
        "period": "medieval",
        "dates": "1100-1200 CE",
        "genre": "epic",
        "language": "vernacular_medieval",
        "tokens_estimate": 35000,
        "description": "Byzantine frontier epic",
        "sources": ["tlg", "open_greek_latin"]
    },
    
    # Ptochoprodromos - Satirical
    "ptochoprodromos": {
        "title": "Ptochoprodromic Poems",
        "author": "Ptochoprodromos",
        "period": "medieval",
        "dates": "1100-1150 CE",
        "genre": "satire",
        "language": "vernacular_medieval",
        "tokens_estimate": 15000,
        "description": "Satirical poems in vernacular Greek",
        "sources": ["tlg"]
    },
    
    # Planudes Translations
    "planudes_ovid": {
        "title": "Metamorphoses (Greek translation)",
        "author": "Maximus Planudes",
        "period": "byzantine",
        "dates": "1280 CE",
        "genre": "translation",
        "language": "learned_byzantine",
        "tokens_estimate": 90000,
        "description": "Greek translation of Ovid's Metamorphoses",
        "sources": ["tlg"]
    },
    "planudes_cato": {
        "title": "Disticha Catonis (Greek translation)",
        "author": "Maximus Planudes",
        "period": "byzantine",
        "dates": "1280 CE",
        "genre": "translation",
        "language": "learned_byzantine",
        "tokens_estimate": 8000,
        "sources": ["tlg"]
    },
    "planudes_boethius": {
        "title": "Consolation of Philosophy (Greek translation)",
        "author": "Maximus Planudes",
        "period": "byzantine",
        "dates": "1280 CE",
        "genre": "translation",
        "language": "learned_byzantine",
        "tokens_estimate": 35000,
        "sources": ["tlg"]
    },
    
    # Church Fathers and Theological
    "john_chrysostom": {
        "title": "Homilies",
        "author": "John Chrysostom",
        "period": "late_antique",
        "dates": "400 CE",
        "genre": "homily",
        "language": "patristic",
        "tokens_estimate": 500000,
        "sources": ["tlg", "perseus"]
    },
    "basil_caesarea": {
        "title": "Hexaemeron",
        "author": "Basil of Caesarea",
        "period": "late_antique",
        "dates": "370 CE",
        "genre": "theology",
        "language": "patristic",
        "tokens_estimate": 40000,
        "sources": ["tlg"]
    },
    "gregory_nazianzus": {
        "title": "Orations",
        "author": "Gregory of Nazianzus",
        "period": "late_antique",
        "dates": "380 CE",
        "genre": "oratory",
        "language": "patristic",
        "tokens_estimate": 120000,
        "sources": ["tlg"]
    },
    
    # Hagiography
    "life_antony": {
        "title": "Life of Antony",
        "author": "Athanasius of Alexandria",
        "period": "late_antique",
        "dates": "360 CE",
        "genre": "hagiography",
        "language": "patristic",
        "tokens_estimate": 25000,
        "sources": ["tlg"]
    },
    
    # Early Modern Greek
    "erotokritos": {
        "title": "Erotokritos",
        "author": "Vitsentzos Kornaros",
        "period": "early_modern",
        "dates": "1600-1670 CE",
        "genre": "romance",
        "language": "cretan",
        "tokens_estimate": 50000,
        "description": "Cretan Renaissance masterpiece",
        "sources": ["open_greek_latin"]
    },
    "sacrifice_abraham": {
        "title": "The Sacrifice of Abraham",
        "author": "Vitsentzos Kornaros",
        "period": "early_modern",
        "dates": "1635 CE",
        "genre": "drama",
        "language": "cretan",
        "tokens_estimate": 8000,
        "sources": ["open_greek_latin"]
    },
    "erophile": {
        "title": "Erophile",
        "author": "Georgios Chortatsis",
        "period": "early_modern",
        "dates": "1600 CE",
        "genre": "tragedy",
        "language": "cretan",
        "tokens_estimate": 12000,
        "sources": ["open_greek_latin"]
    }
}

# Language register classification
LANGUAGE_REGISTERS = {
    "learned_byzantine": {
        "description": "Classicizing Byzantine Greek",
        "features": ["attic_forms", "complex_syntax", "literary_vocabulary"]
    },
    "vernacular_medieval": {
        "description": "Vernacular Medieval Greek",
        "features": ["simplified_morphology", "romance_loanwords", "popular_syntax"]
    },
    "patristic": {
        "description": "Patristic Greek",
        "features": ["koine_base", "theological_vocabulary", "biblical_influence"]
    },
    "cretan": {
        "description": "Cretan Greek",
        "features": ["venetian_influence", "local_dialect", "italian_loanwords"]
    },
    "late_antique": {
        "description": "Late Antique Greek",
        "features": ["koine_evolution", "latin_influence", "christian_vocabulary"]
    }
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ByzantineText:
    """Byzantine/Medieval text entry"""
    id: str
    title: str
    author: str
    period: str
    dates: str
    genre: str
    language_register: str
    content: str = ""
    tokens: int = 0
    sentences: int = 0
    source: str = ""
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# COLLECTOR
# ============================================================================

class ByzantineMedievalCollector:
    """Collector for Byzantine and Medieval Greek texts"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self.corpus = BYZANTINE_CORPUS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreekCorpusPlatform/3.0 (Academic Research)'
        })
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS byzantine_texts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                dates TEXT,
                genre TEXT,
                language_register TEXT,
                content TEXT,
                token_count INTEGER DEFAULT 0,
                sentence_count INTEGER DEFAULT 0,
                source TEXT,
                metadata TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id TEXT,
                action TEXT,
                status TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_catalog(self) -> Dict:
        """Get full catalog of available texts"""
        return self.corpus
    
    def get_by_period(self, period: str) -> Dict:
        """Get texts by period"""
        return {k: v for k, v in self.corpus.items() if v.get('period') == period}
    
    def get_by_genre(self, genre: str) -> Dict:
        """Get texts by genre"""
        return {k: v for k, v in self.corpus.items() if v.get('genre') == genre}
    
    def collect_from_perseus(self, text_id: str) -> Optional[str]:
        """Collect text from Perseus Digital Library"""
        if text_id not in self.corpus:
            logger.error(f"Unknown text: {text_id}")
            return None
        
        text_info = self.corpus[text_id]
        
        # Perseus API endpoints vary by text
        # This is a placeholder for actual Perseus integration
        logger.info(f"Collecting {text_info['title']} from Perseus...")
        
        # Log the attempt
        self._log_action(text_id, "collect", "attempted", "Perseus collection initiated")
        
        return None  # Placeholder
    
    def collect_from_open_greek_latin(self, text_id: str) -> Optional[str]:
        """Collect from Open Greek and Latin project"""
        if text_id not in self.corpus:
            return None
        
        text_info = self.corpus[text_id]
        logger.info(f"Collecting {text_info['title']} from Open Greek and Latin...")
        
        # GitHub-based collection
        base_url = "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data"
        
        self._log_action(text_id, "collect", "attempted", "OGL collection initiated")
        
        return None  # Placeholder
    
    def save_text(self, text: ByzantineText) -> bool:
        """Save collected text to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO byzantine_texts
                (id, title, author, period, dates, genre, language_register,
                 content, token_count, sentence_count, source, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text.id, text.title, text.author, text.period, text.dates,
                text.genre, text.language_register, text.content, text.tokens,
                text.sentences, text.source, json.dumps(text.metadata), 'collected'
            ))
            
            conn.commit()
            conn.close()
            
            self._log_action(text.id, "save", "success", f"Saved {text.tokens} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Error saving text: {e}")
            self._log_action(text.id, "save", "error", str(e))
            return False
    
    def _log_action(self, text_id: str, action: str, status: str, message: str):
        """Log collection action"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collection_log (text_id, action, status, message)
                VALUES (?, ?, ?, ?)
            """, (text_id, action, status, message))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging action: {e}")
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        stats = {
            "total_catalog": len(self.corpus),
            "by_period": {},
            "by_genre": {},
            "by_register": {},
            "total_tokens_estimate": 0
        }
        
        for text_id, text_info in self.corpus.items():
            period = text_info.get('period', 'unknown')
            genre = text_info.get('genre', 'unknown')
            register = text_info.get('language', 'unknown')
            tokens = text_info.get('tokens_estimate', 0)
            
            stats["by_period"][period] = stats["by_period"].get(period, 0) + 1
            stats["by_genre"][genre] = stats["by_genre"].get(genre, 0) + 1
            stats["by_register"][register] = stats["by_register"].get(register, 0) + 1
            stats["total_tokens_estimate"] += tokens
        
        return stats
    
    def export_catalog(self, output_path: str):
        """Export catalog to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "corpus": self.corpus,
                "registers": LANGUAGE_REGISTERS,
                "statistics": self.get_statistics()
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported catalog to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Byzantine/Medieval Greek Collector")
    parser.add_argument('command', choices=['catalog', 'stats', 'collect', 'export'],
                       help="Command to run")
    parser.add_argument('--period', '-p', help="Filter by period")
    parser.add_argument('--genre', '-g', help="Filter by genre")
    parser.add_argument('--text', '-t', help="Specific text ID")
    parser.add_argument('--output', '-o', help="Output file")
    
    args = parser.parse_args()
    
    collector = ByzantineMedievalCollector()
    
    if args.command == 'catalog':
        if args.period:
            texts = collector.get_by_period(args.period)
        elif args.genre:
            texts = collector.get_by_genre(args.genre)
        else:
            texts = collector.get_catalog()
        
        for text_id, info in texts.items():
            print(f"{text_id}: {info['title']} ({info['author']}) - {info['period']}")
    
    elif args.command == 'stats':
        stats = collector.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'collect':
        if args.text:
            collector.collect_from_perseus(args.text)
        else:
            print("Please specify --text")
    
    elif args.command == 'export':
        output = args.output or "byzantine_catalog.json"
        collector.export_catalog(output)


if __name__ == "__main__":
    main()
