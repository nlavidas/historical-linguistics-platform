"""
Translation Tracker - Track intralingual, interlingual, and diachronic translations

This module provides comprehensive translation tracking for:
- Intralingual translations (same language, different periods - e.g., Planudes)
- Interlingual translations (between languages)
- Diachronic translation chains (evolution across time)
- Retranslations and retellings

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TranslationType(Enum):
    INTRALINGUAL = "intralingual"
    INTERLINGUAL = "interlingual"
    DIACHRONIC = "diachronic"
    RETRANSLATION = "retranslation"
    RETELLING = "retelling"
    ADAPTATION = "adaptation"
    MODERNIZATION = "modernization"


@dataclass
class TranslationPair:
    source_id: int
    target_id: int
    source_title: str
    target_title: str
    source_language: str
    target_language: str
    source_period: str
    target_period: str
    translation_type: TranslationType
    translator: str = ""
    translation_year: Optional[int] = None
    notes: str = ""
    alignment_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'source_title': self.source_title,
            'target_title': self.target_title,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'source_period': self.source_period,
            'target_period': self.target_period,
            'translation_type': self.translation_type.value,
            'translator': self.translator,
            'translation_year': self.translation_year,
            'notes': self.notes,
            'alignment_score': self.alignment_score,
            'metadata': self.metadata,
        }


@dataclass
class TranslationChain:
    chain_id: str
    original_title: str
    original_language: str
    original_period: str
    translations: List[TranslationPair] = field(default_factory=list)
    total_languages: int = 0
    total_periods: int = 0
    time_span_years: int = 0
    
    def __post_init__(self):
        if self.translations:
            languages = set()
            periods = set()
            years = []
            
            for t in self.translations:
                languages.add(t.source_language)
                languages.add(t.target_language)
                periods.add(t.source_period)
                periods.add(t.target_period)
                if t.translation_year:
                    years.append(t.translation_year)
            
            self.total_languages = len(languages)
            self.total_periods = len(periods)
            if years:
                self.time_span_years = max(years) - min(years)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain_id': self.chain_id,
            'original_title': self.original_title,
            'original_language': self.original_language,
            'original_period': self.original_period,
            'translations': [t.to_dict() for t in self.translations],
            'total_languages': self.total_languages,
            'total_periods': self.total_periods,
            'time_span_years': self.time_span_years,
        }


class TranslationTracker:
    
    KNOWN_TRANSLATION_CHAINS = {
        'iliad': {
            'original_title': 'Iliad',
            'original_author': 'Homer',
            'original_language': 'grc',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'lat', 'period': 'Ancient', 'translator': 'Various', 'type': 'interlingual'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'George Chapman', 'year': 1616, 'type': 'interlingual'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'Alexander Pope', 'year': 1720, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Samuel Butler', 'year': 1898, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Richmond Lattimore', 'year': 1951, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Robert Fagles', 'year': 1990, 'type': 'retranslation'},
                {'language': 'el', 'period': 'Modern', 'translator': 'Various', 'type': 'intralingual'},
            ],
        },
        'odyssey': {
            'original_title': 'Odyssey',
            'original_author': 'Homer',
            'original_language': 'grc',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'lat', 'period': 'Ancient', 'translator': 'Various', 'type': 'interlingual'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'George Chapman', 'year': 1616, 'type': 'interlingual'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Samuel Butler', 'year': 1900, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Robert Fitzgerald', 'year': 1961, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Emily Wilson', 'year': 2017, 'type': 'retranslation'},
            ],
        },
        'bible': {
            'original_title': 'Bible',
            'original_author': 'Various',
            'original_language': 'grc',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'lat', 'period': 'Ancient', 'translator': 'Jerome (Vulgate)', 'year': 405, 'type': 'interlingual'},
                {'language': 'ang', 'period': 'Medieval', 'translator': 'Various', 'type': 'interlingual'},
                {'language': 'enm', 'period': 'Medieval', 'translator': 'John Wycliffe', 'year': 1382, 'type': 'interlingual'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'William Tyndale', 'year': 1526, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'King James', 'year': 1611, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'RSV', 'year': 1952, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'NIV', 'year': 1978, 'type': 'retranslation'},
            ],
        },
        'aeneid': {
            'original_title': 'Aeneid',
            'original_author': 'Virgil',
            'original_language': 'lat',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'en', 'period': 'Early Modern', 'translator': 'John Dryden', 'year': 1697, 'type': 'interlingual'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Robert Fitzgerald', 'year': 1983, 'type': 'retranslation'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Robert Fagles', 'year': 2006, 'type': 'retranslation'},
            ],
        },
        'planudes_aesop': {
            'original_title': 'Aesop Fables',
            'original_author': 'Aesop',
            'original_language': 'grc',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'grc', 'period': 'Byzantine', 'translator': 'Maximus Planudes', 'year': 1300, 'type': 'intralingual', 'notes': 'Byzantine Greek prose version'},
                {'language': 'lat', 'period': 'Medieval', 'translator': 'Various', 'type': 'interlingual'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Various', 'type': 'interlingual'},
            ],
        },
        'planudes_ovid': {
            'original_title': 'Metamorphoses',
            'original_author': 'Ovid',
            'original_language': 'lat',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'grc', 'period': 'Byzantine', 'translator': 'Maximus Planudes', 'year': 1290, 'type': 'interlingual', 'notes': 'Byzantine Greek translation'},
                {'language': 'en', 'period': 'Early Modern', 'translator': 'Arthur Golding', 'year': 1567, 'type': 'interlingual'},
                {'language': 'en', 'period': 'Modern', 'translator': 'Various', 'type': 'retranslation'},
            ],
        },
        'planudes_cato': {
            'original_title': 'Disticha Catonis',
            'original_author': 'Pseudo-Cato',
            'original_language': 'lat',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'grc', 'period': 'Byzantine', 'translator': 'Maximus Planudes', 'year': 1295, 'type': 'interlingual', 'notes': 'Byzantine Greek translation'},
            ],
        },
        'planudes_boethius': {
            'original_title': 'Consolation of Philosophy',
            'original_author': 'Boethius',
            'original_language': 'lat',
            'original_period': 'Late Ancient',
            'translations': [
                {'language': 'grc', 'period': 'Byzantine', 'translator': 'Maximus Planudes', 'year': 1295, 'type': 'interlingual', 'notes': 'Byzantine Greek translation'},
                {'language': 'ang', 'period': 'Medieval', 'translator': 'King Alfred', 'year': 888, 'type': 'interlingual'},
                {'language': 'enm', 'period': 'Medieval', 'translator': 'Geoffrey Chaucer', 'year': 1380, 'type': 'interlingual'},
            ],
        },
        'septuagint': {
            'original_title': 'Hebrew Bible',
            'original_author': 'Various',
            'original_language': 'heb',
            'original_period': 'Ancient',
            'translations': [
                {'language': 'grc', 'period': 'Ancient', 'translator': 'Seventy Translators', 'year': -250, 'type': 'interlingual', 'notes': 'Septuagint - Koine Greek translation'},
                {'language': 'lat', 'period': 'Ancient', 'translator': 'Jerome', 'year': 405, 'type': 'interlingual'},
            ],
        },
    }
    
    def __init__(self, db_path: str = "data/corpus_platform.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()
    
    def _ensure_tables(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS translation_pairs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id INTEGER,
                        target_id INTEGER,
                        source_title TEXT,
                        target_title TEXT,
                        source_language TEXT,
                        target_language TEXT,
                        source_period TEXT,
                        target_period TEXT,
                        translation_type TEXT,
                        translator TEXT,
                        translation_year INTEGER,
                        notes TEXT,
                        alignment_score REAL DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source_id, target_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS translation_chains (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chain_id TEXT UNIQUE,
                        original_title TEXT,
                        original_language TEXT,
                        original_period TEXT,
                        chain_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_source ON translation_pairs(source_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_target ON translation_pairs(target_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_type ON translation_pairs(translation_type)")
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating translation tables: {e}")
    
    def add_translation_pair(self, pair: TranslationPair) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO translation_pairs
                    (source_id, target_id, source_title, target_title, source_language,
                     target_language, source_period, target_period, translation_type,
                     translator, translation_year, notes, alignment_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair.source_id,
                    pair.target_id,
                    pair.source_title,
                    pair.target_title,
                    pair.source_language,
                    pair.target_language,
                    pair.source_period,
                    pair.target_period,
                    pair.translation_type.value,
                    pair.translator,
                    pair.translation_year,
                    pair.notes,
                    pair.alignment_score,
                    json.dumps(pair.metadata),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding translation pair: {e}")
            return False
    
    def get_translations_for_text(self, text_id: int) -> List[TranslationPair]:
        pairs = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM translation_pairs
                    WHERE source_id = ? OR target_id = ?
                """, (text_id, text_id))
                
                for row in cursor:
                    pairs.append(TranslationPair(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        source_title=row['source_title'],
                        target_title=row['target_title'],
                        source_language=row['source_language'],
                        target_language=row['target_language'],
                        source_period=row['source_period'],
                        target_period=row['target_period'],
                        translation_type=TranslationType(row['translation_type']),
                        translator=row['translator'] or '',
                        translation_year=row['translation_year'],
                        notes=row['notes'] or '',
                        alignment_score=row['alignment_score'] or 0,
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    ))
        except Exception as e:
            logger.error(f"Error getting translations: {e}")
        
        return pairs
    
    def get_translation_chain(self, chain_id: str) -> Optional[TranslationChain]:
        if chain_id in self.KNOWN_TRANSLATION_CHAINS:
            chain_data = self.KNOWN_TRANSLATION_CHAINS[chain_id]
            
            translations = []
            for t in chain_data.get('translations', []):
                translations.append(TranslationPair(
                    source_id=0,
                    target_id=0,
                    source_title=chain_data['original_title'],
                    target_title=f"{chain_data['original_title']} ({t['translator']})",
                    source_language=chain_data['original_language'],
                    target_language=t['language'],
                    source_period=chain_data['original_period'],
                    target_period=t['period'],
                    translation_type=TranslationType(t['type']),
                    translator=t['translator'],
                    translation_year=t.get('year'),
                    notes=t.get('notes', ''),
                ))
            
            return TranslationChain(
                chain_id=chain_id,
                original_title=chain_data['original_title'],
                original_language=chain_data['original_language'],
                original_period=chain_data['original_period'],
                translations=translations,
            )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT * FROM translation_chains WHERE chain_id = ?",
                    (chain_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    chain_data = json.loads(row['chain_data'])
                    return TranslationChain(**chain_data)
        except Exception as e:
            logger.error(f"Error getting translation chain: {e}")
        
        return None
    
    def get_all_chains(self) -> List[TranslationChain]:
        chains = []
        
        for chain_id in self.KNOWN_TRANSLATION_CHAINS:
            chain = self.get_translation_chain(chain_id)
            if chain:
                chains.append(chain)
        
        return chains
    
    def get_intralingual_translations(self) -> List[TranslationPair]:
        pairs = []
        
        for chain_id, chain_data in self.KNOWN_TRANSLATION_CHAINS.items():
            for t in chain_data.get('translations', []):
                if t['type'] == 'intralingual':
                    pairs.append(TranslationPair(
                        source_id=0,
                        target_id=0,
                        source_title=chain_data['original_title'],
                        target_title=f"{chain_data['original_title']} ({t['translator']})",
                        source_language=chain_data['original_language'],
                        target_language=t['language'],
                        source_period=chain_data['original_period'],
                        target_period=t['period'],
                        translation_type=TranslationType.INTRALINGUAL,
                        translator=t['translator'],
                        translation_year=t.get('year'),
                        notes=t.get('notes', ''),
                    ))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM translation_pairs
                    WHERE translation_type = 'intralingual'
                """)
                
                for row in cursor:
                    pairs.append(TranslationPair(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        source_title=row['source_title'],
                        target_title=row['target_title'],
                        source_language=row['source_language'],
                        target_language=row['target_language'],
                        source_period=row['source_period'],
                        target_period=row['target_period'],
                        translation_type=TranslationType.INTRALINGUAL,
                        translator=row['translator'] or '',
                        translation_year=row['translation_year'],
                        notes=row['notes'] or '',
                    ))
        except Exception as e:
            logger.error(f"Error getting intralingual translations: {e}")
        
        return pairs
    
    def get_planudes_translations(self) -> List[TranslationPair]:
        planudes_pairs = []
        
        for chain_id, chain_data in self.KNOWN_TRANSLATION_CHAINS.items():
            if 'planudes' in chain_id:
                for t in chain_data.get('translations', []):
                    if 'Planudes' in t['translator']:
                        planudes_pairs.append(TranslationPair(
                            source_id=0,
                            target_id=0,
                            source_title=chain_data['original_title'],
                            target_title=f"{chain_data['original_title']} (Planudes)",
                            source_language=chain_data['original_language'],
                            target_language=t['language'],
                            source_period=chain_data['original_period'],
                            target_period=t['period'],
                            translation_type=TranslationType(t['type']),
                            translator=t['translator'],
                            translation_year=t.get('year'),
                            notes=t.get('notes', ''),
                        ))
        
        return planudes_pairs
    
    def auto_detect_translations(self) -> int:
        detected = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT id, title, language, period, author, translator,
                           original_language, is_retranslation
                    FROM corpus_items
                    WHERE is_retranslation = 1 OR translator IS NOT NULL
                """)
                
                translations = list(cursor)
                
                for trans in translations:
                    if trans['original_language']:
                        originals = conn.execute("""
                            SELECT id, title, language, period, author
                            FROM corpus_items
                            WHERE language = ? AND title LIKE ?
                            LIMIT 1
                        """, (trans['original_language'], f"%{trans['title'].split('-')[0].strip()}%"))
                        
                        original = originals.fetchone()
                        if original:
                            pair = TranslationPair(
                                source_id=original['id'],
                                target_id=trans['id'],
                                source_title=original['title'],
                                target_title=trans['title'],
                                source_language=original['language'],
                                target_language=trans['language'],
                                source_period=original['period'] or '',
                                target_period=trans['period'] or '',
                                translation_type=TranslationType.RETRANSLATION if trans['is_retranslation'] else TranslationType.INTERLINGUAL,
                                translator=trans['translator'] or '',
                            )
                            
                            if self.add_translation_pair(pair):
                                detected += 1
        
        except Exception as e:
            logger.error(f"Error auto-detecting translations: {e}")
        
        logger.info(f"Auto-detected {detected} translation pairs")
        return detected
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_pairs': 0,
            'by_type': {},
            'by_source_language': {},
            'by_target_language': {},
            'known_chains': len(self.KNOWN_TRANSLATION_CHAINS),
            'planudes_translations': len(self.get_planudes_translations()),
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats['total_pairs'] = conn.execute(
                    "SELECT COUNT(*) FROM translation_pairs"
                ).fetchone()[0]
                
                for row in conn.execute(
                    "SELECT translation_type, COUNT(*) FROM translation_pairs GROUP BY translation_type"
                ):
                    stats['by_type'][row[0]] = row[1]
                
                for row in conn.execute(
                    "SELECT source_language, COUNT(*) FROM translation_pairs GROUP BY source_language"
                ):
                    stats['by_source_language'][row[0]] = row[1]
                
                for row in conn.execute(
                    "SELECT target_language, COUNT(*) FROM translation_pairs GROUP BY target_language"
                ):
                    stats['by_target_language'][row[0]] = row[1]
        
        except Exception as e:
            logger.error(f"Error getting translation statistics: {e}")
        
        return stats
