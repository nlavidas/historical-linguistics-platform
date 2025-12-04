"""
Timeline Generator - Create timelines for diachronic linguistics visualization

This module generates timeline data for visualizing:
- Historical periods of texts
- Language evolution
- Translation chains over time
- Linguistic change events

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    id: str
    title: str
    description: str
    start_year: int
    end_year: Optional[int] = None
    category: str = "text"
    language: str = ""
    author: str = ""
    importance: int = 1
    color: str = "#3498db"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'start_year': self.start_year,
            'end_year': self.end_year or self.start_year,
            'category': self.category,
            'language': self.language,
            'author': self.author,
            'importance': self.importance,
            'color': self.color,
            'metadata': self.metadata,
        }


@dataclass
class TimelinePeriod:
    id: str
    name: str
    start_year: int
    end_year: int
    description: str = ""
    color: str = "#ecf0f1"
    languages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'description': self.description,
            'color': self.color,
            'languages': self.languages,
        }


class TimelineGenerator:
    
    HISTORICAL_PERIODS = [
        TimelinePeriod(
            id='archaic_greek',
            name='Archaic Greek',
            start_year=-800,
            end_year=-500,
            description='Homer, Hesiod, early lyric poetry',
            color='#e74c3c',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='classical_greek',
            name='Classical Greek',
            start_year=-500,
            end_year=-323,
            description='Attic drama, philosophy, historiography',
            color='#e67e22',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='hellenistic',
            name='Hellenistic Period',
            start_year=-323,
            end_year=-31,
            description='Septuagint, Polybius, Koine development',
            color='#f1c40f',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='roman_imperial',
            name='Roman Imperial',
            start_year=-31,
            end_year=284,
            description='New Testament, Plutarch, Marcus Aurelius',
            color='#2ecc71',
            languages=['grc', 'lat'],
        ),
        TimelinePeriod(
            id='late_antiquity',
            name='Late Antiquity',
            start_year=284,
            end_year=600,
            description='Church Fathers, Vulgate, Boethius',
            color='#1abc9c',
            languages=['grc', 'lat'],
        ),
        TimelinePeriod(
            id='early_byzantine',
            name='Early Byzantine',
            start_year=600,
            end_year=850,
            description='Malalas, early chronicles',
            color='#3498db',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='middle_byzantine',
            name='Middle Byzantine',
            start_year=850,
            end_year=1204,
            description='Psellos, Anna Komnene, Digenis Akritas',
            color='#9b59b6',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='late_byzantine',
            name='Late Byzantine',
            start_year=1204,
            end_year=1453,
            description='Planudes, Chronicle of Morea',
            color='#8e44ad',
            languages=['grc'],
        ),
        TimelinePeriod(
            id='old_english',
            name='Old English',
            start_year=450,
            end_year=1100,
            description='Beowulf, Alfred translations',
            color='#34495e',
            languages=['ang'],
        ),
        TimelinePeriod(
            id='middle_english',
            name='Middle English',
            start_year=1100,
            end_year=1500,
            description='Chaucer, Wycliffe Bible',
            color='#7f8c8d',
            languages=['enm'],
        ),
        TimelinePeriod(
            id='early_modern',
            name='Early Modern',
            start_year=1500,
            end_year=1700,
            description='Shakespeare, KJV, Milton',
            color='#95a5a6',
            languages=['en'],
        ),
        TimelinePeriod(
            id='modern',
            name='Modern Period',
            start_year=1700,
            end_year=2025,
            description='Modern translations and scholarship',
            color='#bdc3c7',
            languages=['en', 'el'],
        ),
    ]
    
    MAJOR_TEXTS = [
        TimelineEvent(
            id='homer_iliad',
            title='Iliad',
            description='Homer epic on the Trojan War',
            start_year=-750,
            category='epic',
            language='grc',
            author='Homer',
            importance=10,
            color='#e74c3c',
        ),
        TimelineEvent(
            id='homer_odyssey',
            title='Odyssey',
            description='Homer epic on Odysseus return',
            start_year=-725,
            category='epic',
            language='grc',
            author='Homer',
            importance=10,
            color='#e74c3c',
        ),
        TimelineEvent(
            id='hesiod_theogony',
            title='Theogony',
            description='Genealogy of the gods',
            start_year=-700,
            category='religious',
            language='grc',
            author='Hesiod',
            importance=8,
            color='#e74c3c',
        ),
        TimelineEvent(
            id='herodotus_histories',
            title='Histories',
            description='First major Greek prose history',
            start_year=-440,
            category='history',
            language='grc',
            author='Herodotus',
            importance=9,
            color='#e67e22',
        ),
        TimelineEvent(
            id='thucydides',
            title='Peloponnesian War',
            description='Analytical history of Athens vs Sparta',
            start_year=-411,
            category='history',
            language='grc',
            author='Thucydides',
            importance=9,
            color='#e67e22',
        ),
        TimelineEvent(
            id='sophocles_oedipus',
            title='Oedipus Tyrannus',
            description='Tragedy of Oedipus',
            start_year=-429,
            category='drama',
            language='grc',
            author='Sophocles',
            importance=9,
            color='#e67e22',
        ),
        TimelineEvent(
            id='plato_republic',
            title='Republic',
            description='Dialogue on justice and the ideal state',
            start_year=-375,
            category='philosophy',
            language='grc',
            author='Plato',
            importance=10,
            color='#e67e22',
        ),
        TimelineEvent(
            id='aristotle_poetics',
            title='Poetics',
            description='Treatise on dramatic theory',
            start_year=-335,
            category='philosophy',
            language='grc',
            author='Aristotle',
            importance=9,
            color='#e67e22',
        ),
        TimelineEvent(
            id='septuagint',
            title='Septuagint',
            description='Greek translation of Hebrew Bible',
            start_year=-250,
            end_year=-100,
            category='religious',
            language='grc',
            author='Seventy Translators',
            importance=10,
            color='#f1c40f',
        ),
        TimelineEvent(
            id='virgil_aeneid',
            title='Aeneid',
            description='Roman national epic',
            start_year=-19,
            category='epic',
            language='lat',
            author='Virgil',
            importance=10,
            color='#2ecc71',
        ),
        TimelineEvent(
            id='new_testament',
            title='New Testament',
            description='Christian scriptures in Koine Greek',
            start_year=50,
            end_year=100,
            category='religious',
            language='grc',
            author='Various',
            importance=10,
            color='#2ecc71',
        ),
        TimelineEvent(
            id='marcus_aurelius',
            title='Meditations',
            description='Stoic philosophical reflections',
            start_year=170,
            category='philosophy',
            language='grc',
            author='Marcus Aurelius',
            importance=8,
            color='#2ecc71',
        ),
        TimelineEvent(
            id='vulgate',
            title='Vulgate Bible',
            description='Jerome Latin translation',
            start_year=405,
            category='religious',
            language='lat',
            author='Jerome',
            importance=10,
            color='#1abc9c',
        ),
        TimelineEvent(
            id='boethius',
            title='Consolation of Philosophy',
            description='Philosophical dialogue',
            start_year=524,
            category='philosophy',
            language='lat',
            author='Boethius',
            importance=9,
            color='#1abc9c',
        ),
        TimelineEvent(
            id='malalas',
            title='Chronographia',
            description='World chronicle',
            start_year=565,
            category='history',
            language='grc',
            author='John Malalas',
            importance=7,
            color='#3498db',
        ),
        TimelineEvent(
            id='beowulf',
            title='Beowulf',
            description='Old English epic',
            start_year=750,
            end_year=1000,
            category='epic',
            language='ang',
            author='Anonymous',
            importance=10,
            color='#34495e',
        ),
        TimelineEvent(
            id='alfred_boethius',
            title='OE Boethius',
            description='King Alfred translation',
            start_year=888,
            category='philosophy',
            language='ang',
            author='King Alfred',
            importance=8,
            color='#34495e',
        ),
        TimelineEvent(
            id='theophanes',
            title='Chronographia',
            description='Chronicle 284-813 CE',
            start_year=815,
            category='history',
            language='grc',
            author='Theophanes',
            importance=8,
            color='#9b59b6',
        ),
        TimelineEvent(
            id='psellos',
            title='Chronographia',
            description='History of Byzantine emperors',
            start_year=1070,
            category='history',
            language='grc',
            author='Michael Psellos',
            importance=8,
            color='#9b59b6',
        ),
        TimelineEvent(
            id='anna_komnene',
            title='Alexiad',
            description='History of Alexios I',
            start_year=1148,
            category='history',
            language='grc',
            author='Anna Komnene',
            importance=9,
            color='#9b59b6',
        ),
        TimelineEvent(
            id='digenis_akritas',
            title='Digenis Akritas',
            description='Byzantine vernacular epic',
            start_year=1100,
            end_year=1200,
            category='epic',
            language='grc',
            author='Anonymous',
            importance=8,
            color='#9b59b6',
        ),
        TimelineEvent(
            id='planudes_ovid',
            title='Planudes Metamorphoses',
            description='Greek translation of Ovid',
            start_year=1290,
            category='translation',
            language='grc',
            author='Maximus Planudes',
            importance=8,
            color='#8e44ad',
        ),
        TimelineEvent(
            id='planudes_boethius',
            title='Planudes Boethius',
            description='Greek translation of Boethius',
            start_year=1295,
            category='translation',
            language='grc',
            author='Maximus Planudes',
            importance=8,
            color='#8e44ad',
        ),
        TimelineEvent(
            id='chronicle_morea',
            title='Chronicle of Morea',
            description='Chronicle of Frankish Greece',
            start_year=1340,
            category='history',
            language='grc',
            author='Anonymous',
            importance=7,
            color='#8e44ad',
        ),
        TimelineEvent(
            id='wycliffe_bible',
            title='Wycliffe Bible',
            description='First complete English Bible',
            start_year=1382,
            category='religious',
            language='enm',
            author='John Wycliffe',
            importance=9,
            color='#7f8c8d',
        ),
        TimelineEvent(
            id='chaucer',
            title='Canterbury Tales',
            description='Collection of stories by pilgrims',
            start_year=1387,
            category='literature',
            language='enm',
            author='Geoffrey Chaucer',
            importance=10,
            color='#7f8c8d',
        ),
        TimelineEvent(
            id='tyndale_bible',
            title='Tyndale Bible',
            description='First printed English NT',
            start_year=1526,
            category='religious',
            language='en',
            author='William Tyndale',
            importance=9,
            color='#95a5a6',
        ),
        TimelineEvent(
            id='shakespeare',
            title='Shakespeare Works',
            description='Plays and sonnets',
            start_year=1590,
            end_year=1613,
            category='drama',
            language='en',
            author='William Shakespeare',
            importance=10,
            color='#95a5a6',
        ),
        TimelineEvent(
            id='kjv',
            title='King James Bible',
            description='Authorized English Bible',
            start_year=1611,
            category='religious',
            language='en',
            author='King James translators',
            importance=10,
            color='#95a5a6',
        ),
        TimelineEvent(
            id='chapman_homer',
            title='Chapman Homer',
            description='First complete English Homer',
            start_year=1616,
            category='translation',
            language='en',
            author='George Chapman',
            importance=8,
            color='#95a5a6',
        ),
        TimelineEvent(
            id='milton',
            title='Paradise Lost',
            description='Epic on the Fall of Man',
            start_year=1667,
            category='epic',
            language='en',
            author='John Milton',
            importance=9,
            color='#95a5a6',
        ),
        TimelineEvent(
            id='pope_homer',
            title='Pope Iliad',
            description='Heroic couplet translation',
            start_year=1720,
            category='translation',
            language='en',
            author='Alexander Pope',
            importance=8,
            color='#bdc3c7',
        ),
    ]
    
    def __init__(self, db_path: str = "data/corpus_platform.db"):
        self.db_path = Path(db_path)
    
    def get_all_periods(self) -> List[TimelinePeriod]:
        return self.HISTORICAL_PERIODS
    
    def get_all_events(self) -> List[TimelineEvent]:
        return self.MAJOR_TEXTS
    
    def get_events_by_language(self, language: str) -> List[TimelineEvent]:
        return [e for e in self.MAJOR_TEXTS if e.language == language]
    
    def get_events_by_category(self, category: str) -> List[TimelineEvent]:
        return [e for e in self.MAJOR_TEXTS if e.category == category]
    
    def get_events_in_period(self, start_year: int, end_year: int) -> List[TimelineEvent]:
        return [e for e in self.MAJOR_TEXTS if start_year <= e.start_year <= end_year]
    
    def get_greek_timeline(self) -> Dict[str, Any]:
        greek_periods = [p for p in self.HISTORICAL_PERIODS if 'grc' in p.languages]
        greek_events = [e for e in self.MAJOR_TEXTS if e.language == 'grc']
        
        return {
            'title': 'Greek Language Timeline',
            'periods': [p.to_dict() for p in greek_periods],
            'events': [e.to_dict() for e in greek_events],
        }
    
    def get_english_timeline(self) -> Dict[str, Any]:
        english_langs = ['ang', 'enm', 'en']
        english_periods = [p for p in self.HISTORICAL_PERIODS if any(l in p.languages for l in english_langs)]
        english_events = [e for e in self.MAJOR_TEXTS if e.language in english_langs]
        
        return {
            'title': 'English Language Timeline',
            'periods': [p.to_dict() for p in english_periods],
            'events': [e.to_dict() for e in english_events],
        }
    
    def get_translation_timeline(self) -> Dict[str, Any]:
        translation_events = [e for e in self.MAJOR_TEXTS if e.category == 'translation']
        
        return {
            'title': 'Translation History Timeline',
            'periods': [p.to_dict() for p in self.HISTORICAL_PERIODS],
            'events': [e.to_dict() for e in translation_events],
        }
    
    def get_byzantine_timeline(self) -> Dict[str, Any]:
        byzantine_periods = [p for p in self.HISTORICAL_PERIODS if 'byzantine' in p.id]
        byzantine_events = [e for e in self.MAJOR_TEXTS if 600 <= e.start_year <= 1453 and e.language == 'grc']
        
        return {
            'title': 'Byzantine Greek Timeline',
            'periods': [p.to_dict() for p in byzantine_periods],
            'events': [e.to_dict() for e in byzantine_events],
        }
    
    def get_full_timeline(self) -> Dict[str, Any]:
        return {
            'title': 'Historical Linguistics Timeline',
            'periods': [p.to_dict() for p in self.HISTORICAL_PERIODS],
            'events': [e.to_dict() for e in self.MAJOR_TEXTS],
            'categories': list(set(e.category for e in self.MAJOR_TEXTS)),
            'languages': list(set(e.language for e in self.MAJOR_TEXTS)),
        }
    
    def get_corpus_timeline(self) -> Dict[str, Any]:
        events = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT id, title, author, language, period, diachronic_stage, genre
                    FROM corpus_items
                    WHERE diachronic_stage IS NOT NULL AND diachronic_stage != ''
                    ORDER BY id
                """)
                
                for row in cursor:
                    year = self._extract_year_from_stage(row['diachronic_stage'])
                    if year:
                        events.append(TimelineEvent(
                            id=f"corpus_{row['id']}",
                            title=row['title'] or 'Unknown',
                            description=row['diachronic_stage'] or '',
                            start_year=year,
                            category=row['genre'] or 'text',
                            language=row['language'] or '',
                            author=row['author'] or '',
                            importance=5,
                        ))
        except Exception as e:
            logger.error(f"Error getting corpus timeline: {e}")
        
        return {
            'title': 'Corpus Texts Timeline',
            'periods': [p.to_dict() for p in self.HISTORICAL_PERIODS],
            'events': [e.to_dict() for e in events],
        }
    
    def _extract_year_from_stage(self, stage: str) -> Optional[int]:
        import re
        
        bce_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*century\s*BCE', stage, re.IGNORECASE)
        if bce_match:
            century = int(bce_match.group(1))
            return -(century * 100 - 50)
        
        ce_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*century(?:\s*CE)?', stage, re.IGNORECASE)
        if ce_match:
            century = int(ce_match.group(1))
            return century * 100 - 50
        
        year_match = re.search(r'\b(\d{3,4})\b', stage)
        if year_match:
            return int(year_match.group(1))
        
        return None
    
    def to_json(self) -> str:
        return json.dumps(self.get_full_timeline(), indent=2)
