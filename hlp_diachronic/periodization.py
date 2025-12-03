"""
HLP Diachronic Periodization - Historical Period Definitions

This module provides comprehensive support for defining and managing
historical periods for diachronic linguistic analysis.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
from enum import Enum

from hlp_core.models import Period, Language

logger = logging.getLogger(__name__)


class PeriodGranularity(Enum):
    """Granularity levels for periodization"""
    CENTURY = "century"
    HALF_CENTURY = "half_century"
    QUARTER_CENTURY = "quarter_century"
    DECADE = "decade"
    CUSTOM = "custom"


@dataclass
class PeriodDefinition:
    """Definition of a historical period"""
    name: str
    
    period_enum: Period
    
    start_year: int
    end_year: int
    
    language: Language
    
    description: Optional[str] = None
    
    subperiods: List[PeriodDefinition] = field(default_factory=list)
    
    alternative_names: List[str] = field(default_factory=list)
    
    key_events: List[str] = field(default_factory=list)
    
    linguistic_features: List[str] = field(default_factory=list)
    
    representative_texts: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def contains_year(self, year: int) -> bool:
        """Check if year falls within this period"""
        return self.start_year <= year <= self.end_year
    
    def overlaps(self, other: PeriodDefinition) -> bool:
        """Check if this period overlaps with another"""
        return not (self.end_year < other.start_year or self.start_year > other.end_year)
    
    def duration(self) -> int:
        """Get duration in years"""
        return self.end_year - self.start_year
    
    def midpoint(self) -> int:
        """Get midpoint year"""
        return (self.start_year + self.end_year) // 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "period_enum": self.period_enum.value,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "language": self.language.value,
            "description": self.description,
            "subperiods": [sp.to_dict() for sp in self.subperiods],
            "alternative_names": self.alternative_names,
            "key_events": self.key_events,
            "linguistic_features": self.linguistic_features,
            "representative_texts": self.representative_texts
        }


class PeriodSystem:
    """Base class for period systems"""
    
    def __init__(self, language: Language):
        self.language = language
        self.periods: List[PeriodDefinition] = []
        self._period_map: Dict[Period, PeriodDefinition] = {}
        self._year_cache: Dict[int, PeriodDefinition] = {}
    
    def add_period(self, period: PeriodDefinition):
        """Add a period to the system"""
        self.periods.append(period)
        self._period_map[period.period_enum] = period
        self._year_cache.clear()
    
    def get_period(self, period_enum: Period) -> Optional[PeriodDefinition]:
        """Get period definition by enum"""
        return self._period_map.get(period_enum)
    
    def get_period_for_year(self, year: int) -> Optional[PeriodDefinition]:
        """Get period for a given year"""
        if year in self._year_cache:
            return self._year_cache[year]
        
        for period in self.periods:
            if period.contains_year(year):
                self._year_cache[year] = period
                return period
        
        return None
    
    def get_period_for_date_range(
        self,
        start_year: int,
        end_year: int
    ) -> List[PeriodDefinition]:
        """Get all periods that overlap with date range"""
        result = []
        for period in self.periods:
            if not (period.end_year < start_year or period.start_year > end_year):
                result.append(period)
        return result
    
    def get_ordered_periods(self) -> List[PeriodDefinition]:
        """Get periods ordered by start year"""
        return sorted(self.periods, key=lambda p: p.start_year)
    
    def get_period_sequence(
        self,
        start_period: Period,
        end_period: Period
    ) -> List[PeriodDefinition]:
        """Get sequence of periods between two periods"""
        ordered = self.get_ordered_periods()
        
        start_idx = None
        end_idx = None
        
        for i, period in enumerate(ordered):
            if period.period_enum == start_period:
                start_idx = i
            if period.period_enum == end_period:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            return []
        
        return ordered[start_idx:end_idx + 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "language": self.language.value,
            "periods": [p.to_dict() for p in self.periods]
        }


class GreekPeriodSystem(PeriodSystem):
    """Period system for Ancient Greek"""
    
    def __init__(self):
        super().__init__(Language.ANCIENT_GREEK)
        self._initialize_periods()
    
    def _initialize_periods(self):
        """Initialize Greek periods"""
        
        self.add_period(PeriodDefinition(
            name="Mycenaean Greek",
            period_enum=Period.MYCENAEAN,
            start_year=-1600,
            end_year=-1100,
            language=Language.ANCIENT_GREEK,
            description="The earliest attested form of Greek, written in Linear B",
            alternative_names=["Linear B Greek"],
            key_events=["Mycenaean civilization", "Linear B tablets"],
            linguistic_features=[
                "Syllabic writing system",
                "Archaic morphology",
                "Limited vocabulary attestation"
            ],
            representative_texts=["Pylos tablets", "Knossos tablets"]
        ))
        
        self.add_period(PeriodDefinition(
            name="Archaic Greek",
            period_enum=Period.ARCHAIC,
            start_year=-800,
            end_year=-480,
            language=Language.ANCIENT_GREEK,
            description="The period of Homer and early Greek literature",
            alternative_names=["Homeric Greek", "Early Greek"],
            key_events=["Homeric epics", "Greek alphabet adoption"],
            linguistic_features=[
                "Epic dialect mixture",
                "Digamma preservation",
                "Dual number productive"
            ],
            representative_texts=["Iliad", "Odyssey", "Hesiod's Works and Days"]
        ))
        
        self.add_period(PeriodDefinition(
            name="Classical Greek",
            period_enum=Period.CLASSICAL,
            start_year=-480,
            end_year=-323,
            language=Language.ANCIENT_GREEK,
            description="The golden age of Greek literature and philosophy",
            alternative_names=["Attic Greek", "5th-4th century Greek"],
            key_events=["Persian Wars", "Peloponnesian War", "Death of Alexander"],
            linguistic_features=[
                "Attic dialect dominance",
                "Complex verbal system",
                "Optative mood productive"
            ],
            representative_texts=[
                "Plato's dialogues",
                "Thucydides' History",
                "Sophocles' tragedies"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Hellenistic Greek",
            period_enum=Period.HELLENISTIC,
            start_year=-323,
            end_year=-31,
            language=Language.ANCIENT_GREEK,
            description="The period of the Koine Greek spread",
            alternative_names=["Koine Greek", "Alexandrian Greek"],
            key_events=["Alexander's conquests", "Ptolemaic Egypt", "Roman conquest"],
            linguistic_features=[
                "Dialect leveling",
                "Optative decline",
                "Periphrastic constructions increase"
            ],
            representative_texts=[
                "Septuagint",
                "Polybius' Histories",
                "Papyri from Egypt"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Roman Period Greek",
            period_enum=Period.ROMAN,
            start_year=-31,
            end_year=330,
            language=Language.ANCIENT_GREEK,
            description="Greek under Roman rule",
            alternative_names=["Imperial Greek", "Post-Classical Greek"],
            key_events=["Battle of Actium", "Roman Empire", "Constantine"],
            linguistic_features=[
                "Atticist movement",
                "Continued Koine development",
                "Latin influence"
            ],
            representative_texts=[
                "New Testament",
                "Plutarch's Lives",
                "Epictetus' Discourses"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Late Antique Greek",
            period_enum=Period.LATE_ANTIQUE,
            start_year=330,
            end_year=600,
            language=Language.ANCIENT_GREEK,
            description="Greek in the late Roman and early Byzantine period",
            alternative_names=["Early Byzantine Greek", "Patristic Greek"],
            key_events=["Constantinople founded", "Fall of Rome", "Justinian"],
            linguistic_features=[
                "Phonological changes",
                "Case system simplification",
                "Christian vocabulary"
            ],
            representative_texts=[
                "Church Fathers",
                "Procopius' Wars",
                "Romanos the Melodist"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Byzantine Greek",
            period_enum=Period.BYZANTINE,
            start_year=600,
            end_year=1453,
            language=Language.ANCIENT_GREEK,
            description="Medieval Greek of the Byzantine Empire",
            alternative_names=["Medieval Greek", "Middle Greek"],
            key_events=["Arab conquests", "Iconoclasm", "Fall of Constantinople"],
            linguistic_features=[
                "Diglossia (learned vs. vernacular)",
                "Loss of infinitive",
                "Article + infinitive constructions"
            ],
            representative_texts=[
                "Chronicle of Theophanes",
                "Digenis Akritas",
                "Anna Komnene's Alexiad"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Modern Greek",
            period_enum=Period.MODERN,
            start_year=1453,
            end_year=2100,
            language=Language.ANCIENT_GREEK,
            description="Greek from the fall of Constantinople to present",
            alternative_names=["Neo-Hellenic", "Contemporary Greek"],
            key_events=["Ottoman period", "Greek independence", "Modern state"],
            linguistic_features=[
                "Loss of dative",
                "Simplified verb system",
                "Fixed stress patterns"
            ],
            representative_texts=[
                "Erotokritos",
                "Solomos' poetry",
                "Modern literature"
            ]
        ))


class LatinPeriodSystem(PeriodSystem):
    """Period system for Latin"""
    
    def __init__(self):
        super().__init__(Language.LATIN)
        self._initialize_periods()
    
    def _initialize_periods(self):
        """Initialize Latin periods"""
        
        self.add_period(PeriodDefinition(
            name="Old Latin",
            period_enum=Period.OLD_LATIN,
            start_year=-600,
            end_year=-100,
            language=Language.LATIN,
            description="The earliest attested form of Latin",
            alternative_names=["Archaic Latin", "Early Latin"],
            key_events=["Roman Republic founding", "Punic Wars"],
            linguistic_features=[
                "Archaic morphology",
                "Vowel weakening in progress",
                "Case system intact"
            ],
            representative_texts=[
                "Twelve Tables",
                "Plautus' comedies",
                "Ennius' Annales"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Classical Latin",
            period_enum=Period.CLASSICAL_LATIN,
            start_year=-100,
            end_year=14,
            language=Language.LATIN,
            description="The golden age of Latin literature",
            alternative_names=["Golden Age Latin", "Ciceronian Latin"],
            key_events=["Late Republic", "Civil Wars", "Augustus"],
            linguistic_features=[
                "Standardized grammar",
                "Complex periodic sentences",
                "Subjunctive productive"
            ],
            representative_texts=[
                "Cicero's orations",
                "Caesar's Commentaries",
                "Virgil's Aeneid"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Silver Age Latin",
            period_enum=Period.SILVER_LATIN,
            start_year=14,
            end_year=200,
            language=Language.LATIN,
            description="Latin of the early Empire",
            alternative_names=["Imperial Latin", "Post-Augustan Latin"],
            key_events=["Julio-Claudian dynasty", "Flavians", "Five Good Emperors"],
            linguistic_features=[
                "Rhetorical elaboration",
                "Greek influence",
                "Poetic innovations"
            ],
            representative_texts=[
                "Seneca's works",
                "Tacitus' Annals",
                "Pliny's Letters"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Late Latin",
            period_enum=Period.LATE_LATIN,
            start_year=200,
            end_year=600,
            language=Language.LATIN,
            description="Latin of late antiquity",
            alternative_names=["Patristic Latin", "Vulgar Latin period"],
            key_events=["Crisis of Third Century", "Christianization", "Fall of Rome"],
            linguistic_features=[
                "Case system weakening",
                "Preposition increase",
                "Christian vocabulary"
            ],
            representative_texts=[
                "Vulgate Bible",
                "Augustine's Confessions",
                "Boethius' Consolation"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Medieval Latin",
            period_enum=Period.MEDIEVAL_LATIN,
            start_year=600,
            end_year=1500,
            language=Language.LATIN,
            description="Latin of the Middle Ages",
            alternative_names=["Church Latin", "Scholastic Latin"],
            key_events=["Carolingian Renaissance", "Universities", "Scholasticism"],
            linguistic_features=[
                "Regional variation",
                "Simplified syntax",
                "Technical vocabulary"
            ],
            representative_texts=[
                "Bede's History",
                "Aquinas' Summa",
                "Carmina Burana"
            ]
        ))
        
        self.add_period(PeriodDefinition(
            name="Renaissance Latin",
            period_enum=Period.RENAISSANCE_LATIN,
            start_year=1400,
            end_year=1600,
            language=Language.LATIN,
            description="Humanist revival of Classical Latin",
            alternative_names=["Humanist Latin", "Neo-Latin"],
            key_events=["Italian Renaissance", "Printing press", "Reformation"],
            linguistic_features=[
                "Ciceronian imitation",
                "Classical purism",
                "Scientific terminology"
            ],
            representative_texts=[
                "Petrarch's letters",
                "Erasmus' works",
                "More's Utopia"
            ]
        ))


class CustomPeriodSystem(PeriodSystem):
    """Custom period system for user-defined periods"""
    
    def __init__(self, language: Language, name: str = "Custom"):
        super().__init__(language)
        self.name = name
    
    def add_custom_period(
        self,
        name: str,
        start_year: int,
        end_year: int,
        period_enum: Optional[Period] = None,
        description: Optional[str] = None
    ) -> PeriodDefinition:
        """Add a custom period"""
        if period_enum is None:
            period_enum = Period.CUSTOM
        
        period = PeriodDefinition(
            name=name,
            period_enum=period_enum,
            start_year=start_year,
            end_year=end_year,
            language=self.language,
            description=description
        )
        
        self.add_period(period)
        return period
    
    def create_century_periods(
        self,
        start_century: int,
        end_century: int
    ) -> List[PeriodDefinition]:
        """Create century-based periods"""
        periods = []
        
        for century in range(start_century, end_century + 1):
            if century < 0:
                name = f"{abs(century)}th century BCE"
                start_year = (century - 1) * 100
                end_year = century * 100 - 1
            else:
                name = f"{century}th century CE"
                start_year = (century - 1) * 100 + 1
                end_year = century * 100
            
            period = self.add_custom_period(
                name=name,
                start_year=start_year,
                end_year=end_year,
                description=f"The {name}"
            )
            periods.append(period)
        
        return periods


TEXT_DATE_PATTERNS = {
    r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:century|cent\.?)\s*(?:BCE|BC|B\.C\.E?\.?)": lambda m: -int(m.group(1)) * 100 + 50,
    r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:century|cent\.?)\s*(?:CE|AD|A\.D\.?)": lambda m: int(m.group(1)) * 100 - 50,
    r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:century|cent\.?)": lambda m: int(m.group(1)) * 100 - 50,
    r"(\d{3,4})\s*(?:BCE|BC|B\.C\.E?\.?)": lambda m: -int(m.group(1)),
    r"(\d{3,4})\s*(?:CE|AD|A\.D\.?)": lambda m: int(m.group(1)),
    r"c\.?\s*(\d{3,4})\s*(?:BCE|BC)?": lambda m: -int(m.group(1)) if "BC" in m.group(0).upper() else int(m.group(1)),
}

AUTHOR_DATES = {
    "homer": -750,
    "hesiod": -700,
    "herodotus": -450,
    "thucydides": -420,
    "plato": -380,
    "aristotle": -340,
    "demosthenes": -340,
    "polybius": -150,
    "plutarch": 100,
    "lucian": 170,
    "plautus": -200,
    "terence": -160,
    "cicero": -50,
    "caesar": -50,
    "virgil": -20,
    "horace": -20,
    "ovid": 5,
    "livy": 10,
    "seneca": 50,
    "tacitus": 100,
    "pliny": 100,
    "augustine": 400,
    "jerome": 400,
}


def get_period_for_date(
    year: int,
    language: Language = Language.ANCIENT_GREEK
) -> Optional[PeriodDefinition]:
    """Get period for a given year"""
    if language == Language.ANCIENT_GREEK:
        system = GreekPeriodSystem()
    elif language == Language.LATIN:
        system = LatinPeriodSystem()
    else:
        return None
    
    return system.get_period_for_year(year)


def get_period_for_text(
    text_metadata: Dict[str, Any],
    language: Language = Language.ANCIENT_GREEK
) -> Optional[PeriodDefinition]:
    """Get period for a text based on metadata"""
    year = None
    
    if "date" in text_metadata:
        date_str = str(text_metadata["date"])
        year = _parse_date_string(date_str)
    
    if year is None and "year" in text_metadata:
        year = int(text_metadata["year"])
    
    if year is None and "author" in text_metadata:
        author = text_metadata["author"].lower()
        for known_author, known_year in AUTHOR_DATES.items():
            if known_author in author:
                year = known_year
                break
    
    if year is None and "century" in text_metadata:
        century = int(text_metadata["century"])
        year = century * 100 - 50
    
    if year is not None:
        return get_period_for_date(year, language)
    
    return None


def _parse_date_string(date_str: str) -> Optional[int]:
    """Parse a date string to extract year"""
    for pattern, extractor in TEXT_DATE_PATTERNS.items():
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            return extractor(match)
    
    try:
        year = int(date_str)
        return year
    except ValueError:
        pass
    
    return None


def create_period_system(
    language: Language,
    custom: bool = False
) -> PeriodSystem:
    """Create a period system for a language"""
    if custom:
        return CustomPeriodSystem(language)
    
    if language == Language.ANCIENT_GREEK:
        return GreekPeriodSystem()
    elif language == Language.LATIN:
        return LatinPeriodSystem()
    else:
        return CustomPeriodSystem(language)
