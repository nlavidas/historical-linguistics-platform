#!/usr/bin/env python3
"""
Diachronic Analysis Module for Greek Linguistics Platform
"Windsurf for Greek Diachronic and Contrastive Linguistics"

Comprehensive tools for:
1. Language change analysis across Greek periods
2. Contrastive analysis between periods
3. Transitivity and argument structure evolution
4. Voice system development
5. Tense-aspect changes
6. Lexical change tracking
7. Phonological change patterns
8. Syntactic change detection

Based on the research of Nikolaos Lavidas and diachronic linguistics methodology.
"""

import os
import json
import sqlite3
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GREEK PERIODS DEFINITION
# ============================================================================

class GreekPeriod(Enum):
    """Greek language periods"""
    MYCENAEAN = ("mycenaean", -1600, -1100)
    ARCHAIC = ("archaic", -800, -500)
    CLASSICAL = ("classical", -500, -323)
    HELLENISTIC = ("hellenistic", -323, -31)
    ROMAN = ("roman", -31, 300)
    LATE_ANTIQUE = ("late_antique", 300, 600)
    BYZANTINE = ("byzantine", 600, 1453)
    EARLY_MODERN = ("early_modern", 1453, 1830)
    MODERN = ("modern", 1830, 2025)
    
    def __init__(self, code: str, start: int, end: int):
        self.code = code
        self.start_year = start
        self.end_year = end
    
    @classmethod
    def from_year(cls, year: int) -> Optional['GreekPeriod']:
        """Get period from year"""
        for period in cls:
            if period.start_year <= year <= period.end_year:
                return period
        return None
    
    @classmethod
    def from_code(cls, code: str) -> Optional['GreekPeriod']:
        """Get period from code"""
        for period in cls:
            if period.code == code:
                return period
        return None


# ============================================================================
# LINGUISTIC FEATURES
# ============================================================================

@dataclass
class TransitivityFeatures:
    """Transitivity features for a verb"""
    verb: str
    lemma: str
    period: str
    transitivity_class: str  # transitive, intransitive, ambitransitive
    argument_structure: List[str]  # [ARG0, ARG1, ARG2, ...]
    alternations: List[str]  # causative, anticausative, passive, etc.
    object_marking: str  # accusative, genitive, dative, null
    examples: List[str] = field(default_factory=list)


@dataclass
class VoiceFeatures:
    """Voice system features"""
    verb: str
    lemma: str
    period: str
    voice_form: str  # active, middle, passive
    voice_function: str  # active, middle, passive, deponent
    morphology: str
    semantic_type: str  # reflexive, reciprocal, anticausative, passive
    examples: List[str] = field(default_factory=list)


@dataclass
class TenseAspectFeatures:
    """Tense-aspect features"""
    verb: str
    lemma: str
    period: str
    tense: str  # present, imperfect, aorist, perfect, pluperfect, future
    aspect: str  # imperfective, perfective, perfect
    mood: str  # indicative, subjunctive, optative, imperative
    periphrastic: bool = False
    auxiliary: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class LexicalChange:
    """Lexical change record"""
    lemma: str
    change_type: str  # semantic_shift, borrowing, loss, innovation
    source_period: str
    target_period: str
    source_meaning: str
    target_meaning: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class SyntacticChange:
    """Syntactic change record"""
    construction: str
    change_type: str  # word_order, case_loss, new_construction
    source_period: str
    target_period: str
    source_pattern: str
    target_pattern: str
    frequency_change: float = 0.0
    examples: List[str] = field(default_factory=list)


# ============================================================================
# DIACHRONIC DATABASE
# ============================================================================

class DiachronicDatabase:
    """Database for diachronic analysis"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize diachronic analysis tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Transitivity features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transitivity_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb TEXT,
                lemma TEXT,
                period TEXT,
                transitivity_class TEXT,
                argument_structure TEXT,
                alternations TEXT,
                object_marking TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Voice features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb TEXT,
                lemma TEXT,
                period TEXT,
                voice_form TEXT,
                voice_function TEXT,
                morphology TEXT,
                semantic_type TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tense-aspect features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tense_aspect_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb TEXT,
                lemma TEXT,
                period TEXT,
                tense TEXT,
                aspect TEXT,
                mood TEXT,
                periphrastic INTEGER DEFAULT 0,
                auxiliary TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Lexical changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lexical_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT,
                change_type TEXT,
                source_period TEXT,
                target_period TEXT,
                source_meaning TEXT,
                target_meaning TEXT,
                evidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Syntactic changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS syntactic_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                construction TEXT,
                change_type TEXT,
                source_period TEXT,
                target_period TEXT,
                source_pattern TEXT,
                target_pattern TEXT,
                frequency_change REAL DEFAULT 0,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Period statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS period_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT,
                feature_type TEXT,
                feature_name TEXT,
                count INTEGER,
                frequency REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_transitivity_feature(self, feature: TransitivityFeatures):
        """Add transitivity feature"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO transitivity_features
            (verb, lemma, period, transitivity_class, argument_structure,
             alternations, object_marking, examples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feature.verb, feature.lemma, feature.period,
            feature.transitivity_class,
            json.dumps(feature.argument_structure),
            json.dumps(feature.alternations),
            feature.object_marking,
            json.dumps(feature.examples)
        ))
        
        conn.commit()
        conn.close()
    
    def get_transitivity_by_period(self, period: str) -> List[Dict]:
        """Get transitivity features by period"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM transitivity_features WHERE period = ?
        """, (period,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def add_voice_feature(self, feature: VoiceFeatures):
        """Add voice feature"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO voice_features
            (verb, lemma, period, voice_form, voice_function,
             morphology, semantic_type, examples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feature.verb, feature.lemma, feature.period,
            feature.voice_form, feature.voice_function,
            feature.morphology, feature.semantic_type,
            json.dumps(feature.examples)
        ))
        
        conn.commit()
        conn.close()
    
    def add_lexical_change(self, change: LexicalChange):
        """Add lexical change"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO lexical_changes
            (lemma, change_type, source_period, target_period,
             source_meaning, target_meaning, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            change.lemma, change.change_type,
            change.source_period, change.target_period,
            change.source_meaning, change.target_meaning,
            json.dumps(change.evidence)
        ))
        
        conn.commit()
        conn.close()
    
    def add_syntactic_change(self, change: SyntacticChange):
        """Add syntactic change"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO syntactic_changes
            (construction, change_type, source_period, target_period,
             source_pattern, target_pattern, frequency_change, examples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            change.construction, change.change_type,
            change.source_period, change.target_period,
            change.source_pattern, change.target_pattern,
            change.frequency_change,
            json.dumps(change.examples)
        ))
        
        conn.commit()
        conn.close()


# ============================================================================
# TRANSITIVITY ANALYZER
# ============================================================================

class TransitivityAnalyzer:
    """Analyzer for transitivity patterns"""
    
    # Known transitivity alternations in Greek
    ALTERNATIONS = {
        "causative": "Agent causes Patient to undergo action",
        "anticausative": "Patient undergoes action without expressed Agent",
        "passive": "Patient promoted to subject, Agent demoted",
        "middle": "Subject affected by action",
        "reflexive": "Agent and Patient are same entity",
        "reciprocal": "Multiple agents act on each other"
    }
    
    # Verb classes by transitivity
    VERB_CLASSES = {
        "unergative": ["βαίνω", "τρέχω", "πηδάω", "γελάω"],
        "unaccusative": ["πίπτω", "ἔρχομαι", "γίγνομαι", "φαίνομαι"],
        "transitive": ["λέγω", "ποιέω", "γράφω", "φέρω", "ἄγω"],
        "ditransitive": ["δίδωμι", "λέγω", "δείκνυμι"],
        "ambitransitive": ["ἐσθίω", "πίνω", "γράφω", "ᾄδω"]
    }
    
    def __init__(self, db: DiachronicDatabase):
        self.db = db
    
    def analyze_verb(self, verb: str, lemma: str, period: str,
                     context: str = "") -> TransitivityFeatures:
        """Analyze transitivity of a verb"""
        # Determine transitivity class
        trans_class = self._determine_class(lemma)
        
        # Determine argument structure
        arg_structure = self._determine_arguments(lemma, context)
        
        # Identify alternations
        alternations = self._identify_alternations(lemma, period)
        
        # Determine object marking
        obj_marking = self._determine_object_marking(context)
        
        return TransitivityFeatures(
            verb=verb,
            lemma=lemma,
            period=period,
            transitivity_class=trans_class,
            argument_structure=arg_structure,
            alternations=alternations,
            object_marking=obj_marking,
            examples=[context] if context else []
        )
    
    def _determine_class(self, lemma: str) -> str:
        """Determine transitivity class"""
        for cls, verbs in self.VERB_CLASSES.items():
            if lemma in verbs:
                return cls
        return "unknown"
    
    def _determine_arguments(self, lemma: str, context: str) -> List[str]:
        """Determine argument structure"""
        # Default structures by class
        structures = {
            "unergative": ["ARG0"],
            "unaccusative": ["ARG1"],
            "transitive": ["ARG0", "ARG1"],
            "ditransitive": ["ARG0", "ARG1", "ARG2"],
            "ambitransitive": ["ARG0", "ARG1?"]
        }
        
        cls = self._determine_class(lemma)
        return structures.get(cls, ["ARG0"])
    
    def _identify_alternations(self, lemma: str, period: str) -> List[str]:
        """Identify available alternations"""
        alternations = []
        
        # Most Greek verbs have causative/anticausative alternation
        if self._determine_class(lemma) in ["transitive", "ambitransitive"]:
            alternations.append("causative")
            alternations.append("anticausative")
        
        # All verbs can passivize
        alternations.append("passive")
        
        # Check for middle voice forms
        if lemma.endswith("ομαι") or lemma.endswith("μαι"):
            alternations.append("middle")
        
        return alternations
    
    def _determine_object_marking(self, context: str) -> str:
        """Determine object marking from context"""
        # Look for case markers
        if 'τόν' in context or 'τήν' in context or 'τούς' in context:
            return "accusative"
        elif 'τοῦ' in context or 'τῆς' in context or 'τῶν' in context:
            return "genitive"
        elif 'τῷ' in context or 'τῇ' in context or 'τοῖς' in context:
            return "dative"
        return "unknown"
    
    def compare_periods(self, lemma: str, period1: str, period2: str) -> Dict:
        """Compare transitivity across periods"""
        features1 = self.db.get_transitivity_by_period(period1)
        features2 = self.db.get_transitivity_by_period(period2)
        
        # Filter by lemma
        f1 = [f for f in features1 if f.get('lemma') == lemma]
        f2 = [f for f in features2 if f.get('lemma') == lemma]
        
        return {
            "lemma": lemma,
            "period1": {
                "period": period1,
                "features": f1
            },
            "period2": {
                "period": period2,
                "features": f2
            },
            "changes": self._identify_changes(f1, f2)
        }
    
    def _identify_changes(self, features1: List[Dict], 
                          features2: List[Dict]) -> List[str]:
        """Identify changes between periods"""
        changes = []
        
        if not features1 or not features2:
            return ["Insufficient data for comparison"]
        
        # Compare transitivity classes
        classes1 = set(f.get('transitivity_class') for f in features1)
        classes2 = set(f.get('transitivity_class') for f in features2)
        
        if classes1 != classes2:
            changes.append(f"Transitivity class change: {classes1} -> {classes2}")
        
        # Compare alternations
        alts1 = set()
        alts2 = set()
        for f in features1:
            alts = json.loads(f.get('alternations', '[]'))
            alts1.update(alts)
        for f in features2:
            alts = json.loads(f.get('alternations', '[]'))
            alts2.update(alts)
        
        new_alts = alts2 - alts1
        lost_alts = alts1 - alts2
        
        if new_alts:
            changes.append(f"New alternations: {new_alts}")
        if lost_alts:
            changes.append(f"Lost alternations: {lost_alts}")
        
        return changes


# ============================================================================
# VOICE ANALYZER
# ============================================================================

class VoiceAnalyzer:
    """Analyzer for voice system"""
    
    # Voice morphology patterns
    VOICE_MORPHOLOGY = {
        "active": {
            "present": ["ω", "εις", "ει", "ομεν", "ετε", "ουσι"],
            "aorist": ["α", "ας", "ε", "αμεν", "ατε", "αν"]
        },
        "middle_passive": {
            "present": ["ομαι", "ῃ", "εται", "όμεθα", "εσθε", "ονται"],
            "aorist_middle": ["άμην", "ω", "ατο", "άμεθα", "ασθε", "αντο"],
            "aorist_passive": ["ην", "ης", "η", "ημεν", "ητε", "ησαν"]
        }
    }
    
    # Semantic types of middle voice
    MIDDLE_TYPES = {
        "reflexive": "Subject acts on self",
        "reciprocal": "Subjects act on each other",
        "autocausative": "Subject causes change in self",
        "anticausative": "Subject undergoes change",
        "passive": "Subject is affected by external agent",
        "deponent": "Middle form with active meaning"
    }
    
    def __init__(self, db: DiachronicDatabase):
        self.db = db
    
    def analyze_voice(self, verb: str, lemma: str, period: str,
                      context: str = "") -> VoiceFeatures:
        """Analyze voice of a verb form"""
        # Determine voice form
        voice_form = self._determine_voice_form(verb)
        
        # Determine voice function
        voice_function = self._determine_voice_function(verb, context)
        
        # Get morphology
        morphology = self._get_morphology(verb)
        
        # Determine semantic type
        semantic_type = self._determine_semantic_type(verb, context)
        
        return VoiceFeatures(
            verb=verb,
            lemma=lemma,
            period=period,
            voice_form=voice_form,
            voice_function=voice_function,
            morphology=morphology,
            semantic_type=semantic_type,
            examples=[context] if context else []
        )
    
    def _determine_voice_form(self, verb: str) -> str:
        """Determine morphological voice form"""
        # Check endings
        for ending in self.VOICE_MORPHOLOGY["middle_passive"]["present"]:
            if verb.endswith(ending):
                return "middle_passive"
        
        for ending in self.VOICE_MORPHOLOGY["middle_passive"]["aorist_passive"]:
            if verb.endswith(ending):
                return "passive"
        
        for ending in self.VOICE_MORPHOLOGY["active"]["present"]:
            if verb.endswith(ending):
                return "active"
        
        return "unknown"
    
    def _determine_voice_function(self, verb: str, context: str) -> str:
        """Determine functional voice"""
        voice_form = self._determine_voice_form(verb)
        
        if voice_form == "active":
            return "active"
        
        # Check for passive agent (ὑπό + genitive)
        if "ὑπό" in context or "ὑπ'" in context:
            return "passive"
        
        # Check for reflexive pronouns
        if "ἑαυτ" in context or "αὑτ" in context:
            return "reflexive"
        
        # Check for reciprocal markers
        if "ἀλλήλ" in context:
            return "reciprocal"
        
        # Default to middle
        return "middle"
    
    def _get_morphology(self, verb: str) -> str:
        """Get morphological description"""
        # Simplified morphological analysis
        if verb.endswith(("ομαι", "εται", "ονται")):
            return "present_middle_passive"
        elif verb.endswith(("άμην", "ατο", "αντο")):
            return "aorist_middle"
        elif verb.endswith(("ην", "ης", "ησαν")):
            return "aorist_passive"
        elif verb.endswith(("ω", "εις", "ει")):
            return "present_active"
        elif verb.endswith(("α", "ας", "ε")):
            return "aorist_active"
        return "unknown"
    
    def _determine_semantic_type(self, verb: str, context: str) -> str:
        """Determine semantic type of voice"""
        voice_func = self._determine_voice_function(verb, context)
        
        if voice_func == "passive":
            return "passive"
        elif voice_func == "reflexive":
            return "reflexive"
        elif voice_func == "reciprocal":
            return "reciprocal"
        elif voice_func == "middle":
            # Could be autocausative or anticausative
            return "middle_general"
        
        return "active"
    
    def track_voice_development(self, lemma: str) -> Dict:
        """Track voice system development for a verb"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT period, voice_form, voice_function, semantic_type, COUNT(*) as count
            FROM voice_features
            WHERE lemma = ?
            GROUP BY period, voice_form, voice_function, semantic_type
            ORDER BY period
        """, (lemma,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Organize by period
        by_period = defaultdict(list)
        for r in results:
            by_period[r['period']].append(r)
        
        return {
            "lemma": lemma,
            "development": dict(by_period)
        }


# ============================================================================
# TENSE-ASPECT ANALYZER
# ============================================================================

class TenseAspectAnalyzer:
    """Analyzer for tense-aspect system"""
    
    # Greek tense-aspect system
    TENSE_ASPECT_SYSTEM = {
        "present": {"tense": "present", "aspect": "imperfective"},
        "imperfect": {"tense": "past", "aspect": "imperfective"},
        "aorist": {"tense": "past", "aspect": "perfective"},
        "perfect": {"tense": "present", "aspect": "perfect"},
        "pluperfect": {"tense": "past", "aspect": "perfect"},
        "future": {"tense": "future", "aspect": "neutral"},
        "future_perfect": {"tense": "future", "aspect": "perfect"}
    }
    
    # Periphrastic constructions by period
    PERIPHRASTIC = {
        "classical": [
            ("εἰμί + participle", "perfect/progressive"),
            ("ἔχω + participle", "perfect")
        ],
        "hellenistic": [
            ("εἰμί + participle", "progressive"),
            ("ἔχω + infinitive", "obligation")
        ],
        "byzantine": [
            ("ἔχω + infinitive", "future"),
            ("θέλω + infinitive", "future")
        ]
    }
    
    def __init__(self, db: DiachronicDatabase):
        self.db = db
    
    def analyze_tense_aspect(self, verb: str, lemma: str, period: str,
                              context: str = "") -> TenseAspectFeatures:
        """Analyze tense-aspect of a verb form"""
        # Determine tense
        tense = self._determine_tense(verb)
        
        # Determine aspect
        aspect = self._determine_aspect(verb, tense)
        
        # Determine mood
        mood = self._determine_mood(verb)
        
        # Check for periphrastic
        is_periphrastic, auxiliary = self._check_periphrastic(context, period)
        
        return TenseAspectFeatures(
            verb=verb,
            lemma=lemma,
            period=period,
            tense=tense,
            aspect=aspect,
            mood=mood,
            periphrastic=is_periphrastic,
            auxiliary=auxiliary,
            examples=[context] if context else []
        )
    
    def _determine_tense(self, verb: str) -> str:
        """Determine tense from morphology"""
        # Augment indicates past
        if verb.startswith(('ἐ', 'ἠ', 'ὠ', 'ᾐ', 'ηὐ')):
            # Could be imperfect, aorist, or pluperfect
            if 'κ' in verb or 'χ' in verb:  # Perfect stem markers
                return "pluperfect"
            return "past"  # Need more context for imperfect vs aorist
        
        # Reduplication indicates perfect
        if len(verb) > 2 and verb[0] == verb[2]:
            return "perfect"
        
        # Future markers
        if 'σ' in verb and not verb.startswith('ἐ'):
            return "future"
        
        return "present"
    
    def _determine_aspect(self, verb: str, tense: str) -> str:
        """Determine aspect"""
        if tense in ["perfect", "pluperfect"]:
            return "perfect"
        
        # Aorist markers
        if 'σα' in verb or verb.endswith(('α', 'ας', 'ε', 'αμεν')):
            return "perfective"
        
        return "imperfective"
    
    def _determine_mood(self, verb: str) -> str:
        """Determine mood"""
        # Subjunctive: long vowel in ending
        if verb.endswith(('ω', 'ῃς', 'ῃ', 'ωμεν', 'ητε', 'ωσι')):
            # Could be indicative or subjunctive - need context
            return "indicative_or_subjunctive"
        
        # Optative: οι, αι markers
        if 'οι' in verb or 'αι' in verb:
            return "optative"
        
        # Imperative: specific endings
        if verb.endswith(('ε', 'έτω', 'ετε', 'όντων', 'ου', 'έσθω')):
            return "imperative"
        
        # Infinitive
        if verb.endswith(('ειν', 'αι', 'ναι', 'σθαι')):
            return "infinitive"
        
        # Participle
        if verb.endswith(('ων', 'ουσα', 'ον', 'ας', 'ασα', 'αν', 'ως', 'υια', 'ος')):
            return "participle"
        
        return "indicative"
    
    def _check_periphrastic(self, context: str, period: str) -> Tuple[bool, str]:
        """Check for periphrastic construction"""
        auxiliaries = ['εἰμί', 'ἔχω', 'θέλω', 'μέλλω']
        
        for aux in auxiliaries:
            if aux in context:
                return True, aux
        
        return False, ""
    
    def track_perfect_development(self) -> Dict:
        """Track development of the perfect tense"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT period, tense, aspect, periphrastic, COUNT(*) as count
            FROM tense_aspect_features
            WHERE tense IN ('perfect', 'pluperfect') OR aspect = 'perfect'
            GROUP BY period, tense, aspect, periphrastic
            ORDER BY period
        """)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {
            "feature": "perfect_tense",
            "development": results,
            "notes": [
                "Classical: synthetic perfect with resultative meaning",
                "Hellenistic: perfect merging with aorist",
                "Byzantine: periphrastic perfect with ἔχω"
            ]
        }


# ============================================================================
# CONTRASTIVE ANALYZER
# ============================================================================

class ContrastiveAnalyzer:
    """Contrastive analysis between periods"""
    
    def __init__(self, db: DiachronicDatabase):
        self.db = db
    
    def compare_periods(self, period1: str, period2: str, 
                        feature_type: str = "all") -> Dict:
        """Compare linguistic features between periods"""
        results = {
            "period1": period1,
            "period2": period2,
            "comparisons": {}
        }
        
        if feature_type in ["all", "transitivity"]:
            results["comparisons"]["transitivity"] = self._compare_transitivity(
                period1, period2
            )
        
        if feature_type in ["all", "voice"]:
            results["comparisons"]["voice"] = self._compare_voice(
                period1, period2
            )
        
        if feature_type in ["all", "tense_aspect"]:
            results["comparisons"]["tense_aspect"] = self._compare_tense_aspect(
                period1, period2
            )
        
        return results
    
    def _compare_transitivity(self, period1: str, period2: str) -> Dict:
        """Compare transitivity patterns"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Get transitivity class distribution
        cursor.execute("""
            SELECT transitivity_class, COUNT(*) as count
            FROM transitivity_features
            WHERE period = ?
            GROUP BY transitivity_class
        """, (period1,))
        dist1 = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT transitivity_class, COUNT(*) as count
            FROM transitivity_features
            WHERE period = ?
            GROUP BY transitivity_class
        """, (period2,))
        dist2 = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "period1_distribution": dist1,
            "period2_distribution": dist2,
            "changes": self._calculate_distribution_change(dist1, dist2)
        }
    
    def _compare_voice(self, period1: str, period2: str) -> Dict:
        """Compare voice system"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT voice_function, COUNT(*) as count
            FROM voice_features
            WHERE period = ?
            GROUP BY voice_function
        """, (period1,))
        dist1 = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT voice_function, COUNT(*) as count
            FROM voice_features
            WHERE period = ?
            GROUP BY voice_function
        """, (period2,))
        dist2 = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "period1_distribution": dist1,
            "period2_distribution": dist2,
            "changes": self._calculate_distribution_change(dist1, dist2)
        }
    
    def _compare_tense_aspect(self, period1: str, period2: str) -> Dict:
        """Compare tense-aspect system"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tense, aspect, COUNT(*) as count
            FROM tense_aspect_features
            WHERE period = ?
            GROUP BY tense, aspect
        """, (period1,))
        dist1 = {f"{row[0]}_{row[1]}": row[2] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT tense, aspect, COUNT(*) as count
            FROM tense_aspect_features
            WHERE period = ?
            GROUP BY tense, aspect
        """, (period2,))
        dist2 = {f"{row[0]}_{row[1]}": row[2] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "period1_distribution": dist1,
            "period2_distribution": dist2,
            "changes": self._calculate_distribution_change(dist1, dist2)
        }
    
    def _calculate_distribution_change(self, dist1: Dict, dist2: Dict) -> List[Dict]:
        """Calculate changes between distributions"""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        total1 = sum(dist1.values()) or 1
        total2 = sum(dist2.values()) or 1
        
        changes = []
        for key in all_keys:
            freq1 = dist1.get(key, 0) / total1
            freq2 = dist2.get(key, 0) / total2
            
            if freq1 > 0 or freq2 > 0:
                change = {
                    "feature": key,
                    "period1_freq": freq1,
                    "period2_freq": freq2,
                    "change": freq2 - freq1,
                    "change_percent": ((freq2 - freq1) / freq1 * 100) if freq1 > 0 else float('inf')
                }
                changes.append(change)
        
        # Sort by absolute change
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes


# ============================================================================
# DIACHRONIC ANALYSIS ENGINE
# ============================================================================

class DiachronicAnalysisEngine:
    """Main engine for diachronic analysis"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db = DiachronicDatabase(db_path)
        self.transitivity = TransitivityAnalyzer(self.db)
        self.voice = VoiceAnalyzer(self.db)
        self.tense_aspect = TenseAspectAnalyzer(self.db)
        self.contrastive = ContrastiveAnalyzer(self.db)
    
    def analyze_text(self, text: str, period: str) -> Dict:
        """Comprehensive diachronic analysis of text"""
        results = {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "period": period,
            "analysis": {}
        }
        
        # Tokenize (simplified)
        tokens = text.split()
        
        # Analyze each potential verb
        verbs_analyzed = []
        for token in tokens:
            # Simple verb detection
            if any(token.endswith(e) for e in ['ω', 'ει', 'ομεν', 'ουσι', 'εται', 'ονται']):
                trans = self.transitivity.analyze_verb(token, token, period, text)
                voice = self.voice.analyze_voice(token, token, period, text)
                ta = self.tense_aspect.analyze_tense_aspect(token, token, period, text)
                
                verbs_analyzed.append({
                    "verb": token,
                    "transitivity": {
                        "class": trans.transitivity_class,
                        "arguments": trans.argument_structure
                    },
                    "voice": {
                        "form": voice.voice_form,
                        "function": voice.voice_function
                    },
                    "tense_aspect": {
                        "tense": ta.tense,
                        "aspect": ta.aspect,
                        "mood": ta.mood
                    }
                })
        
        results["analysis"]["verbs"] = verbs_analyzed
        results["analysis"]["verb_count"] = len(verbs_analyzed)
        
        return results
    
    def generate_diachronic_report(self, lemma: str) -> Dict:
        """Generate comprehensive diachronic report for a lemma"""
        report = {
            "lemma": lemma,
            "generated_at": datetime.now().isoformat(),
            "sections": {}
        }
        
        # Transitivity development
        report["sections"]["transitivity"] = {
            "title": "Transitivity Development",
            "data": self.db.get_transitivity_by_period("classical")  # Example
        }
        
        # Voice development
        report["sections"]["voice"] = {
            "title": "Voice System Development",
            "data": self.voice.track_voice_development(lemma)
        }
        
        # Tense-aspect development
        report["sections"]["tense_aspect"] = {
            "title": "Tense-Aspect Development",
            "data": self.tense_aspect.track_perfect_development()
        }
        
        return report
    
    def compare_all_periods(self, feature_type: str = "transitivity") -> Dict:
        """Compare feature across all periods"""
        periods = [p.code for p in GreekPeriod]
        
        comparisons = []
        for i in range(len(periods) - 1):
            comp = self.contrastive.compare_periods(
                periods[i], periods[i+1], feature_type
            )
            comparisons.append(comp)
        
        return {
            "feature_type": feature_type,
            "periods": periods,
            "comparisons": comparisons
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diachronic Analysis")
    parser.add_argument('command', choices=['analyze', 'compare', 'report'],
                       help="Command to run")
    parser.add_argument('--text', '-t', help="Text to analyze")
    parser.add_argument('--period', '-p', help="Period")
    parser.add_argument('--period1', help="First period for comparison")
    parser.add_argument('--period2', help="Second period for comparison")
    parser.add_argument('--lemma', '-l', help="Lemma for report")
    parser.add_argument('--feature', '-f', default="all", help="Feature type")
    
    args = parser.parse_args()
    
    engine = DiachronicAnalysisEngine()
    
    if args.command == 'analyze':
        if args.text and args.period:
            result = engine.analyze_text(args.text, args.period)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Requires --text and --period")
    
    elif args.command == 'compare':
        if args.period1 and args.period2:
            result = engine.contrastive.compare_periods(
                args.period1, args.period2, args.feature
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Requires --period1 and --period2")
    
    elif args.command == 'report':
        if args.lemma:
            result = engine.generate_diachronic_report(args.lemma)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Requires --lemma")


if __name__ == "__main__":
    main()
