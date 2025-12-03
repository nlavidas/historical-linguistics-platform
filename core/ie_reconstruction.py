#!/usr/bin/env python3
"""
INDO-EUROPEAN RECONSTRUCTION ENGINE
Tools for Proto-Indo-European reconstruction and comparative linguistics

Features:
1. Sound correspondence detection
2. Cognate identification
3. PIE root reconstruction
4. Semantic shift tracking
5. Morphological reconstruction
6. Laryngeal theory support
7. Ablaut grade analysis
"""

import os
import re
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PIE PHONOLOGY
# =============================================================================

class PIEPhoneme(Enum):
    """Proto-Indo-European phonemes"""
    # Stops
    P = 'p'
    B = 'b'
    BH = 'bʰ'
    T = 't'
    D = 'd'
    DH = 'dʰ'
    K = 'k'
    G = 'g'
    GH = 'gʰ'
    KW = 'kʷ'
    GW = 'gʷ'
    GWH = 'gʷʰ'
    # Laryngeals
    H1 = 'h₁'
    H2 = 'h₂'
    H3 = 'h₃'
    # Resonants
    M = 'm'
    N = 'n'
    R = 'r'
    L = 'l'
    Y = 'y'
    W = 'w'
    # Vowels
    E = 'e'
    O = 'o'
    E_LONG = 'ē'
    O_LONG = 'ō'

class AblautGrade(Enum):
    """Ablaut grades"""
    E_GRADE = 'e-grade'  # Full grade with e
    O_GRADE = 'o-grade'  # Full grade with o
    ZERO_GRADE = 'zero-grade'  # No vowel
    LENGTHENED_E = 'lengthened-e'  # ē
    LENGTHENED_O = 'lengthened-o'  # ō

# =============================================================================
# SOUND CORRESPONDENCES
# =============================================================================

# PIE to daughter language correspondences
SOUND_CORRESPONDENCES = {
    'grc': {  # Ancient Greek
        'p': 'π',
        'b': 'β',
        'bʰ': 'φ',
        't': 'τ',
        'd': 'δ',
        'dʰ': 'θ',
        'k': 'κ',
        'g': 'γ',
        'gʰ': 'χ',
        'kʷ': ['π', 'τ', 'κ'],  # Context-dependent
        'gʷ': ['β', 'δ', 'γ'],
        'gʷʰ': ['φ', 'θ', 'χ'],
        'h₁': '',  # Lost
        'h₂': 'α',  # Colors adjacent vowel
        'h₃': 'ο',  # Colors adjacent vowel
        'm': 'μ',
        'n': 'ν',
        'r': 'ρ',
        'l': 'λ',
        'y': '',  # Various reflexes
        'w': '',  # Lost in most positions
        's': ['σ', 'ἁ'],  # Initial s > h (rough breathing)
        'e': 'ε',
        'o': 'ο',
        'ē': 'η',
        'ō': 'ω',
    },
    'la': {  # Latin
        'p': 'p',
        'b': 'b',
        'bʰ': ['f', 'b'],  # f initially, b medially
        't': 't',
        'd': 'd',
        'dʰ': ['f', 'd'],
        'k': 'c',
        'g': 'g',
        'gʰ': ['h', 'g'],
        'kʷ': 'qu',
        'gʷ': ['v', 'gu'],
        'gʷʰ': ['f', 'gu'],
        'h₁': '',
        'h₂': 'a',
        'h₃': 'o',
        'm': 'm',
        'n': 'n',
        'r': 'r',
        'l': 'l',
        'y': 'i',
        'w': 'v',
        's': 's',
        'e': 'e',
        'o': 'o',
        'ē': 'ē',
        'ō': 'ō',
    },
    'got': {  # Gothic
        'p': 'f',  # Grimm's Law
        'b': 'p',
        'bʰ': 'b',
        't': 'þ',
        'd': 't',
        'dʰ': 'd',
        'k': 'h',
        'g': 'k',
        'gʰ': 'g',
        'kʷ': 'ƕ',
        'gʷ': 'q',
        'gʷʰ': 'g',
        'm': 'm',
        'n': 'n',
        'r': 'r',
        'l': 'l',
        'y': 'j',
        'w': 'w',
        's': 's',
        'e': ['i', 'ai'],
        'o': ['u', 'au'],
    },
    'sa': {  # Sanskrit
        'p': 'p',
        'b': 'b',
        'bʰ': 'bh',
        't': 't',
        'd': 'd',
        'dʰ': 'dh',
        'k': ['k', 'c', 'ś'],  # Satem
        'g': ['g', 'j'],
        'gʰ': ['gh', 'h'],
        'kʷ': 'k',  # Merged with plain velars
        'gʷ': 'g',
        'gʷʰ': 'gh',
        'h₁': '',
        'h₂': ['a', 'i'],
        'h₃': ['a', 'u'],
        'm': 'm',
        'n': 'n',
        'r': 'r',
        'l': ['l', 'r'],
        'y': 'y',
        'w': 'v',
        's': ['s', 'ṣ'],
        'e': 'a',  # Merged
        'o': 'a',
        'ē': 'ā',
        'ō': 'ā',
    }
}

# =============================================================================
# PIE ROOTS DATABASE
# =============================================================================

PIE_ROOTS = {
    # Motion verbs
    '*h₁ey-': {
        'meaning': 'to go',
        'reflexes': {
            'grc': ['εἶμι', 'ἰέναι'],
            'la': ['eō', 'īre'],
            'sa': ['éti', 'áyati'],
            'got': ['iddja']
        },
        'ablaut': ['*h₁éy-ti', '*h₁i-', '*h₁oy-']
    },
    '*gʷem-': {
        'meaning': 'to come, go',
        'reflexes': {
            'grc': ['βαίνω', 'βῆναι'],
            'la': ['veniō', 'venīre'],
            'sa': ['gámati'],
            'got': ['qiman']
        }
    },
    '*steh₂-': {
        'meaning': 'to stand',
        'reflexes': {
            'grc': ['ἵστημι', 'στῆναι'],
            'la': ['stō', 'stāre', 'sistō'],
            'sa': ['tíṣṭhati'],
            'got': ['standan']
        }
    },
    '*sed-': {
        'meaning': 'to sit',
        'reflexes': {
            'grc': ['ἕζομαι', 'ἵζω'],
            'la': ['sedeō', 'sedēre'],
            'sa': ['sīdati', 'sádati'],
            'got': ['sitan']
        }
    },
    
    # Perception verbs
    '*weid-': {
        'meaning': 'to see, know',
        'reflexes': {
            'grc': ['εἶδον', 'οἶδα', 'ἰδεῖν'],
            'la': ['videō', 'vidēre'],
            'sa': ['véda', 'vidáti'],
            'got': ['witan', 'wait']
        }
    },
    '*ḱlew-': {
        'meaning': 'to hear',
        'reflexes': {
            'grc': ['κλύω', 'κλέος'],
            'la': ['cluō', 'inclutus'],
            'sa': ['śṛṇóti', 'śrávas'],
            'got': ['hliuma']
        }
    },
    
    # Transfer verbs
    '*deh₃-': {
        'meaning': 'to give',
        'reflexes': {
            'grc': ['δίδωμι', 'δοῦναι'],
            'la': ['dō', 'dare', 'dōnum'],
            'sa': ['dádāti', 'dānam'],
            'got': []
        }
    },
    '*bʰer-': {
        'meaning': 'to carry, bear',
        'reflexes': {
            'grc': ['φέρω', 'φέρειν'],
            'la': ['ferō', 'ferre'],
            'sa': ['bhárati'],
            'got': ['bairan']
        }
    },
    
    # Speech verbs
    '*wekʷ-': {
        'meaning': 'to speak',
        'reflexes': {
            'grc': ['εἶπον', 'ἔπος'],
            'la': ['vōx', 'vocāre'],
            'sa': ['vákti', 'vácas'],
            'got': []
        }
    },
    
    # Basic nouns
    '*ph₂tḗr': {
        'meaning': 'father',
        'reflexes': {
            'grc': ['πατήρ'],
            'la': ['pater'],
            'sa': ['pitár-'],
            'got': ['fadar']
        }
    },
    '*méh₂tēr': {
        'meaning': 'mother',
        'reflexes': {
            'grc': ['μήτηρ'],
            'la': ['māter'],
            'sa': ['mātár-'],
            'got': []
        }
    },
    '*bʰréh₂tēr': {
        'meaning': 'brother',
        'reflexes': {
            'grc': ['φράτηρ', 'φράτωρ'],
            'la': ['frāter'],
            'sa': ['bhrā́tar-'],
            'got': ['broþar']
        }
    },
    '*swésōr': {
        'meaning': 'sister',
        'reflexes': {
            'grc': [],
            'la': ['soror'],
            'sa': ['svásar-'],
            'got': ['swistar']
        }
    },
    
    # Body parts
    '*h₃ekʷ-': {
        'meaning': 'eye',
        'reflexes': {
            'grc': ['ὄψ', 'ὄμμα', 'ὄσσε'],
            'la': ['oculus'],
            'sa': ['ákṣi'],
            'got': ['augo']
        }
    },
    '*h₂ews-': {
        'meaning': 'ear',
        'reflexes': {
            'grc': ['οὖς'],
            'la': ['auris'],
            'sa': [],
            'got': ['auso']
        }
    },
    
    # Numbers
    '*sem-': {
        'meaning': 'one',
        'reflexes': {
            'grc': ['εἷς', 'μία', 'ἕν'],
            'la': ['semel', 'similis'],
            'sa': ['sám'],
            'got': ['sums']
        }
    },
    '*dwóh₁': {
        'meaning': 'two',
        'reflexes': {
            'grc': ['δύο'],
            'la': ['duo'],
            'sa': ['dvā́'],
            'got': ['twai']
        }
    },
    '*tréyes': {
        'meaning': 'three',
        'reflexes': {
            'grc': ['τρεῖς'],
            'la': ['trēs'],
            'sa': ['tráyas'],
            'got': ['þreis']
        }
    },
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Cognate:
    """A cognate set across languages"""
    pie_root: str
    meaning: str
    forms: Dict[str, List[str]]  # language -> forms
    confidence: float = 0.0
    notes: str = ""

@dataclass
class SoundChange:
    """A sound change rule"""
    source: str
    target: str
    environment: str  # e.g., "word-initial", "before vowel"
    language: str
    period: str = ""
    examples: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class Reconstruction:
    """A PIE reconstruction"""
    form: str
    meaning: str
    pos: str
    evidence: List[Dict]
    confidence: float
    notes: str = ""

# =============================================================================
# COGNATE FINDER
# =============================================================================

class CognateFinder:
    """Find cognates across IE languages"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.correspondences = SOUND_CORRESPONDENCES
        self.known_roots = PIE_ROOTS
    
    def find_cognates(self, lemma: str, source_lang: str) -> List[Cognate]:
        """Find cognates for a lemma"""
        cognates = []
        
        # Check known roots first
        for root, data in self.known_roots.items():
            reflexes = data.get('reflexes', {})
            if source_lang in reflexes:
                if lemma in reflexes[source_lang]:
                    # Found in known cognate set
                    cognates.append(Cognate(
                        pie_root=root,
                        meaning=data['meaning'],
                        forms=reflexes,
                        confidence=1.0,
                        notes="From PIE roots database"
                    ))
        
        # Try to find by sound correspondences
        if not cognates:
            potential = self._find_by_correspondences(lemma, source_lang)
            cognates.extend(potential)
        
        return cognates
    
    def _find_by_correspondences(self, lemma: str, source_lang: str) -> List[Cognate]:
        """Find potential cognates using sound correspondences"""
        # This would use the database to find similar forms
        # For now, return empty list
        return []
    
    def verify_cognate(self, forms: Dict[str, str]) -> Tuple[bool, float, str]:
        """Verify if forms are cognates"""
        # Check sound correspondences
        violations = []
        matches = 0
        total_checks = 0
        
        languages = list(forms.keys())
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                form1 = forms[lang1]
                form2 = forms[lang2]
                
                # Check initial consonant
                if form1 and form2:
                    c1 = form1[0] if form1 else ''
                    c2 = form2[0] if form2 else ''
                    
                    # Check if correspondence is valid
                    # (simplified check)
                    total_checks += 1
                    # Would need full correspondence checking here
        
        confidence = matches / total_checks if total_checks > 0 else 0.0
        
        return len(violations) == 0, confidence, "; ".join(violations)


# =============================================================================
# RECONSTRUCTION ENGINE
# =============================================================================

class ReconstructionEngine:
    """Reconstruct PIE forms from daughter language evidence"""
    
    def __init__(self):
        self.correspondences = SOUND_CORRESPONDENCES
    
    def reconstruct(self, cognate_set: Dict[str, str]) -> Reconstruction:
        """Reconstruct PIE form from cognates"""
        evidence = []
        
        # Analyze each form
        for lang, form in cognate_set.items():
            if lang in self.correspondences:
                analysis = self._analyze_form(form, lang)
                evidence.append({
                    'language': lang,
                    'form': form,
                    'analysis': analysis
                })
        
        # Attempt reconstruction
        pie_form = self._build_reconstruction(evidence)
        
        return Reconstruction(
            form=pie_form,
            meaning="",  # Would need semantic analysis
            pos="",
            evidence=evidence,
            confidence=self._calculate_confidence(evidence)
        )
    
    def _analyze_form(self, form: str, language: str) -> Dict:
        """Analyze a form in terms of PIE correspondences"""
        analysis = {
            'segments': [],
            'possible_pie': []
        }
        
        corr = self.correspondences.get(language, {})
        
        # Reverse mapping
        reverse_map = {}
        for pie, reflex in corr.items():
            if isinstance(reflex, list):
                for r in reflex:
                    if r not in reverse_map:
                        reverse_map[r] = []
                    reverse_map[r].append(pie)
            else:
                if reflex not in reverse_map:
                    reverse_map[reflex] = []
                reverse_map[reflex].append(pie)
        
        # Analyze each character
        for char in form:
            if char in reverse_map:
                analysis['segments'].append({
                    'char': char,
                    'pie_options': reverse_map[char]
                })
            else:
                analysis['segments'].append({
                    'char': char,
                    'pie_options': [char]  # Assume unchanged
                })
        
        return analysis
    
    def _build_reconstruction(self, evidence: List[Dict]) -> str:
        """Build PIE reconstruction from evidence"""
        if not evidence:
            return "*?"
        
        # Simple approach: use first form's analysis
        first = evidence[0]['analysis']
        
        pie_form = "*"
        for seg in first.get('segments', []):
            options = seg.get('pie_options', [])
            if options:
                pie_form += options[0]
        
        return pie_form
    
    def _calculate_confidence(self, evidence: List[Dict]) -> float:
        """Calculate confidence in reconstruction"""
        if not evidence:
            return 0.0
        
        # More languages = higher confidence
        lang_count = len(evidence)
        base_confidence = min(lang_count / 4, 1.0)  # Max at 4 languages
        
        return base_confidence


# =============================================================================
# ABLAUT ANALYZER
# =============================================================================

class AblautAnalyzer:
    """Analyze ablaut patterns"""
    
    # Ablaut patterns
    PATTERNS = {
        'e-grade': ['e'],
        'o-grade': ['o'],
        'zero-grade': [''],
        'lengthened-e': ['ē'],
        'lengthened-o': ['ō']
    }
    
    def analyze_root(self, root: str) -> Dict:
        """Analyze ablaut grades of a root"""
        # Extract the vowel
        vowel_match = re.search(r'[eēoō]', root)
        
        if not vowel_match:
            return {'root': root, 'grades': {}}
        
        vowel = vowel_match.group()
        vowel_pos = vowel_match.start()
        
        grades = {}
        
        # Generate all grades
        prefix = root[:vowel_pos]
        suffix = root[vowel_pos + 1:]
        
        grades['e-grade'] = prefix + 'e' + suffix
        grades['o-grade'] = prefix + 'o' + suffix
        grades['zero-grade'] = prefix + suffix
        grades['lengthened-e'] = prefix + 'ē' + suffix
        grades['lengthened-o'] = prefix + 'ō' + suffix
        
        return {
            'root': root,
            'base_vowel': vowel,
            'grades': grades
        }
    
    def identify_grade(self, form: str, root: str) -> Optional[str]:
        """Identify which ablaut grade a form represents"""
        analysis = self.analyze_root(root)
        
        for grade, pattern in analysis['grades'].items():
            if pattern in form or form in pattern:
                return grade
        
        return None


# =============================================================================
# SEMANTIC SHIFT TRACKER
# =============================================================================

class SemanticShiftTracker:
    """Track semantic shifts across time and languages"""
    
    # Common semantic shift patterns
    SHIFT_PATTERNS = {
        'narrowing': 'General meaning becomes more specific',
        'broadening': 'Specific meaning becomes more general',
        'amelioration': 'Meaning becomes more positive',
        'pejoration': 'Meaning becomes more negative',
        'metaphor': 'Concrete to abstract or vice versa',
        'metonymy': 'Part for whole or associated concept',
        'synecdoche': 'Part for whole',
        'euphemism': 'Taboo avoidance',
    }
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def track_shifts(self, lemma: str, language: str) -> List[Dict]:
        """Track semantic shifts for a lemma"""
        shifts = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get meanings by period
        cursor.execute("""
            SELECT DISTINCT d.period, t.misc
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            JOIN documents d ON s.document_id = d.id
            WHERE t.lemma = ?
            ORDER BY d.period
        """, (lemma,))
        
        meanings_by_period = defaultdict(set)
        for period, misc in cursor.fetchall():
            if misc and misc != '_':
                meanings_by_period[period].add(misc)
        
        conn.close()
        
        # Detect shifts between periods
        periods = sorted(meanings_by_period.keys())
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i+1]
            m1 = meanings_by_period[p1]
            m2 = meanings_by_period[p2]
            
            # Check for changes
            new_meanings = m2 - m1
            lost_meanings = m1 - m2
            
            if new_meanings or lost_meanings:
                shifts.append({
                    'from_period': p1,
                    'to_period': p2,
                    'new_meanings': list(new_meanings),
                    'lost_meanings': list(lost_meanings)
                })
        
        return shifts


# =============================================================================
# MORPHOLOGICAL RECONSTRUCTION
# =============================================================================

class MorphologicalReconstructor:
    """Reconstruct PIE morphology"""
    
    # PIE nominal endings
    PIE_NOMINAL_ENDINGS = {
        'thematic': {
            'nom_sg': '-os',
            'gen_sg': '-osyo',
            'dat_sg': '-ōi',
            'acc_sg': '-om',
            'voc_sg': '-e',
            'nom_pl': '-ōs',
            'gen_pl': '-ōm',
            'dat_pl': '-oybʰos',
            'acc_pl': '-ons',
        },
        'athematic': {
            'nom_sg': '-s',
            'gen_sg': '-és/-ós',
            'dat_sg': '-éy',
            'acc_sg': '-m̥',
            'voc_sg': '-∅',
            'nom_pl': '-es',
            'gen_pl': '-óm',
            'dat_pl': '-bʰyós',
            'acc_pl': '-n̥s',
        }
    }
    
    # PIE verbal endings
    PIE_VERBAL_ENDINGS = {
        'primary_active': {
            '1sg': '-mi',
            '2sg': '-si',
            '3sg': '-ti',
            '1pl': '-mos',
            '2pl': '-te',
            '3pl': '-nti',
        },
        'secondary_active': {
            '1sg': '-m',
            '2sg': '-s',
            '3sg': '-t',
            '1pl': '-me',
            '2pl': '-te',
            '3pl': '-nt',
        },
        'perfect': {
            '1sg': '-h₂e',
            '2sg': '-th₂e',
            '3sg': '-e',
            '1pl': '-mé',
            '2pl': '-é',
            '3pl': '-ḗr',
        }
    }
    
    def reconstruct_paradigm(self, root: str, pos: str) -> Dict:
        """Reconstruct full paradigm"""
        if pos == 'NOUN':
            return self._reconstruct_nominal(root)
        elif pos == 'VERB':
            return self._reconstruct_verbal(root)
        else:
            return {}
    
    def _reconstruct_nominal(self, root: str) -> Dict:
        """Reconstruct nominal paradigm"""
        paradigm = {'root': root, 'forms': {}}
        
        # Assume thematic for simplicity
        for case_num, ending in self.PIE_NOMINAL_ENDINGS['thematic'].items():
            paradigm['forms'][case_num] = f"*{root}{ending}"
        
        return paradigm
    
    def _reconstruct_verbal(self, root: str) -> Dict:
        """Reconstruct verbal paradigm"""
        paradigm = {'root': root, 'tenses': {}}
        
        # Present
        paradigm['tenses']['present'] = {}
        for person, ending in self.PIE_VERBAL_ENDINGS['primary_active'].items():
            paradigm['tenses']['present'][person] = f"*{root}{ending}"
        
        # Aorist
        paradigm['tenses']['aorist'] = {}
        for person, ending in self.PIE_VERBAL_ENDINGS['secondary_active'].items():
            paradigm['tenses']['aorist'][person] = f"*{root}{ending}"
        
        # Perfect
        paradigm['tenses']['perfect'] = {}
        for person, ending in self.PIE_VERBAL_ENDINGS['perfect'].items():
            # Perfect has reduplication
            paradigm['tenses']['perfect'][person] = f"*{root[0]}e-{root}{ending}"
        
        return paradigm


# =============================================================================
# IE COMPARATIVE DATABASE
# =============================================================================

class IEComparativeDB:
    """Database for IE comparative data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # PIE roots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pie_roots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root TEXT NOT NULL UNIQUE,
                meaning TEXT,
                pos TEXT,
                ablaut_grades TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cognates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_id INTEGER,
                language TEXT NOT NULL,
                form TEXT NOT NULL,
                meaning TEXT,
                period TEXT,
                source TEXT,
                FOREIGN KEY (root_id) REFERENCES pie_roots(id)
            )
        """)
        
        # Sound changes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sound_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_phoneme TEXT NOT NULL,
                target_phoneme TEXT NOT NULL,
                language TEXT NOT NULL,
                environment TEXT,
                period TEXT,
                examples TEXT
            )
        """)
        
        # Semantic shifts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                language TEXT NOT NULL,
                old_meaning TEXT,
                new_meaning TEXT,
                shift_type TEXT,
                from_period TEXT,
                to_period TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_root(self, root: str, meaning: str, pos: str = None) -> int:
        """Add a PIE root"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pie_roots (root, meaning, pos)
            VALUES (?, ?, ?)
        """, (root, meaning, pos))
        
        root_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return root_id
    
    def add_cognate(self, root_id: int, language: str, form: str,
                    meaning: str = None, period: str = None):
        """Add a cognate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cognates (root_id, language, form, meaning, period)
            VALUES (?, ?, ?, ?, ?)
        """, (root_id, language, form, meaning, period))
        
        conn.commit()
        conn.close()
    
    def get_cognates(self, root: str) -> List[Dict]:
        """Get all cognates for a root"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.* FROM cognates c
            JOIN pie_roots r ON c.root_id = r.id
            WHERE r.root = ?
        """, (root,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def populate_from_known_roots(self):
        """Populate database with known PIE roots"""
        for root, data in PIE_ROOTS.items():
            root_id = self.add_root(root, data['meaning'])
            
            for lang, forms in data.get('reflexes', {}).items():
                for form in forms:
                    self.add_cognate(root_id, lang, form)
        
        logger.info(f"Populated {len(PIE_ROOTS)} PIE roots")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "/root/corpus_platform/data/ie_comparative.db"
    
    print("=" * 70)
    print("INDO-EUROPEAN RECONSTRUCTION ENGINE")
    print("=" * 70)
    
    # Initialize database
    ie_db = IEComparativeDB(db_path)
    ie_db.populate_from_known_roots()
    
    # Test cognate finder
    finder = CognateFinder(db_path)
    
    print("\n📚 Testing cognate finder:")
    test_words = ['πατήρ', 'φέρω', 'εἶμι']
    for word in test_words:
        cognates = finder.find_cognates(word, 'grc')
        if cognates:
            print(f"\n  {word}:")
            for cog in cognates:
                print(f"    PIE: {cog.pie_root} '{cog.meaning}'")
                for lang, forms in cog.forms.items():
                    print(f"      {lang}: {', '.join(forms)}")
    
    # Test ablaut analyzer
    print("\n🔄 Testing ablaut analyzer:")
    ablaut = AblautAnalyzer()
    
    for root in ['*bʰer-', '*weid-', '*steh₂-']:
        analysis = ablaut.analyze_root(root.replace('*', ''))
        print(f"\n  {root}:")
        for grade, form in analysis['grades'].items():
            print(f"    {grade}: *{form}")
    
    # Test morphological reconstruction
    print("\n📝 Testing morphological reconstruction:")
    morph = MorphologicalReconstructor()
    
    paradigm = morph.reconstruct_paradigm('bʰer', 'VERB')
    print(f"\n  *bʰer- 'to carry' (present):")
    for person, form in paradigm['tenses']['present'].items():
        print(f"    {person}: {form}")
    
    print("\n✅ IE Reconstruction Engine ready!")
