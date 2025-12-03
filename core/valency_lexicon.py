"""
Valency Lexicon for Greek Verbs
Comprehensive argument structure database
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CASE AND ARGUMENT TYPES
# =============================================================================

class Case(Enum):
    """Greek grammatical cases"""
    NOM = "nominative"
    GEN = "genitive"
    DAT = "dative"
    ACC = "accusative"
    VOC = "vocative"


class ArgumentType(Enum):
    """Types of verbal arguments"""
    SUBJECT = "subject"
    DIRECT_OBJECT = "direct_object"
    INDIRECT_OBJECT = "indirect_object"
    OBLIQUE = "oblique"
    COMPLEMENT = "complement"
    ADJUNCT = "adjunct"


# =============================================================================
# VALENCY PATTERNS
# =============================================================================

@dataclass
class ValencyArgument:
    """A single argument in a valency pattern"""
    
    role: str  # ARG0, ARG1, ARG2, etc.
    case: str  # nominative, genitive, dative, accusative
    preposition: str = ""  # Optional preposition
    optional: bool = False
    semantic_role: str = ""  # agent, patient, theme, etc.
    description: str = ""
    
    def to_pattern_string(self) -> str:
        """Convert to pattern string like 'NOM', 'ACC', 'DAT', 'εἰς+ACC'"""
        if self.preposition:
            return f"{self.preposition}+{self.case.upper()[:3]}"
        return self.case.upper()[:3]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValencyFrame:
    """A complete valency frame for a verb sense"""
    
    lemma: str
    sense_id: str
    sense_description: str = ""
    
    # Arguments
    arguments: List[ValencyArgument] = field(default_factory=list)
    
    # Pattern string (e.g., "NOM + ACC + DAT")
    pattern: str = ""
    
    # Semantic class
    semantic_class: str = ""
    
    # Alternations (e.g., passive, causative)
    alternations: List[str] = field(default_factory=list)
    
    # Examples
    examples: List[Dict] = field(default_factory=list)
    
    # Frequency
    frequency: int = 0
    
    # Period restrictions
    period_from: str = ""
    period_to: str = ""
    
    def get_pattern_string(self) -> str:
        """Generate pattern string from arguments"""
        if self.pattern:
            return self.pattern
        return " + ".join(arg.to_pattern_string() for arg in self.arguments if not arg.optional)
    
    def to_dict(self) -> Dict:
        return {
            'lemma': self.lemma,
            'sense_id': self.sense_id,
            'sense_description': self.sense_description,
            'arguments': [a.to_dict() for a in self.arguments],
            'pattern': self.get_pattern_string(),
            'semantic_class': self.semantic_class,
            'alternations': self.alternations,
            'examples': self.examples,
            'frequency': self.frequency
        }


@dataclass
class ValencyEntry:
    """Complete valency entry for a verb lemma"""
    
    lemma: str
    pos: str = "verb"
    
    # All senses/frames
    frames: List[ValencyFrame] = field(default_factory=list)
    
    # General information
    transitivity: str = ""  # transitive, intransitive, ditransitive, ambitransitive
    voice_alternations: List[str] = field(default_factory=list)
    
    # Morphological information
    conjugation_class: str = ""
    principal_parts: List[str] = field(default_factory=list)
    
    # Etymology
    etymology: str = ""
    cognates: List[str] = field(default_factory=list)
    
    def get_all_patterns(self) -> List[str]:
        """Get all unique patterns"""
        return list(set(f.get_pattern_string() for f in self.frames))
    
    def to_dict(self) -> Dict:
        return {
            'lemma': self.lemma,
            'pos': self.pos,
            'frames': [f.to_dict() for f in self.frames],
            'transitivity': self.transitivity,
            'voice_alternations': self.voice_alternations,
            'conjugation_class': self.conjugation_class,
            'principal_parts': self.principal_parts
        }


# =============================================================================
# GREEK VALENCY LEXICON DATA
# =============================================================================

GREEK_VALENCY_LEXICON = {
    # ==========================================================================
    # VERBS OF GIVING/TRANSFER
    # ==========================================================================
    "δίδωμι": ValencyEntry(
        lemma="δίδωμι",
        transitivity="ditransitive",
        conjugation_class="μι-verb",
        principal_parts=["δίδωμι", "δώσω", "ἔδωκα", "δέδωκα", "δέδομαι", "ἐδόθην"],
        frames=[
            ValencyFrame(
                lemma="δίδωμι",
                sense_id="give.01",
                sense_description="transfer possession",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent", description="giver"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme", description="thing given"),
                    ValencyArgument(role="ARG2", case="dative", semantic_role="recipient", description="recipient"),
                ],
                pattern="NOM + ACC + DAT",
                semantic_class="transfer",
                examples=[
                    {"text": "ὁ πατὴρ δίδωσι τῷ παιδὶ τὸ βιβλίον", "gloss": "The father gives the book to the child"}
                ]
            ),
            ValencyFrame(
                lemma="δίδωμι",
                sense_id="give.02",
                sense_description="grant, allow",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="transfer"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF COMMUNICATION
    # ==========================================================================
    "λέγω": ValencyEntry(
        lemma="λέγω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["λέγω", "λέξω/ἐρῶ", "ἔλεξα/εἶπον", "εἴρηκα", "λέλεγμαι/εἴρημαι", "ἐλέχθην/ἐρρήθην"],
        frames=[
            ValencyFrame(
                lemma="λέγω",
                sense_id="say.01",
                sense_description="say, speak",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="speaker"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                ],
                pattern="NOM + ACC",
                semantic_class="communication"
            ),
            ValencyFrame(
                lemma="λέγω",
                sense_id="say.02",
                sense_description="say to someone",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="speaker"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                    ValencyArgument(role="ARG2", case="dative", semantic_role="addressee"),
                ],
                pattern="NOM + ACC + DAT",
                semantic_class="communication"
            ),
            ValencyFrame(
                lemma="λέγω",
                sense_id="say.03",
                sense_description="say that (with ὅτι clause)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="speaker"),
                    ValencyArgument(role="ARG1", case="clause", semantic_role="content", description="ὅτι clause"),
                ],
                pattern="NOM + ὅτι-clause",
                semantic_class="communication"
            ),
            ValencyFrame(
                lemma="λέγω",
                sense_id="say.04",
                sense_description="say (with infinitive)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="speaker"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                ],
                pattern="NOM + INF",
                semantic_class="communication"
            )
        ]
    ),
    
    "φημί": ValencyEntry(
        lemma="φημί",
        transitivity="transitive",
        conjugation_class="μι-verb",
        frames=[
            ValencyFrame(
                lemma="φημί",
                sense_id="say.01",
                sense_description="say, assert",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="speaker"),
                    ValencyArgument(role="ARG1", case="accusative+infinitive", semantic_role="content"),
                ],
                pattern="NOM + ACC+INF",
                semantic_class="communication"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF MOTION
    # ==========================================================================
    "ἔρχομαι": ValencyEntry(
        lemma="ἔρχομαι",
        transitivity="intransitive",
        conjugation_class="deponent",
        principal_parts=["ἔρχομαι", "ἐλεύσομαι", "ἦλθον", "ἐλήλυθα", "-", "-"],
        frames=[
            ValencyFrame(
                lemma="ἔρχομαι",
                sense_id="come.01",
                sense_description="come, go",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="theme"),
                ],
                pattern="NOM",
                semantic_class="motion"
            ),
            ValencyFrame(
                lemma="ἔρχομαι",
                sense_id="come.02",
                sense_description="come to a place",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG4", case="accusative", preposition="εἰς", semantic_role="goal"),
                ],
                pattern="NOM + εἰς+ACC",
                semantic_class="motion"
            ),
            ValencyFrame(
                lemma="ἔρχομαι",
                sense_id="come.03",
                sense_description="come from a place",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG3", case="genitive", preposition="ἐκ", semantic_role="source"),
                ],
                pattern="NOM + ἐκ+GEN",
                semantic_class="motion"
            )
        ]
    ),
    
    "ἄγω": ValencyEntry(
        lemma="ἄγω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["ἄγω", "ἄξω", "ἤγαγον", "ἦχα", "ἦγμαι", "ἤχθην"],
        frames=[
            ValencyFrame(
                lemma="ἄγω",
                sense_id="lead.01",
                sense_description="lead, bring",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="motion_caused"
            ),
            ValencyFrame(
                lemma="ἄγω",
                sense_id="lead.02",
                sense_description="lead to a place",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                    ValencyArgument(role="ARG4", case="accusative", preposition="εἰς", semantic_role="goal"),
                ],
                pattern="NOM + ACC + εἰς+ACC",
                semantic_class="motion_caused"
            )
        ]
    ),
    
    "φέρω": ValencyEntry(
        lemma="φέρω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["φέρω", "οἴσω", "ἤνεγκον/ἤνεγκα", "ἐνήνοχα", "ἐνήνεγμαι", "ἠνέχθην"],
        frames=[
            ValencyFrame(
                lemma="φέρω",
                sense_id="carry.01",
                sense_description="carry, bear",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="motion_caused"
            )
        ]
    ),
    
    "πέμπω": ValencyEntry(
        lemma="πέμπω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["πέμπω", "πέμψω", "ἔπεμψα", "πέπομφα", "πέπεμμαι", "ἐπέμφθην"],
        frames=[
            ValencyFrame(
                lemma="πέμπω",
                sense_id="send.01",
                sense_description="send",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="transfer"
            ),
            ValencyFrame(
                lemma="πέμπω",
                sense_id="send.02",
                sense_description="send to",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                    ValencyArgument(role="ARG4", case="accusative", preposition="εἰς", semantic_role="goal"),
                ],
                pattern="NOM + ACC + εἰς+ACC",
                semantic_class="transfer"
            ),
            ValencyFrame(
                lemma="πέμπω",
                sense_id="send.03",
                sense_description="send to someone",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="dative", semantic_role="recipient"),
                ],
                pattern="NOM + ACC + DAT",
                semantic_class="transfer"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF PERCEPTION
    # ==========================================================================
    "ὁράω": ValencyEntry(
        lemma="ὁράω",
        transitivity="transitive",
        conjugation_class="contract",
        principal_parts=["ὁράω", "ὄψομαι", "εἶδον", "ἑώρακα/ἑόρακα", "ἑώραμαι/ὦμμαι", "ὤφθην"],
        frames=[
            ValencyFrame(
                lemma="ὁράω",
                sense_id="see.01",
                sense_description="see, perceive visually",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="stimulus"),
                ],
                pattern="NOM + ACC",
                semantic_class="perception"
            ),
            ValencyFrame(
                lemma="ὁράω",
                sense_id="see.02",
                sense_description="see that",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="clause", semantic_role="stimulus", description="ὅτι clause"),
                ],
                pattern="NOM + ὅτι-clause",
                semantic_class="perception"
            ),
            ValencyFrame(
                lemma="ὁράω",
                sense_id="see.03",
                sense_description="see someone doing",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="stimulus"),
                    ValencyArgument(role="ARG2", case="participle", semantic_role="activity"),
                ],
                pattern="NOM + ACC + PTCP",
                semantic_class="perception"
            )
        ]
    ),
    
    "ἀκούω": ValencyEntry(
        lemma="ἀκούω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["ἀκούω", "ἀκούσομαι", "ἤκουσα", "ἀκήκοα", "ἤκουσμαι", "ἠκούσθην"],
        frames=[
            ValencyFrame(
                lemma="ἀκούω",
                sense_id="hear.01",
                sense_description="hear (a sound/thing)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="stimulus"),
                ],
                pattern="NOM + ACC",
                semantic_class="perception"
            ),
            ValencyFrame(
                lemma="ἀκούω",
                sense_id="hear.02",
                sense_description="hear/listen to (a person)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="genitive", semantic_role="source"),
                ],
                pattern="NOM + GEN",
                semantic_class="perception"
            ),
            ValencyFrame(
                lemma="ἀκούω",
                sense_id="hear.03",
                sense_description="hear that",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="clause", semantic_role="content", description="ὅτι/ὡς clause"),
                ],
                pattern="NOM + ὅτι-clause",
                semantic_class="perception"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF COGNITION
    # ==========================================================================
    "οἶδα": ValencyEntry(
        lemma="οἶδα",
        transitivity="transitive",
        conjugation_class="perfect",
        frames=[
            ValencyFrame(
                lemma="οἶδα",
                sense_id="know.01",
                sense_description="know (a fact/thing)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                ],
                pattern="NOM + ACC",
                semantic_class="cognition"
            ),
            ValencyFrame(
                lemma="οἶδα",
                sense_id="know.02",
                sense_description="know that",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="clause", semantic_role="content"),
                ],
                pattern="NOM + ὅτι-clause",
                semantic_class="cognition"
            ),
            ValencyFrame(
                lemma="οἶδα",
                sense_id="know.03",
                sense_description="know how to",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                ],
                pattern="NOM + INF",
                semantic_class="cognition"
            )
        ]
    ),
    
    "νομίζω": ValencyEntry(
        lemma="νομίζω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        frames=[
            ValencyFrame(
                lemma="νομίζω",
                sense_id="think.01",
                sense_description="think, believe",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative+infinitive", semantic_role="content"),
                ],
                pattern="NOM + ACC+INF",
                semantic_class="cognition"
            ),
            ValencyFrame(
                lemma="νομίζω",
                sense_id="think.02",
                sense_description="think that",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="clause", semantic_role="content"),
                ],
                pattern="NOM + ὅτι-clause",
                semantic_class="cognition"
            )
        ]
    ),
    
    "μανθάνω": ValencyEntry(
        lemma="μανθάνω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["μανθάνω", "μαθήσομαι", "ἔμαθον", "μεμάθηκα", "-", "-"],
        frames=[
            ValencyFrame(
                lemma="μανθάνω",
                sense_id="learn.01",
                sense_description="learn (a thing/skill)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                ],
                pattern="NOM + ACC",
                semantic_class="cognition"
            ),
            ValencyFrame(
                lemma="μανθάνω",
                sense_id="learn.02",
                sense_description="learn from someone",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                    ValencyArgument(role="ARG2", case="genitive", preposition="παρά", semantic_role="source"),
                ],
                pattern="NOM + ACC + παρά+GEN",
                semantic_class="cognition"
            )
        ]
    ),
    
    "διδάσκω": ValencyEntry(
        lemma="διδάσκω",
        transitivity="ditransitive",
        conjugation_class="ω-verb",
        principal_parts=["διδάσκω", "διδάξω", "ἐδίδαξα", "δεδίδαχα", "δεδίδαγμαι", "ἐδιδάχθην"],
        frames=[
            ValencyFrame(
                lemma="διδάσκω",
                sense_id="teach.01",
                sense_description="teach (double accusative)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                    ValencyArgument(role="ARG2", case="accusative", semantic_role="recipient"),
                ],
                pattern="NOM + ACC + ACC",
                semantic_class="transfer"
            ),
            ValencyFrame(
                lemma="διδάσκω",
                sense_id="teach.02",
                sense_description="teach to do",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                    ValencyArgument(role="ARG2", case="accusative", semantic_role="recipient"),
                ],
                pattern="NOM + ACC + INF",
                semantic_class="transfer"
            )
        ]
    ),
    
    # ==========================================================================
    # STATIVE VERBS
    # ==========================================================================
    "εἰμί": ValencyEntry(
        lemma="εἰμί",
        transitivity="copular",
        conjugation_class="μι-verb",
        principal_parts=["εἰμί", "ἔσομαι", "-", "-", "-", "-"],
        frames=[
            ValencyFrame(
                lemma="εἰμί",
                sense_id="be.01",
                sense_description="be (predicate nominative)",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="nominative", semantic_role="attribute"),
                ],
                pattern="NOM + NOM",
                semantic_class="stative"
            ),
            ValencyFrame(
                lemma="εἰμί",
                sense_id="be.02",
                sense_description="be (predicate adjective)",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="adjective", semantic_role="attribute"),
                ],
                pattern="NOM + ADJ",
                semantic_class="stative"
            ),
            ValencyFrame(
                lemma="εἰμί",
                sense_id="be.03",
                sense_description="be (locative)",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARGM-LOC", case="dative", preposition="ἐν", semantic_role="location"),
                ],
                pattern="NOM + ἐν+DAT",
                semantic_class="stative"
            ),
            ValencyFrame(
                lemma="εἰμί",
                sense_id="be.04",
                sense_description="be (possessive dative)",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="dative", semantic_role="possessor"),
                ],
                pattern="NOM + DAT",
                semantic_class="stative"
            )
        ]
    ),
    
    "γίγνομαι": ValencyEntry(
        lemma="γίγνομαι",
        transitivity="copular",
        conjugation_class="deponent",
        principal_parts=["γίγνομαι", "γενήσομαι", "ἐγενόμην", "γέγονα", "γεγένημαι", "-"],
        frames=[
            ValencyFrame(
                lemma="γίγνομαι",
                sense_id="become.01",
                sense_description="become",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="nominative", semantic_role="result"),
                ],
                pattern="NOM + NOM",
                semantic_class="change_of_state"
            ),
            ValencyFrame(
                lemma="γίγνομαι",
                sense_id="become.02",
                sense_description="happen, occur",
                arguments=[
                    ValencyArgument(role="ARG1", case="nominative", semantic_role="theme"),
                ],
                pattern="NOM",
                semantic_class="occurrence"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF CREATION
    # ==========================================================================
    "ποιέω": ValencyEntry(
        lemma="ποιέω",
        transitivity="transitive",
        conjugation_class="contract",
        principal_parts=["ποιέω", "ποιήσω", "ἐποίησα", "πεποίηκα", "πεποίημαι", "ἐποιήθην"],
        frames=[
            ValencyFrame(
                lemma="ποιέω",
                sense_id="make.01",
                sense_description="make, create",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="result"),
                ],
                pattern="NOM + ACC",
                semantic_class="creation"
            ),
            ValencyFrame(
                lemma="ποιέω",
                sense_id="make.02",
                sense_description="make X Y (double accusative)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="accusative", semantic_role="result"),
                ],
                pattern="NOM + ACC + ACC",
                semantic_class="causation"
            ),
            ValencyFrame(
                lemma="ποιέω",
                sense_id="do.01",
                sense_description="do, perform",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="activity"
            )
        ]
    ),
    
    "γράφω": ValencyEntry(
        lemma="γράφω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["γράφω", "γράψω", "ἔγραψα", "γέγραφα", "γέγραμμαι", "ἐγράφην"],
        frames=[
            ValencyFrame(
                lemma="γράφω",
                sense_id="write.01",
                sense_description="write",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="result"),
                ],
                pattern="NOM + ACC",
                semantic_class="creation"
            ),
            ValencyFrame(
                lemma="γράφω",
                sense_id="write.02",
                sense_description="write to someone",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="result"),
                    ValencyArgument(role="ARG2", case="dative", semantic_role="recipient"),
                ],
                pattern="NOM + ACC + DAT",
                semantic_class="creation"
            )
        ]
    ),
    
    # ==========================================================================
    # MODAL/DESIDERATIVE VERBS
    # ==========================================================================
    "βούλομαι": ValencyEntry(
        lemma="βούλομαι",
        transitivity="transitive",
        conjugation_class="deponent",
        principal_parts=["βούλομαι", "βουλήσομαι", "-", "-", "βεβούλημαι", "ἐβουλήθην"],
        frames=[
            ValencyFrame(
                lemma="βούλομαι",
                sense_id="want.01",
                sense_description="want, wish",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                ],
                pattern="NOM + INF",
                semantic_class="desire"
            ),
            ValencyFrame(
                lemma="βούλομαι",
                sense_id="want.02",
                sense_description="want something",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="content"),
                ],
                pattern="NOM + ACC",
                semantic_class="desire"
            )
        ]
    ),
    
    "δύναμαι": ValencyEntry(
        lemma="δύναμαι",
        transitivity="transitive",
        conjugation_class="deponent",
        principal_parts=["δύναμαι", "δυνήσομαι", "-", "-", "δεδύνημαι", "ἐδυνήθην"],
        frames=[
            ValencyFrame(
                lemma="δύναμαι",
                sense_id="can.01",
                sense_description="be able, can",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="experiencer"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                ],
                pattern="NOM + INF",
                semantic_class="modal"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF RECEIVING/TAKING
    # ==========================================================================
    "λαμβάνω": ValencyEntry(
        lemma="λαμβάνω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["λαμβάνω", "λήψομαι", "ἔλαβον", "εἴληφα", "εἴλημμαι", "ἐλήφθην"],
        frames=[
            ValencyFrame(
                lemma="λαμβάνω",
                sense_id="take.01",
                sense_description="take, receive",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="recipient"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                ],
                pattern="NOM + ACC",
                semantic_class="transfer"
            ),
            ValencyFrame(
                lemma="λαμβάνω",
                sense_id="take.02",
                sense_description="receive from",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="recipient"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="theme"),
                    ValencyArgument(role="ARG2", case="genitive", preposition="παρά", semantic_role="source"),
                ],
                pattern="NOM + ACC + παρά+GEN",
                semantic_class="transfer"
            )
        ]
    ),
    
    # ==========================================================================
    # VERBS OF COMMAND
    # ==========================================================================
    "κελεύω": ValencyEntry(
        lemma="κελεύω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["κελεύω", "κελεύσω", "ἐκέλευσα", "κεκέλευκα", "κεκέλευσμαι", "ἐκελεύσθην"],
        frames=[
            ValencyFrame(
                lemma="κελεύω",
                sense_id="order.01",
                sense_description="order, command",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="infinitive", semantic_role="content"),
                    ValencyArgument(role="ARG2", case="accusative", semantic_role="recipient"),
                ],
                pattern="NOM + ACC + INF",
                semantic_class="communication"
            )
        ]
    ),
    
    "πείθω": ValencyEntry(
        lemma="πείθω",
        transitivity="transitive",
        conjugation_class="ω-verb",
        principal_parts=["πείθω", "πείσω", "ἔπεισα", "πέπεικα/πέποιθα", "πέπεισμαι", "ἐπείσθην"],
        voice_alternations=["active: persuade", "middle: obey (+ dative)"],
        frames=[
            ValencyFrame(
                lemma="πείθω",
                sense_id="persuade.01",
                sense_description="persuade (active)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="accusative", semantic_role="recipient"),
                    ValencyArgument(role="ARG2", case="infinitive", semantic_role="content"),
                ],
                pattern="NOM + ACC + INF",
                semantic_class="communication"
            ),
            ValencyFrame(
                lemma="πείθω",
                sense_id="obey.01",
                sense_description="obey (middle/passive)",
                arguments=[
                    ValencyArgument(role="ARG0", case="nominative", semantic_role="agent"),
                    ValencyArgument(role="ARG1", case="dative", semantic_role="stimulus"),
                ],
                pattern="NOM + DAT",
                semantic_class="social"
            )
        ]
    ),
}


# =============================================================================
# VALENCY DATABASE
# =============================================================================

class ValencyDatabase:
    """Database for storing valency information"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Valency entries table
        c.execute('''CREATE TABLE IF NOT EXISTS valency_entries (
            lemma TEXT PRIMARY KEY,
            pos TEXT DEFAULT 'verb',
            transitivity TEXT,
            conjugation_class TEXT,
            principal_parts TEXT,
            voice_alternations TEXT,
            data TEXT
        )''')
        
        # Valency frames table
        c.execute('''CREATE TABLE IF NOT EXISTS valency_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lemma TEXT NOT NULL,
            sense_id TEXT,
            sense_description TEXT,
            pattern TEXT,
            semantic_class TEXT,
            arguments TEXT,
            examples TEXT,
            frequency INTEGER DEFAULT 0,
            FOREIGN KEY (lemma) REFERENCES valency_entries(lemma)
        )''')
        
        # Pattern index table
        c.execute('''CREATE TABLE IF NOT EXISTS valency_patterns (
            pattern TEXT PRIMARY KEY,
            lemmas TEXT,
            count INTEGER DEFAULT 0
        )''')
        
        # Indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_valency_frames_lemma ON valency_frames(lemma)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_valency_frames_pattern ON valency_frames(pattern)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_valency_frames_class ON valency_frames(semantic_class)')
        
        conn.commit()
        conn.close()
        
        # Load default lexicon
        self._load_default_lexicon()
    
    def _load_default_lexicon(self):
        """Load default valency lexicon"""
        conn = self.get_connection()
        c = conn.cursor()
        
        pattern_lemmas = defaultdict(list)
        
        for lemma, entry in GREEK_VALENCY_LEXICON.items():
            # Insert entry
            c.execute('''INSERT OR REPLACE INTO valency_entries 
                        (lemma, pos, transitivity, conjugation_class, principal_parts, voice_alternations, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (lemma, entry.pos, entry.transitivity, entry.conjugation_class,
                      json.dumps(entry.principal_parts), json.dumps(entry.voice_alternations),
                      json.dumps(entry.to_dict())))
            
            # Insert frames
            for frame in entry.frames:
                pattern = frame.get_pattern_string()
                
                c.execute('''INSERT INTO valency_frames 
                            (lemma, sense_id, sense_description, pattern, semantic_class, arguments, examples, frequency)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (lemma, frame.sense_id, frame.sense_description, pattern,
                          frame.semantic_class, json.dumps([a.to_dict() for a in frame.arguments]),
                          json.dumps(frame.examples), frame.frequency))
                
                pattern_lemmas[pattern].append(lemma)
        
        # Update pattern index
        for pattern, lemmas in pattern_lemmas.items():
            c.execute('''INSERT OR REPLACE INTO valency_patterns (pattern, lemmas, count)
                        VALUES (?, ?, ?)''',
                     (pattern, json.dumps(list(set(lemmas))), len(set(lemmas))))
        
        conn.commit()
        conn.close()
    
    def get_entry(self, lemma: str) -> Optional[ValencyEntry]:
        """Get valency entry by lemma"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('SELECT data FROM valency_entries WHERE lemma = ?', (lemma,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return None
        
        data = json.loads(row[0])
        
        # Reconstruct entry
        entry = ValencyEntry(
            lemma=data['lemma'],
            pos=data.get('pos', 'verb'),
            transitivity=data.get('transitivity', ''),
            conjugation_class=data.get('conjugation_class', ''),
            principal_parts=data.get('principal_parts', []),
            voice_alternations=data.get('voice_alternations', [])
        )
        
        # Get frames
        c.execute('SELECT * FROM valency_frames WHERE lemma = ?', (lemma,))
        for row in c.fetchall():
            frame = ValencyFrame(
                lemma=row[1],
                sense_id=row[2] or '',
                sense_description=row[3] or '',
                pattern=row[4] or '',
                semantic_class=row[5] or '',
                arguments=[ValencyArgument(**a) for a in json.loads(row[6])] if row[6] else [],
                examples=json.loads(row[7]) if row[7] else [],
                frequency=row[8] or 0
            )
            entry.frames.append(frame)
        
        conn.close()
        return entry
    
    def search_by_pattern(self, pattern: str) -> List[str]:
        """Search lemmas by valency pattern"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('SELECT lemmas FROM valency_patterns WHERE pattern = ?', (pattern,))
        row = c.fetchone()
        
        conn.close()
        
        if row:
            return json.loads(row[0])
        return []
    
    def get_all_patterns(self) -> Dict[str, int]:
        """Get all patterns with counts"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('SELECT pattern, count FROM valency_patterns ORDER BY count DESC')
        patterns = {r[0]: r[1] for r in c.fetchall()}
        
        conn.close()
        return patterns
    
    def get_statistics(self) -> Dict:
        """Get lexicon statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM valency_entries")
        stats['entry_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM valency_frames")
        stats['frame_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM valency_patterns")
        stats['pattern_count'] = c.fetchone()[0]
        
        c.execute("SELECT transitivity, COUNT(*) FROM valency_entries GROUP BY transitivity")
        stats['by_transitivity'] = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT semantic_class, COUNT(*) FROM valency_frames WHERE semantic_class != '' GROUP BY semantic_class")
        stats['by_semantic_class'] = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT pattern, count FROM valency_patterns ORDER BY count DESC LIMIT 10")
        stats['top_patterns'] = {r[0]: r[1] for r in c.fetchall()}
        
        conn.close()
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test valency lexicon
    print("=" * 60)
    print("GREEK VALENCY LEXICON")
    print("=" * 60)
    
    # Print sample entries
    for lemma in ["δίδωμι", "λέγω", "ἔρχομαι", "εἰμί"]:
        entry = GREEK_VALENCY_LEXICON.get(lemma)
        if entry:
            print(f"\n{lemma} ({entry.transitivity})")
            print(f"  Principal parts: {', '.join(entry.principal_parts[:3])}")
            print(f"  Patterns: {entry.get_all_patterns()}")
            for frame in entry.frames[:2]:
                print(f"    - {frame.sense_id}: {frame.get_pattern_string()}")
    
    # Initialize database
    db = ValencyDatabase("/tmp/valency_test.db")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Entries: {stats['entry_count']}")
    print(f"  Frames: {stats['frame_count']}")
    print(f"  Patterns: {stats['pattern_count']}")
    print(f"\nTop patterns:")
    for pattern, count in list(stats['top_patterns'].items())[:5]:
        print(f"  {pattern}: {count} verbs")
