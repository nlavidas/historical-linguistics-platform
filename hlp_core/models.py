"""
HLP Core Models - Domain Objects for Diachronic Linguistics

This module defines the core data structures used throughout the platform,
designed to represent linguistic data in both PROIEL/Syntacticus and
Universal Dependencies (CoNLL-U) formats.

The models support:
- Multi-layer annotation (morphology, syntax, semantics, pragmatics)
- Diachronic analysis across time periods
- Valency pattern extraction and analysis
- Cross-linguistic comparison (Greek, Latin, other IE languages)

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    Iterator, Callable, TypeVar, Generic
)
from enum import Enum, auto
from datetime import datetime
from abc import ABC, abstractmethod
import re
from collections import defaultdict


class Language(Enum):
    """Supported languages in the platform"""
    ANCIENT_GREEK = "grc"
    CLASSICAL_GREEK = "grc-cla"
    HELLENISTIC_GREEK = "grc-hel"
    BYZANTINE_GREEK = "grc-byz"
    MEDIEVAL_GREEK = "grc-med"
    EARLY_MODERN_GREEK = "grc-emo"
    MODERN_GREEK = "ell"
    LATIN = "lat"
    CLASSICAL_LATIN = "lat-cla"
    MEDIEVAL_LATIN = "lat-med"
    OLD_CHURCH_SLAVONIC = "chu"
    GOTHIC = "got"
    OLD_ARMENIAN = "xcl"
    ARMENIAN = "hy"
    OLD_ENGLISH = "ang"
    OLD_NORSE = "non"
    SANSKRIT = "san"
    AVESTAN = "ave"
    OLD_PERSIAN = "peo"
    HITTITE = "hit"
    TOCHARIAN_A = "xto"
    TOCHARIAN_B = "txb"
    OLD_IRISH = "sga"
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    ITALIAN = "it"
    SPANISH = "es"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    UNKNOWN = "und"


class Period(Enum):
    """Diachronic periods for Greek and other IE languages"""
    MYCENAEAN = "mycenaean"
    ARCHAIC = "archaic"
    CLASSICAL = "classical"
    HELLENISTIC = "hellenistic"
    ROMAN = "roman"
    ROMAN_IMPERIAL = "roman_imperial"
    LATE_ANTIQUE = "late_antique"
    OLD_LATIN = "old_latin"
    CLASSICAL_LATIN = "classical_latin"
    SILVER_LATIN = "silver_latin"
    LATE_LATIN = "late_latin"
    MEDIEVAL_LATIN = "medieval_latin"
    RENAISSANCE_LATIN = "renaissance_latin"
    BYZANTINE = "byzantine"
    EARLY_BYZANTINE = "early_byzantine"
    MIDDLE_BYZANTINE = "middle_byzantine"
    LATE_BYZANTINE = "late_byzantine"
    MEDIEVAL = "medieval"
    EARLY_MODERN = "early_modern"
    MODERN = "modern"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class Genre(Enum):
    """Text genres for corpus classification"""
    EPIC = "epic"
    LYRIC = "lyric"
    DRAMA_TRAGEDY = "drama_tragedy"
    DRAMA_COMEDY = "drama_comedy"
    HISTORIOGRAPHY = "historiography"
    PHILOSOPHY = "philosophy"
    RHETORIC = "rhetoric"
    ORATORY = "oratory"
    SCIENTIFIC = "scientific"
    MEDICAL = "medical"
    LEGAL = "legal"
    RELIGIOUS = "religious"
    BIBLICAL = "biblical"
    PATRISTIC = "patristic"
    HAGIOGRAPHY = "hagiography"
    CHRONICLE = "chronicle"
    EPISTOLARY = "epistolary"
    DOCUMENTARY = "documentary"
    INSCRIPTION = "inscription"
    PAPYRUS = "papyrus"
    COMMENTARY = "commentary"
    LEXICOGRAPHY = "lexicography"
    GRAMMAR = "grammar"
    TRANSLATION = "translation"
    UNKNOWN = "unknown"


class AnnotationStatus(Enum):
    """Status of annotation for a document or sentence"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    REVIEWED = "reviewed"
    VERIFIED = "verified"
    GOLD = "gold"
    REJECTED = "rejected"


class PartOfSpeech(Enum):
    """Universal POS tags with PROIEL extensions"""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    PRON = "PRON"
    DET = "DET"
    ADP = "ADP"
    AUX = "AUX"
    CCONJ = "CCONJ"
    SCONJ = "SCONJ"
    NUM = "NUM"
    PART = "PART"
    INTJ = "INTJ"
    PUNCT = "PUNCT"
    SYM = "SYM"
    X = "X"
    PROPN = "PROPN"
    
    NE = "Ne"
    NB = "Nb"
    NC = "Nc"
    NP = "Np"
    A_ = "A-"
    DF = "Df"
    DQ = "Dq"
    V_ = "V-"
    C_ = "C-"
    G_ = "G-"
    I_ = "I-"
    MA = "Ma"
    MO = "Mo"
    PD = "Pd"
    PI = "Pi"
    PK = "Pk"
    PP = "Pp"
    PQ = "Pq"
    PR = "Pr"
    PS = "Ps"
    PT = "Pt"
    PX = "Px"
    PY = "Py"
    R_ = "R-"
    S_ = "S-"
    X_ = "X-"
    
    UNKNOWN = "UNK"


class Case(Enum):
    """Morphological case values"""
    NOMINATIVE = "Nom"
    GENITIVE = "Gen"
    DATIVE = "Dat"
    ACCUSATIVE = "Acc"
    VOCATIVE = "Voc"
    ABLATIVE = "Abl"
    LOCATIVE = "Loc"
    INSTRUMENTAL = "Ins"
    UNKNOWN = "Unk"


class Number(Enum):
    """Morphological number values"""
    SINGULAR = "Sing"
    DUAL = "Dual"
    PLURAL = "Plur"
    UNKNOWN = "Unk"


class Gender(Enum):
    """Morphological gender values"""
    MASCULINE = "Masc"
    FEMININE = "Fem"
    NEUTER = "Neut"
    COMMON = "Com"
    UNKNOWN = "Unk"


class Person(Enum):
    """Morphological person values"""
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"
    UNKNOWN = "Unk"


class Tense(Enum):
    """Morphological tense values"""
    PRESENT = "Pres"
    IMPERFECT = "Imp"
    FUTURE = "Fut"
    AORIST = "Aor"
    PERFECT = "Perf"
    PLUPERFECT = "Plup"
    FUTURE_PERFECT = "FutPerf"
    UNKNOWN = "Unk"


class Mood(Enum):
    """Morphological mood values"""
    INDICATIVE = "Ind"
    SUBJUNCTIVE = "Sub"
    OPTATIVE = "Opt"
    IMPERATIVE = "Imp"
    INFINITIVE = "Inf"
    PARTICIPLE = "Part"
    GERUND = "Ger"
    GERUNDIVE = "Gdv"
    SUPINE = "Sup"
    UNKNOWN = "Unk"


class Voice(Enum):
    """Morphological voice values"""
    ACTIVE = "Act"
    MIDDLE = "Mid"
    PASSIVE = "Pass"
    MIDDLE_PASSIVE = "MidPass"
    DEPONENT = "Dep"
    UNKNOWN = "Unk"


class Degree(Enum):
    """Morphological degree values for adjectives/adverbs"""
    POSITIVE = "Pos"
    COMPARATIVE = "Cmp"
    SUPERLATIVE = "Sup"
    UNKNOWN = "Unk"


class DependencyRelation(Enum):
    """Universal Dependencies relations with PROIEL extensions"""
    ROOT = "root"
    ACL = "acl"
    ACL_RELCL = "acl:relcl"
    NSUBJ = "nsubj"
    NSUBJ_PASS = "nsubj:pass"
    OBJ = "obj"
    IOBJ = "iobj"
    CSUBJ = "csubj"
    CSUBJ_PASS = "csubj:pass"
    CCOMP = "ccomp"
    XCOMP = "xcomp"
    OBL = "obl"
    OBL_AGENT = "obl:agent"
    VOCATIVE = "vocative"
    EXPL = "expl"
    DISLOCATED = "dislocated"
    ADVCL = "advcl"
    ADVMOD = "advmod"
    DISCOURSE = "discourse"
    AUX = "aux"
    AUX_PASS = "aux:pass"
    COP = "cop"
    MARK = "mark"
    NMOD = "nmod"
    APPOS = "appos"
    NUMMOD = "nummod"
    AMOD = "amod"
    DET = "det"
    CLF = "clf"
    CASE = "case"
    CONJ = "conj"
    CC = "cc"
    FIXED = "fixed"
    FLAT = "flat"
    FLAT_NAME = "flat:name"
    COMPOUND = "compound"
    LIST = "list"
    PARATAXIS = "parataxis"
    ORPHAN = "orphan"
    GOESWITH = "goeswith"
    REPARANDUM = "reparandum"
    PUNCT = "punct"
    DEP = "dep"
    
    PRED = "PRED"
    SUB = "SUB"
    OBJ_PROIEL = "OBJ"
    OBL_PROIEL = "OBL"
    ADV = "ADV"
    ATR = "ATR"
    APOS = "APOS"
    AUX_PROIEL = "AUX"
    COMP = "COMP"
    EXPL_PROIEL = "EXPL"
    NARG = "NARG"
    NONSUB = "NONSUB"
    PARPRED = "PARPRED"
    PER = "PER"
    PART_PROIEL = "PART"
    XADV = "XADV"
    XOBJ = "XOBJ"
    XSUB = "XSUB"
    VOC = "VOC"
    AG = "AG"
    
    UNKNOWN = "unknown"


class SemanticRoleLabel(Enum):
    """Semantic role labels (PropBank/VerbNet style)"""
    ARG0 = "ARG0"
    ARG1 = "ARG1"
    ARG2 = "ARG2"
    ARG3 = "ARG3"
    ARG4 = "ARG4"
    ARG5 = "ARG5"
    ARGM_LOC = "ARGM-LOC"
    ARGM_TMP = "ARGM-TMP"
    ARGM_MNR = "ARGM-MNR"
    ARGM_CAU = "ARGM-CAU"
    ARGM_PRP = "ARGM-PRP"
    ARGM_DIR = "ARGM-DIR"
    ARGM_EXT = "ARGM-EXT"
    ARGM_ADV = "ARGM-ADV"
    ARGM_NEG = "ARGM-NEG"
    ARGM_MOD = "ARGM-MOD"
    ARGM_DIS = "ARGM-DIS"
    ARGM_PRD = "ARGM-PRD"
    ARGM_REC = "ARGM-REC"
    ARGM_GOL = "ARGM-GOL"
    ARGM_COM = "ARGM-COM"
    ARGM_LVB = "ARGM-LVB"
    V = "V"
    UNKNOWN = "UNK"


class NamedEntityType(Enum):
    """Named entity types for ancient texts"""
    PERSON = "PER"
    LOCATION = "LOC"
    ORGANIZATION = "ORG"
    GPE = "GPE"
    DEITY = "DEITY"
    MYTHOLOGICAL = "MYTH"
    ETHNIC = "ETH"
    WORK = "WORK"
    EVENT = "EVENT"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    NORP = "NORP"
    FACILITY = "FAC"
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    PERCENT = "PERCENT"
    MISC = "MISC"
    UNKNOWN = "UNK"


class InformationStatusType(Enum):
    """Information structure status (Givenness)"""
    NEW = "new"
    OLD = "old"
    ACCESSIBLE = "accessible"
    INFERABLE = "inferable"
    UNKNOWN = "unknown"


class TopicFocusType(Enum):
    """Topic-Focus articulation"""
    TOPIC = "topic"
    ABOUTNESS_TOPIC = "aboutness_topic"
    CONTRASTIVE_TOPIC = "contrastive_topic"
    FOCUS = "focus"
    INFORMATION_FOCUS = "information_focus"
    CONTRASTIVE_FOCUS = "contrastive_focus"
    BACKGROUND = "background"
    TAIL = "tail"
    UNKNOWN = "unknown"


@dataclass
class MorphologicalFeatures:
    """Complete morphological feature bundle"""
    pos: Optional[PartOfSpeech] = None
    case: Optional[Case] = None
    number: Optional[Number] = None
    gender: Optional[Gender] = None
    person: Optional[Person] = None
    tense: Optional[Tense] = None
    mood: Optional[Mood] = None
    voice: Optional[Voice] = None
    degree: Optional[Degree] = None
    
    definiteness: Optional[str] = None
    aspect: Optional[str] = None
    polarity: Optional[str] = None
    prontype: Optional[str] = None
    numtype: Optional[str] = None
    poss: Optional[str] = None
    reflex: Optional[str] = None
    foreign: Optional[str] = None
    abbr: Optional[str] = None
    typo: Optional[str] = None
    
    proiel_morph: Optional[str] = None
    
    additional_features: Dict[str, str] = field(default_factory=dict)
    
    def to_ud_string(self) -> str:
        """Convert to Universal Dependencies feature string"""
        features = []
        if self.case:
            features.append(f"Case={self.case.value}")
        if self.number:
            features.append(f"Number={self.number.value}")
        if self.gender:
            features.append(f"Gender={self.gender.value}")
        if self.person:
            features.append(f"Person={self.person.value}")
        if self.tense:
            features.append(f"Tense={self.tense.value}")
        if self.mood:
            features.append(f"Mood={self.mood.value}")
        if self.voice:
            features.append(f"Voice={self.voice.value}")
        if self.degree:
            features.append(f"Degree={self.degree.value}")
        if self.definiteness:
            features.append(f"Definite={self.definiteness}")
        if self.aspect:
            features.append(f"Aspect={self.aspect}")
        if self.polarity:
            features.append(f"Polarity={self.polarity}")
        if self.prontype:
            features.append(f"PronType={self.prontype}")
        if self.numtype:
            features.append(f"NumType={self.numtype}")
        if self.poss:
            features.append(f"Poss={self.poss}")
        if self.reflex:
            features.append(f"Reflex={self.reflex}")
        
        for key, value in self.additional_features.items():
            features.append(f"{key}={value}")
        
        return "|".join(sorted(features)) if features else "_"
    
    def to_proiel_string(self) -> str:
        """Convert to PROIEL morphology string"""
        if self.proiel_morph:
            return self.proiel_morph
        
        morph = ["-"] * 11
        
        if self.person:
            person_map = {"1": "1", "2": "2", "3": "3"}
            morph[0] = person_map.get(self.person.value, "-")
        
        if self.number:
            number_map = {"Sing": "s", "Dual": "d", "Plur": "p"}
            morph[1] = number_map.get(self.number.value, "-")
        
        if self.tense:
            tense_map = {
                "Pres": "p", "Imp": "i", "Fut": "f", 
                "Aor": "a", "Perf": "r", "Plup": "l", "FutPerf": "t"
            }
            morph[2] = tense_map.get(self.tense.value, "-")
        
        if self.mood:
            mood_map = {
                "Ind": "i", "Sub": "s", "Opt": "o", 
                "Imp": "m", "Inf": "n", "Part": "p",
                "Ger": "g", "Gdv": "d", "Sup": "u"
            }
            morph[3] = mood_map.get(self.mood.value, "-")
        
        if self.voice:
            voice_map = {"Act": "a", "Mid": "m", "Pass": "p", "MidPass": "e"}
            morph[4] = voice_map.get(self.voice.value, "-")
        
        if self.gender:
            gender_map = {"Masc": "m", "Fem": "f", "Neut": "n"}
            morph[5] = gender_map.get(self.gender.value, "-")
        
        if self.case:
            case_map = {
                "Nom": "n", "Gen": "g", "Dat": "d", 
                "Acc": "a", "Voc": "v", "Abl": "b",
                "Loc": "l", "Ins": "i"
            }
            morph[6] = case_map.get(self.case.value, "-")
        
        if self.degree:
            degree_map = {"Pos": "p", "Cmp": "c", "Sup": "s"}
            morph[7] = degree_map.get(self.degree.value, "-")
        
        return "".join(morph)
    
    @classmethod
    def from_ud_string(cls, ud_string: str) -> "MorphologicalFeatures":
        """Parse from Universal Dependencies feature string"""
        features = cls()
        if not ud_string or ud_string == "_":
            return features
        
        for feat in ud_string.split("|"):
            if "=" not in feat:
                continue
            key, value = feat.split("=", 1)
            
            if key == "Case":
                try:
                    features.case = Case(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Number":
                try:
                    features.number = Number(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Gender":
                try:
                    features.gender = Gender(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Person":
                try:
                    features.person = Person(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Tense":
                try:
                    features.tense = Tense(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Mood":
                try:
                    features.mood = Mood(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Voice":
                try:
                    features.voice = Voice(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Degree":
                try:
                    features.degree = Degree(value)
                except ValueError:
                    features.additional_features[key] = value
            elif key == "Definite":
                features.definiteness = value
            elif key == "Aspect":
                features.aspect = value
            elif key == "Polarity":
                features.polarity = value
            elif key == "PronType":
                features.prontype = value
            elif key == "NumType":
                features.numtype = value
            elif key == "Poss":
                features.poss = value
            elif key == "Reflex":
                features.reflex = value
            else:
                features.additional_features[key] = value
        
        return features
    
    @classmethod
    def from_proiel_string(cls, proiel_string: str) -> "MorphologicalFeatures":
        """Parse from PROIEL morphology string"""
        features = cls()
        features.proiel_morph = proiel_string
        
        if not proiel_string or len(proiel_string) < 7:
            return features
        
        person_map = {"1": Person.FIRST, "2": Person.SECOND, "3": Person.THIRD}
        if proiel_string[0] in person_map:
            features.person = person_map[proiel_string[0]]
        
        number_map = {"s": Number.SINGULAR, "d": Number.DUAL, "p": Number.PLURAL}
        if len(proiel_string) > 1 and proiel_string[1] in number_map:
            features.number = number_map[proiel_string[1]]
        
        tense_map = {
            "p": Tense.PRESENT, "i": Tense.IMPERFECT, "f": Tense.FUTURE,
            "a": Tense.AORIST, "r": Tense.PERFECT, "l": Tense.PLUPERFECT,
            "t": Tense.FUTURE_PERFECT
        }
        if len(proiel_string) > 2 and proiel_string[2] in tense_map:
            features.tense = tense_map[proiel_string[2]]
        
        mood_map = {
            "i": Mood.INDICATIVE, "s": Mood.SUBJUNCTIVE, "o": Mood.OPTATIVE,
            "m": Mood.IMPERATIVE, "n": Mood.INFINITIVE, "p": Mood.PARTICIPLE,
            "g": Mood.GERUND, "d": Mood.GERUNDIVE, "u": Mood.SUPINE
        }
        if len(proiel_string) > 3 and proiel_string[3] in mood_map:
            features.mood = mood_map[proiel_string[3]]
        
        voice_map = {
            "a": Voice.ACTIVE, "m": Voice.MIDDLE, 
            "p": Voice.PASSIVE, "e": Voice.MIDDLE_PASSIVE
        }
        if len(proiel_string) > 4 and proiel_string[4] in voice_map:
            features.voice = voice_map[proiel_string[4]]
        
        gender_map = {"m": Gender.MASCULINE, "f": Gender.FEMININE, "n": Gender.NEUTER}
        if len(proiel_string) > 5 and proiel_string[5] in gender_map:
            features.gender = gender_map[proiel_string[5]]
        
        case_map = {
            "n": Case.NOMINATIVE, "g": Case.GENITIVE, "d": Case.DATIVE,
            "a": Case.ACCUSATIVE, "v": Case.VOCATIVE, "b": Case.ABLATIVE,
            "l": Case.LOCATIVE, "i": Case.INSTRUMENTAL
        }
        if len(proiel_string) > 6 and proiel_string[6] in case_map:
            features.case = case_map[proiel_string[6]]
        
        degree_map = {"p": Degree.POSITIVE, "c": Degree.COMPARATIVE, "s": Degree.SUPERLATIVE}
        if len(proiel_string) > 7 and proiel_string[7] in degree_map:
            features.degree = degree_map[proiel_string[7]]
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        if self.pos:
            result["pos"] = self.pos.value
        if self.case:
            result["case"] = self.case.value
        if self.number:
            result["number"] = self.number.value
        if self.gender:
            result["gender"] = self.gender.value
        if self.person:
            result["person"] = self.person.value
        if self.tense:
            result["tense"] = self.tense.value
        if self.mood:
            result["mood"] = self.mood.value
        if self.voice:
            result["voice"] = self.voice.value
        if self.degree:
            result["degree"] = self.degree.value
        if self.definiteness:
            result["definiteness"] = self.definiteness
        if self.aspect:
            result["aspect"] = self.aspect
        if self.polarity:
            result["polarity"] = self.polarity
        if self.prontype:
            result["prontype"] = self.prontype
        if self.numtype:
            result["numtype"] = self.numtype
        if self.poss:
            result["poss"] = self.poss
        if self.reflex:
            result["reflex"] = self.reflex
        if self.proiel_morph:
            result["proiel_morph"] = self.proiel_morph
        if self.additional_features:
            result["additional"] = self.additional_features
        return result


@dataclass
class SyntacticRelation:
    """Syntactic dependency relation"""
    head_id: int
    relation: DependencyRelation
    enhanced_deps: List[Tuple[int, str]] = field(default_factory=list)
    secondary_edges: List[Tuple[int, str]] = field(default_factory=list)
    
    proiel_relation: Optional[str] = None
    proiel_slashes: List[str] = field(default_factory=list)
    
    def to_conllu_string(self) -> str:
        """Convert to CoNLL-U DEPREL string"""
        return self.relation.value
    
    def to_enhanced_string(self) -> str:
        """Convert to CoNLL-U enhanced dependencies string"""
        deps = [(self.head_id, self.relation.value)]
        deps.extend(self.enhanced_deps)
        return "|".join(f"{h}:{r}" for h, r in sorted(deps))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "head_id": self.head_id,
            "relation": self.relation.value
        }
        if self.enhanced_deps:
            result["enhanced_deps"] = self.enhanced_deps
        if self.secondary_edges:
            result["secondary_edges"] = self.secondary_edges
        if self.proiel_relation:
            result["proiel_relation"] = self.proiel_relation
        if self.proiel_slashes:
            result["proiel_slashes"] = self.proiel_slashes
        return result


@dataclass
class MorphologyAnnotation:
    """Morphological annotation for a token"""
    case: Optional[Case] = None
    number: Optional[str] = None
    gender: Optional[str] = None
    person: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    voice: Optional[str] = None
    ud_feats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        if self.case:
            result["case"] = self.case.value if hasattr(self.case, 'value') else str(self.case)
        if self.number:
            result["number"] = self.number
        if self.gender:
            result["gender"] = self.gender
        if self.person:
            result["person"] = self.person
        if self.tense:
            result["tense"] = self.tense
        if self.mood:
            result["mood"] = self.mood
        if self.voice:
            result["voice"] = self.voice
        if self.ud_feats:
            result["ud_feats"] = self.ud_feats
        return result


@dataclass
class SyntaxAnnotation:
    """Syntactic annotation for a token"""
    head: Optional[str] = None
    deprel: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        if self.head is not None:
            result["head"] = self.head
        if self.deprel:
            result["deprel"] = self.deprel
        return result


@dataclass
class SemanticRole:
    """Semantic role annotation for a token"""
    predicate_id: int
    role: SemanticRoleLabel
    span_start: int
    span_end: int
    confidence: float = 1.0
    source: str = "manual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "predicate_id": self.predicate_id,
            "role": self.role.value,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class NamedEntity:
    """Named entity annotation"""
    entity_type: NamedEntityType
    span_start: int
    span_end: int
    text: str
    normalized_form: Optional[str] = None
    wikidata_id: Optional[str] = None
    confidence: float = 1.0
    source: str = "manual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "entity_type": self.entity_type.value,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source
        }
        if self.normalized_form:
            result["normalized_form"] = self.normalized_form
        if self.wikidata_id:
            result["wikidata_id"] = self.wikidata_id
        return result


@dataclass
class InformationStructure:
    """Information structure annotation for a token/constituent"""
    info_status: InformationStatusType = InformationStatusType.UNKNOWN
    topic_focus: TopicFocusType = TopicFocusType.UNKNOWN
    contrast: bool = False
    emphasis: bool = False
    antecedent_id: Optional[int] = None
    confidence: float = 1.0
    source: str = "manual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "info_status": self.info_status.value,
            "topic_focus": self.topic_focus.value,
            "contrast": self.contrast,
            "emphasis": self.emphasis,
            "confidence": self.confidence,
            "source": self.source
        }
        if self.antecedent_id is not None:
            result["antecedent_id"] = self.antecedent_id
        return result


@dataclass
class Token:
    """
    Core token representation supporting both PROIEL and CoNLL-U formats.
    
    This is the fundamental unit of annotation in the platform.
    """
    id: int
    form: str
    lemma: Optional[str] = None
    
    morphology: MorphologicalFeatures = field(default_factory=MorphologicalFeatures)
    syntax: Optional[SyntacticRelation] = None
    
    semantic_roles: List[SemanticRole] = field(default_factory=list)
    named_entity: Optional[NamedEntity] = None
    info_structure: Optional[InformationStructure] = None
    
    misc: Dict[str, str] = field(default_factory=dict)
    
    proiel_id: Optional[str] = None
    proiel_presentation_before: Optional[str] = None
    proiel_presentation_after: Optional[str] = None
    proiel_empty_token_sort: Optional[str] = None
    proiel_citation_part: Optional[str] = None
    proiel_antecedent_id: Optional[str] = None
    proiel_information_status: Optional[str] = None
    proiel_contrast: Optional[str] = None
    
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    
    is_multiword: bool = False
    multiword_id: Optional[str] = None
    multiword_form: Optional[str] = None
    
    is_empty: bool = False
    empty_node_id: Optional[str] = None
    
    annotation_status: AnnotationStatus = AnnotationStatus.PENDING
    annotator: Optional[str] = None
    annotation_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.morphology is None:
            self.morphology = MorphologicalFeatures()
    
    @property
    def pos(self) -> Optional[PartOfSpeech]:
        """Get part of speech"""
        return self.morphology.pos if self.morphology else None
    
    @pos.setter
    def pos(self, value: PartOfSpeech):
        """Set part of speech"""
        if self.morphology is None:
            self.morphology = MorphologicalFeatures()
        self.morphology.pos = value
    
    @property
    def head(self) -> Optional[int]:
        """Get syntactic head ID"""
        return self.syntax.head_id if self.syntax else None
    
    @property
    def deprel(self) -> Optional[DependencyRelation]:
        """Get dependency relation"""
        return self.syntax.relation if self.syntax else None
    
    def to_conllu_line(self) -> str:
        """Convert to CoNLL-U format line"""
        token_id = self.multiword_id if self.multiword_id else str(self.id)
        if self.is_empty and self.empty_node_id:
            token_id = self.empty_node_id
        
        form = self.form if self.form else "_"
        lemma = self.lemma if self.lemma else "_"
        upos = self.morphology.pos.value if self.morphology and self.morphology.pos else "_"
        xpos = self.morphology.proiel_morph if self.morphology and self.morphology.proiel_morph else "_"
        feats = self.morphology.to_ud_string() if self.morphology else "_"
        head = str(self.syntax.head_id) if self.syntax else "_"
        deprel = self.syntax.relation.value if self.syntax else "_"
        deps = self.syntax.to_enhanced_string() if self.syntax and self.syntax.enhanced_deps else "_"
        
        misc_parts = []
        if self.span_start is not None:
            misc_parts.append(f"SpanStart={self.span_start}")
        if self.span_end is not None:
            misc_parts.append(f"SpanEnd={self.span_end}")
        for key, value in self.misc.items():
            misc_parts.append(f"{key}={value}")
        misc = "|".join(misc_parts) if misc_parts else "_"
        
        return f"{token_id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}"
    
    @classmethod
    def from_conllu_line(cls, line: str) -> "Token":
        """Parse from CoNLL-U format line"""
        parts = line.strip().split("\t")
        if len(parts) != 10:
            raise ValueError(f"Invalid CoNLL-U line: expected 10 fields, got {len(parts)}")
        
        token_id_str, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts
        
        is_multiword = "-" in token_id_str
        is_empty = "." in token_id_str
        
        if is_multiword:
            token_id = int(token_id_str.split("-")[0])
            multiword_id = token_id_str
        elif is_empty:
            token_id = int(token_id_str.split(".")[0])
            empty_node_id = token_id_str
        else:
            token_id = int(token_id_str)
            multiword_id = None
            empty_node_id = None
        
        morphology = MorphologicalFeatures.from_ud_string(feats)
        try:
            morphology.pos = PartOfSpeech(upos) if upos != "_" else None
        except ValueError:
            morphology.pos = PartOfSpeech.UNKNOWN
        
        if xpos != "_":
            morphology.proiel_morph = xpos
        
        syntax = None
        if head != "_" and deprel != "_":
            try:
                relation = DependencyRelation(deprel)
            except ValueError:
                relation = DependencyRelation.UNKNOWN
            
            enhanced_deps = []
            if deps != "_":
                for dep in deps.split("|"):
                    if ":" in dep:
                        h, r = dep.split(":", 1)
                        try:
                            enhanced_deps.append((int(h), r))
                        except ValueError:
                            pass
            
            syntax = SyntacticRelation(
                head_id=int(head),
                relation=relation,
                enhanced_deps=enhanced_deps
            )
        
        misc_dict = {}
        span_start = None
        span_end = None
        if misc != "_":
            for item in misc.split("|"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    if key == "SpanStart":
                        span_start = int(value)
                    elif key == "SpanEnd":
                        span_end = int(value)
                    else:
                        misc_dict[key] = value
        
        return cls(
            id=token_id,
            form=form if form != "_" else "",
            lemma=lemma if lemma != "_" else None,
            morphology=morphology,
            syntax=syntax,
            misc=misc_dict,
            span_start=span_start,
            span_end=span_end,
            is_multiword=is_multiword,
            multiword_id=multiword_id if is_multiword else None,
            is_empty=is_empty,
            empty_node_id=empty_node_id if is_empty else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "id": self.id,
            "form": self.form
        }
        if self.lemma:
            result["lemma"] = self.lemma
        if self.morphology:
            result["morphology"] = self.morphology.to_dict()
        if self.syntax:
            result["syntax"] = self.syntax.to_dict()
        if self.semantic_roles:
            result["semantic_roles"] = [sr.to_dict() for sr in self.semantic_roles]
        if self.named_entity:
            result["named_entity"] = self.named_entity.to_dict()
        if self.info_structure:
            result["info_structure"] = self.info_structure.to_dict()
        if self.misc:
            result["misc"] = self.misc
        if self.proiel_id:
            result["proiel_id"] = self.proiel_id
        if self.annotation_status != AnnotationStatus.PENDING:
            result["annotation_status"] = self.annotation_status.value
        return result
    
    def get_hash(self) -> str:
        """Get unique hash for this token"""
        content = f"{self.form}|{self.lemma}|{self.morphology.to_ud_string() if self.morphology else ''}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class CoNLLUToken(Token):
    """Extended token with CoNLL-U specific features"""
    deps_str: Optional[str] = None
    misc_str: Optional[str] = None
    
    def to_conllu_line(self) -> str:
        """Convert to CoNLL-U format with original strings preserved"""
        base_line = super().to_conllu_line()
        if self.deps_str or self.misc_str:
            parts = base_line.split("\t")
            if self.deps_str:
                parts[8] = self.deps_str
            if self.misc_str:
                parts[9] = self.misc_str
            return "\t".join(parts)
        return base_line


@dataclass
class TreeNode:
    """Generic tree node for syntactic trees"""
    id: int
    token: Token
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    
    def add_child(self, child: "TreeNode"):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def get_descendants(self) -> List["TreeNode"]:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def get_ancestors(self) -> List["TreeNode"]:
        """Get all ancestor nodes"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_depth(self) -> int:
        """Get depth in tree (root = 0)"""
        return len(self.get_ancestors())
    
    def get_span(self) -> Tuple[int, int]:
        """Get token span covered by this subtree"""
        all_ids = [self.id] + [d.id for d in self.get_descendants()]
        return min(all_ids), max(all_ids)
    
    def is_projective(self) -> bool:
        """Check if subtree is projective"""
        span_start, span_end = self.get_span()
        descendant_ids = {d.id for d in self.get_descendants()}
        descendant_ids.add(self.id)
        
        for i in range(span_start, span_end + 1):
            if i not in descendant_ids:
                return False
        return True


@dataclass
class PROIELNode(TreeNode):
    """PROIEL-specific tree node with additional attributes"""
    proiel_id: Optional[str] = None
    slashes: List[str] = field(default_factory=list)
    empty_token_sort: Optional[str] = None
    antecedent_id: Optional[str] = None
    information_status: Optional[str] = None
    contrast: Optional[str] = None
    
    def get_slash_targets(self) -> List[str]:
        """Get IDs of slash targets"""
        return self.slashes


@dataclass
class DependencyTree:
    """Complete dependency tree for a sentence"""
    root: Optional[TreeNode] = None
    nodes: Dict[int, TreeNode] = field(default_factory=dict)
    
    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "DependencyTree":
        """Build tree from list of tokens"""
        tree = cls()
        
        for token in tokens:
            node = TreeNode(id=token.id, token=token)
            tree.nodes[token.id] = node
        
        for token in tokens:
            if token.syntax:
                node = tree.nodes[token.id]
                head_id = token.syntax.head_id
                
                if head_id == 0:
                    tree.root = node
                elif head_id in tree.nodes:
                    tree.nodes[head_id].add_child(node)
        
        return tree
    
    def get_node(self, token_id: int) -> Optional[TreeNode]:
        """Get node by token ID"""
        return self.nodes.get(token_id)
    
    def is_projective(self) -> bool:
        """Check if entire tree is projective"""
        if not self.root:
            return True
        return self.root.is_projective()
    
    def get_depth(self) -> int:
        """Get maximum depth of tree"""
        if not self.nodes:
            return 0
        return max(node.get_depth() for node in self.nodes.values())
    
    def to_brackets(self) -> str:
        """Convert to bracketed string representation"""
        if not self.root:
            return "()"
        
        def node_to_brackets(node: TreeNode) -> str:
            children_str = " ".join(node_to_brackets(c) for c in sorted(node.children, key=lambda x: x.id))
            if children_str:
                return f"({node.token.form} {children_str})"
            return f"({node.token.form})"
        
        return node_to_brackets(self.root)


@dataclass
class Sentence:
    """
    Sentence representation with multi-layer annotations.
    
    Supports both PROIEL and CoNLL-U formats.
    """
    id: str
    tokens: List[Token] = field(default_factory=list)
    
    text: Optional[str] = None
    translation: Optional[str] = None
    
    document_id: Optional[str] = None
    sentence_index: int = 0
    
    proiel_id: Optional[str] = None
    proiel_status: Optional[str] = None
    proiel_presentation_before: Optional[str] = None
    proiel_presentation_after: Optional[str] = None
    
    sent_id: Optional[str] = None
    newdoc: Optional[str] = None
    newpar: Optional[str] = None
    
    metadata: Dict[str, str] = field(default_factory=dict)
    
    annotation_status: AnnotationStatus = AnnotationStatus.PENDING
    annotator: Optional[str] = None
    annotation_time: Optional[datetime] = None
    
    _tree: Optional[DependencyTree] = field(default=None, repr=False)
    
    def __post_init__(self):
        if not self.text and self.tokens:
            self.text = " ".join(t.form for t in self.tokens if t.form)
    
    @property
    def tree(self) -> DependencyTree:
        """Get or build dependency tree"""
        if self._tree is None:
            self._tree = DependencyTree.from_tokens(self.tokens)
        return self._tree
    
    def rebuild_tree(self):
        """Force rebuild of dependency tree"""
        self._tree = DependencyTree.from_tokens(self.tokens)
    
    def get_token(self, token_id: int) -> Optional[Token]:
        """Get token by ID"""
        for token in self.tokens:
            if token.id == token_id:
                return token
        return None
    
    def get_tokens_by_pos(self, pos: PartOfSpeech) -> List[Token]:
        """Get all tokens with given POS"""
        return [t for t in self.tokens if t.pos == pos]
    
    def get_verbs(self) -> List[Token]:
        """Get all verb tokens"""
        return [t for t in self.tokens if t.pos in (PartOfSpeech.VERB, PartOfSpeech.AUX, PartOfSpeech.V_)]
    
    def get_predicates(self) -> List[Token]:
        """Get predicate tokens (for SRL)"""
        predicates = []
        for token in self.tokens:
            if token.semantic_roles:
                for sr in token.semantic_roles:
                    if sr.role == SemanticRoleLabel.V:
                        predicates.append(token)
                        break
        if not predicates:
            predicates = self.get_verbs()
        return predicates
    
    def get_named_entities(self) -> List[NamedEntity]:
        """Get all named entities in sentence"""
        entities = []
        for token in self.tokens:
            if token.named_entity:
                entities.append(token.named_entity)
        return entities
    
    def get_dependents(self, head_id: int) -> List[Token]:
        """Get all direct dependents of a token"""
        return [t for t in self.tokens if t.syntax and t.syntax.head_id == head_id]
    
    def get_subtree(self, head_id: int) -> List[Token]:
        """Get all tokens in subtree rooted at head_id"""
        subtree = []
        to_process = [head_id]
        
        while to_process:
            current_id = to_process.pop(0)
            token = self.get_token(current_id)
            if token:
                subtree.append(token)
                dependents = self.get_dependents(current_id)
                to_process.extend(d.id for d in dependents)
        
        return sorted(subtree, key=lambda t: t.id)
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        lines = []
        
        if self.sent_id or self.id:
            lines.append(f"# sent_id = {self.sent_id or self.id}")
        if self.text:
            lines.append(f"# text = {self.text}")
        if self.translation:
            lines.append(f"# text_en = {self.translation}")
        if self.newdoc:
            lines.append(f"# newdoc id = {self.newdoc}")
        if self.newpar:
            lines.append(f"# newpar id = {self.newpar}")
        
        for key, value in self.metadata.items():
            if key not in ("sent_id", "text", "text_en", "newdoc", "newpar"):
                lines.append(f"# {key} = {value}")
        
        for token in self.tokens:
            lines.append(token.to_conllu_line())
        
        return "\n".join(lines)
    
    @classmethod
    def from_conllu(cls, conllu_text: str) -> "Sentence":
        """Parse from CoNLL-U format"""
        lines = conllu_text.strip().split("\n")
        
        metadata = {}
        tokens = []
        sent_id = None
        text = None
        translation = None
        newdoc = None
        newpar = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("#"):
                comment = line[1:].strip()
                if "=" in comment:
                    key, value = comment.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "sent_id":
                        sent_id = value
                    elif key == "text":
                        text = value
                    elif key == "text_en":
                        translation = value
                    elif key == "newdoc id":
                        newdoc = value
                    elif key == "newpar id":
                        newpar = value
                    else:
                        metadata[key] = value
            else:
                try:
                    token = Token.from_conllu_line(line)
                    tokens.append(token)
                except ValueError:
                    continue
        
        return cls(
            id=sent_id or f"s{hash(conllu_text) % 10000}",
            tokens=tokens,
            text=text,
            translation=translation,
            sent_id=sent_id,
            newdoc=newdoc,
            newpar=newpar,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "id": self.id,
            "tokens": [t.to_dict() for t in self.tokens]
        }
        if self.text:
            result["text"] = self.text
        if self.translation:
            result["translation"] = self.translation
        if self.document_id:
            result["document_id"] = self.document_id
        if self.sentence_index:
            result["sentence_index"] = self.sentence_index
        if self.metadata:
            result["metadata"] = self.metadata
        if self.annotation_status != AnnotationStatus.PENDING:
            result["annotation_status"] = self.annotation_status.value
        return result
    
    def get_hash(self) -> str:
        """Get unique hash for this sentence"""
        content = self.text or " ".join(t.form for t in self.tokens)
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class SourceMetadata:
    """Metadata about text source"""
    source_type: str
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    retrieval_date: Optional[datetime] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    original_format: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {"source_type": self.source_type}
        if self.source_url:
            result["source_url"] = self.source_url
        if self.source_id:
            result["source_id"] = self.source_id
        if self.retrieval_date:
            result["retrieval_date"] = self.retrieval_date.isoformat()
        if self.license:
            result["license"] = self.license
        if self.citation:
            result["citation"] = self.citation
        if self.original_format:
            result["original_format"] = self.original_format
        return result


@dataclass
class DiachronicStage:
    """Diachronic period/stage information"""
    period: Period
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    century: Optional[str] = None
    sub_period: Optional[str] = None
    
    @classmethod
    def from_year(cls, year: int, language: Language = Language.ANCIENT_GREEK) -> "DiachronicStage":
        """Determine diachronic stage from year"""
        if language in (Language.ANCIENT_GREEK, Language.CLASSICAL_GREEK, 
                       Language.HELLENISTIC_GREEK, Language.BYZANTINE_GREEK):
            if year < -1200:
                return cls(period=Period.MYCENAEAN, start_year=year)
            elif year < -700:
                return cls(period=Period.ARCHAIC, start_year=year)
            elif year < -323:
                return cls(period=Period.CLASSICAL, start_year=year)
            elif year < -31:
                return cls(period=Period.HELLENISTIC, start_year=year)
            elif year < 284:
                return cls(period=Period.ROMAN_IMPERIAL, start_year=year)
            elif year < 641:
                return cls(period=Period.LATE_ANTIQUE, start_year=year)
            elif year < 843:
                return cls(period=Period.EARLY_BYZANTINE, start_year=year)
            elif year < 1204:
                return cls(period=Period.MIDDLE_BYZANTINE, start_year=year)
            elif year < 1453:
                return cls(period=Period.LATE_BYZANTINE, start_year=year)
            elif year < 1830:
                return cls(period=Period.EARLY_MODERN, start_year=year)
            else:
                return cls(period=Period.MODERN, start_year=year)
        
        return cls(period=Period.UNKNOWN, start_year=year)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {"period": self.period.value}
        if self.start_year:
            result["start_year"] = self.start_year
        if self.end_year:
            result["end_year"] = self.end_year
        if self.century:
            result["century"] = self.century
        if self.sub_period:
            result["sub_period"] = self.sub_period
        return result


@dataclass
class Document:
    """
    Document representation containing multiple sentences.
    
    Represents a complete text with metadata and annotations.
    """
    id: str
    title: str
    sentences: List[Sentence] = field(default_factory=list)
    
    author: Optional[str] = None
    language: Language = Language.ANCIENT_GREEK
    period: Optional[Period] = None
    diachronic_stage: Optional[DiachronicStage] = None
    genre: Optional[Genre] = None
    
    source: Optional[SourceMetadata] = None
    
    proiel_id: Optional[str] = None
    proiel_source_id: Optional[str] = None
    
    date_composed: Optional[str] = None
    date_composed_start: Optional[int] = None
    date_composed_end: Optional[int] = None
    
    edition: Optional[str] = None
    editor: Optional[str] = None
    translator: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    annotation_status: AnnotationStatus = AnnotationStatus.PENDING
    annotation_progress: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        for i, sentence in enumerate(self.sentences):
            sentence.document_id = self.id
            if sentence.sentence_index == 0:
                sentence.sentence_index = i + 1
    
    @property
    def sentence_count(self) -> int:
        """Get number of sentences"""
        return len(self.sentences)
    
    @property
    def token_count(self) -> int:
        """Get total number of tokens"""
        return sum(len(s.tokens) for s in self.sentences)
    
    @property
    def word_count(self) -> int:
        """Get word count (excluding punctuation)"""
        count = 0
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.pos != PartOfSpeech.PUNCT:
                    count += 1
        return count
    
    def get_sentence(self, sentence_id: str) -> Optional[Sentence]:
        """Get sentence by ID"""
        for sentence in self.sentences:
            if sentence.id == sentence_id:
                return sentence
        return None
    
    def get_all_tokens(self) -> List[Token]:
        """Get all tokens in document"""
        tokens = []
        for sentence in self.sentences:
            tokens.extend(sentence.tokens)
        return tokens
    
    def get_vocabulary(self) -> Set[str]:
        """Get unique lemmas in document"""
        vocab = set()
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.lemma:
                    vocab.add(token.lemma)
        return vocab
    
    def get_pos_distribution(self) -> Dict[str, int]:
        """Get POS tag distribution"""
        distribution = defaultdict(int)
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.pos:
                    distribution[token.pos.value] += 1
        return dict(distribution)
    
    def to_conllu(self) -> str:
        """Convert entire document to CoNLL-U format"""
        lines = []
        
        lines.append(f"# newdoc id = {self.id}")
        if self.title:
            lines.append(f"# title = {self.title}")
        if self.author:
            lines.append(f"# author = {self.author}")
        if self.language:
            lines.append(f"# lang = {self.language.value}")
        
        for sentence in self.sentences:
            lines.append("")
            lines.append(sentence.to_conllu())
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "id": self.id,
            "title": self.title,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "language": self.language.value
        }
        if self.author:
            result["author"] = self.author
        if self.period:
            result["period"] = self.period.value
        if self.diachronic_stage:
            result["diachronic_stage"] = self.diachronic_stage.to_dict()
        if self.genre:
            result["genre"] = self.genre.value
        if self.source:
            result["source"] = self.source.to_dict()
        if self.date_composed:
            result["date_composed"] = self.date_composed
        if self.metadata:
            result["metadata"] = self.metadata
        result["annotation_status"] = self.annotation_status.value
        result["annotation_progress"] = self.annotation_progress
        return result
    
    def get_hash(self) -> str:
        """Get unique hash for this document"""
        content = f"{self.title}|{self.author}|{self.token_count}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class Corpus:
    """
    Corpus representation containing multiple documents.
    
    Top-level container for linguistic data.
    """
    id: str
    name: str
    documents: List[Document] = field(default_factory=list)
    
    description: Optional[str] = None
    language: Language = Language.ANCIENT_GREEK
    languages: List[Language] = field(default_factory=list)
    
    version: str = "1.0.0"
    license: Optional[str] = None
    citation: Optional[str] = None
    
    source_url: Optional[str] = None
    homepage: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.languages and self.language:
            self.languages = [self.language]
    
    @property
    def document_count(self) -> int:
        """Get number of documents"""
        return len(self.documents)
    
    @property
    def sentence_count(self) -> int:
        """Get total number of sentences"""
        return sum(d.sentence_count for d in self.documents)
    
    @property
    def token_count(self) -> int:
        """Get total number of tokens"""
        return sum(d.token_count for d in self.documents)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def add_document(self, document: Document):
        """Add document to corpus"""
        self.documents.append(document)
        self.updated_at = datetime.now()
    
    def get_documents_by_period(self, period: Period) -> List[Document]:
        """Get all documents from a specific period"""
        return [d for d in self.documents if d.period == period]
    
    def get_documents_by_genre(self, genre: Genre) -> List[Document]:
        """Get all documents of a specific genre"""
        return [d for d in self.documents if d.genre == genre]
    
    def get_documents_by_author(self, author: str) -> List[Document]:
        """Get all documents by a specific author"""
        return [d for d in self.documents if d.author == author]
    
    def get_all_sentences(self) -> Iterator[Sentence]:
        """Iterate over all sentences in corpus"""
        for document in self.documents:
            yield from document.sentences
    
    def get_all_tokens(self) -> Iterator[Token]:
        """Iterate over all tokens in corpus"""
        for document in self.documents:
            for sentence in document.sentences:
                yield from sentence.tokens
    
    def get_vocabulary(self) -> Set[str]:
        """Get unique lemmas in corpus"""
        vocab = set()
        for document in self.documents:
            vocab.update(document.get_vocabulary())
        return vocab
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        pos_dist = defaultdict(int)
        period_dist = defaultdict(int)
        genre_dist = defaultdict(int)
        
        for document in self.documents:
            if document.period:
                period_dist[document.period.value] += 1
            if document.genre:
                genre_dist[document.genre.value] += 1
            
            for pos, count in document.get_pos_distribution().items():
                pos_dist[pos] += count
        
        return {
            "document_count": self.document_count,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "vocabulary_size": len(self.get_vocabulary()),
            "pos_distribution": dict(pos_dist),
            "period_distribution": dict(period_dist),
            "genre_distribution": dict(genre_dist)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "language": self.language.value,
            "languages": [l.value for l in self.languages],
            "version": self.version,
            "license": self.license,
            "document_count": self.document_count,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "statistics": self.get_statistics()
        }


@dataclass
class ValencyFrame:
    """Valency frame for a verb sense"""
    frame_id: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    
    obligatory_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)
    
    semantic_roles: List[SemanticRoleLabel] = field(default_factory=list)
    
    syntactic_pattern: Optional[str] = None
    
    examples: List[str] = field(default_factory=list)
    
    def add_argument(self, role: str, case: Optional[Case] = None, 
                    preposition: Optional[str] = None, obligatory: bool = True):
        """Add an argument to the frame"""
        arg = {"role": role}
        if case:
            arg["case"] = case.value
        if preposition:
            arg["preposition"] = preposition
        arg["obligatory"] = obligatory
        
        self.arguments.append(arg)
        if obligatory:
            self.obligatory_args.append(role)
        else:
            self.optional_args.append(role)
    
    def to_pattern_string(self) -> str:
        """Convert to pattern string representation"""
        if self.syntactic_pattern:
            return self.syntactic_pattern
        
        parts = ["V"]
        for arg in self.arguments:
            role = arg.get("role", "?")
            case = arg.get("case", "")
            prep = arg.get("preposition", "")
            
            if prep:
                parts.append(f"[{prep}+{case}]" if case else f"[{prep}]")
            elif case:
                parts.append(f"[{case}]")
            else:
                parts.append(f"[{role}]")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "frame_id": self.frame_id,
            "arguments": self.arguments,
            "obligatory_args": self.obligatory_args,
            "optional_args": self.optional_args,
            "semantic_roles": [sr.value for sr in self.semantic_roles],
            "pattern": self.to_pattern_string(),
            "examples": self.examples
        }


@dataclass
class ValencyPattern:
    """Extracted valency pattern from corpus"""
    pattern_id: str
    verb_lemma: str
    frame: ValencyFrame
    
    language: Language = Language.ANCIENT_GREEK
    period: Optional[Period] = None
    
    frequency: int = 1
    relative_frequency: float = 0.0
    
    source_sentences: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    
    confidence: float = 1.0
    extraction_method: str = "automatic"
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_occurrence(self, sentence_id: str, document_id: str):
        """Record an occurrence of this pattern"""
        self.frequency += 1
        if sentence_id not in self.source_sentences:
            self.source_sentences.append(sentence_id)
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "pattern_id": self.pattern_id,
            "verb_lemma": self.verb_lemma,
            "frame": self.frame.to_dict(),
            "language": self.language.value,
            "period": self.period.value if self.period else None,
            "frequency": self.frequency,
            "relative_frequency": self.relative_frequency,
            "source_count": len(self.source_sentences),
            "confidence": self.confidence,
            "extraction_method": self.extraction_method
        }


@dataclass
class Lexeme:
    """Lexical entry for a word"""
    lemma: str
    language: Language
    
    pos: Optional[PartOfSpeech] = None
    
    senses: List["LemmaSense"] = field(default_factory=list)
    
    valency_patterns: List[ValencyPattern] = field(default_factory=list)
    
    etymology: Optional[str] = None
    cognates: List[str] = field(default_factory=list)
    
    frequency: int = 0
    frequency_by_period: Dict[str, int] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_sense(self, sense: "LemmaSense"):
        """Add a sense to this lexeme"""
        self.senses.append(sense)
    
    def add_valency_pattern(self, pattern: ValencyPattern):
        """Add a valency pattern"""
        self.valency_patterns.append(pattern)
    
    def get_valency_patterns_by_period(self, period: Period) -> List[ValencyPattern]:
        """Get valency patterns for a specific period"""
        return [p for p in self.valency_patterns if p.period == period]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "lemma": self.lemma,
            "language": self.language.value,
            "pos": self.pos.value if self.pos else None,
            "sense_count": len(self.senses),
            "valency_pattern_count": len(self.valency_patterns),
            "frequency": self.frequency,
            "frequency_by_period": self.frequency_by_period
        }


@dataclass
class LemmaSense:
    """Sense/meaning of a lemma"""
    sense_id: str
    definition: str
    
    gloss: Optional[str] = None
    
    examples: List[str] = field(default_factory=list)
    
    semantic_field: Optional[str] = None
    
    valency_frame: Optional[ValencyFrame] = None
    
    frequency: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "sense_id": self.sense_id,
            "definition": self.definition,
            "frequency": self.frequency
        }
        if self.gloss:
            result["gloss"] = self.gloss
        if self.examples:
            result["examples"] = self.examples
        if self.semantic_field:
            result["semantic_field"] = self.semantic_field
        if self.valency_frame:
            result["valency_frame"] = self.valency_frame.to_dict()
        return result


@dataclass
class AnnotationLayer:
    """Represents a layer of annotation"""
    layer_id: str
    layer_type: str
    
    annotator: Optional[str] = None
    annotation_time: Optional[datetime] = None
    
    status: AnnotationStatus = AnnotationStatus.PENDING
    
    confidence: float = 1.0
    
    source: str = "manual"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "annotator": self.annotator,
            "status": self.status.value,
            "confidence": self.confidence,
            "source": self.source
        }
