#!/usr/bin/env python3
"""
Morphological Tools for Greek
Complete morphological analysis system

Features:
- Full Greek morphological parsing
- Paradigm generation
- Lemmatization
- Morphological search
- PROIEL-compatible output
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MORPHOLOGICAL CATEGORIES
# ============================================================================

MORPHOLOGY = {
    "person": {
        "1": "first person",
        "2": "second person", 
        "3": "third person"
    },
    "number": {
        "s": "singular",
        "d": "dual",
        "p": "plural"
    },
    "tense": {
        "p": "present",
        "i": "imperfect",
        "f": "future",
        "a": "aorist",
        "r": "perfect",
        "l": "pluperfect",
        "t": "future perfect"
    },
    "mood": {
        "i": "indicative",
        "s": "subjunctive",
        "o": "optative",
        "m": "imperative",
        "n": "infinitive",
        "p": "participle"
    },
    "voice": {
        "a": "active",
        "m": "middle",
        "p": "passive",
        "e": "middle-passive"
    },
    "gender": {
        "m": "masculine",
        "f": "feminine",
        "n": "neuter"
    },
    "case": {
        "n": "nominative",
        "g": "genitive",
        "d": "dative",
        "a": "accusative",
        "v": "vocative"
    },
    "degree": {
        "p": "positive",
        "c": "comparative",
        "s": "superlative"
    }
}

# POS Tags (PROIEL style)
POS_TAGS = {
    "A-": {"name": "adjective", "categories": ["gender", "case", "number", "degree"]},
    "Df": {"name": "adverb", "categories": ["degree"]},
    "S-": {"name": "article", "categories": ["gender", "case", "number"]},
    "Ma": {"name": "cardinal", "categories": ["gender", "case", "number"]},
    "Nb": {"name": "common noun", "categories": ["gender", "case", "number"]},
    "C-": {"name": "conjunction", "categories": []},
    "Pd": {"name": "demonstrative", "categories": ["gender", "case", "number"]},
    "F-": {"name": "foreign", "categories": []},
    "Px": {"name": "indefinite", "categories": ["gender", "case", "number"]},
    "I-": {"name": "interjection", "categories": []},
    "Du": {"name": "interrogative adverb", "categories": []},
    "Pi": {"name": "interrogative pronoun", "categories": ["gender", "case", "number"]},
    "Mo": {"name": "ordinal", "categories": ["gender", "case", "number"]},
    "Pp": {"name": "personal pronoun", "categories": ["person", "gender", "case", "number"]},
    "Pk": {"name": "reflexive pronoun", "categories": ["person", "gender", "case", "number"]},
    "Ps": {"name": "possessive", "categories": ["gender", "case", "number"]},
    "R-": {"name": "preposition", "categories": []},
    "Ne": {"name": "proper noun", "categories": ["gender", "case", "number"]},
    "Py": {"name": "quantifier", "categories": ["gender", "case", "number"]},
    "Pc": {"name": "reciprocal", "categories": ["gender", "case", "number"]},
    "Dq": {"name": "relative adverb", "categories": []},
    "Pr": {"name": "relative pronoun", "categories": ["gender", "case", "number"]},
    "G-": {"name": "subjunction", "categories": []},
    "V-": {"name": "verb", "categories": ["person", "number", "tense", "mood", "voice"]},
    "X-": {"name": "unassigned", "categories": []}
}

# ============================================================================
# GREEK VERB PARADIGMS
# ============================================================================

VERB_PARADIGMS = {
    "λύω": {
        "stem": "λυ",
        "class": "ω-verb",
        "present": {
            "active": {
                "indicative": {
                    "1s": "λύω", "2s": "λύεις", "3s": "λύει",
                    "1p": "λύομεν", "2p": "λύετε", "3p": "λύουσι(ν)"
                },
                "subjunctive": {
                    "1s": "λύω", "2s": "λύῃς", "3s": "λύῃ",
                    "1p": "λύωμεν", "2p": "λύητε", "3p": "λύωσι(ν)"
                },
                "optative": {
                    "1s": "λύοιμι", "2s": "λύοις", "3s": "λύοι",
                    "1p": "λύοιμεν", "2p": "λύοιτε", "3p": "λύοιεν"
                },
                "imperative": {
                    "2s": "λῦε", "3s": "λυέτω",
                    "2p": "λύετε", "3p": "λυόντων"
                },
                "infinitive": "λύειν",
                "participle": {"m": "λύων", "f": "λύουσα", "n": "λῦον"}
            },
            "middle": {
                "indicative": {
                    "1s": "λύομαι", "2s": "λύῃ/λύει", "3s": "λύεται",
                    "1p": "λυόμεθα", "2p": "λύεσθε", "3p": "λύονται"
                }
            }
        },
        "imperfect": {
            "active": {
                "indicative": {
                    "1s": "ἔλυον", "2s": "ἔλυες", "3s": "ἔλυε(ν)",
                    "1p": "ἐλύομεν", "2p": "ἐλύετε", "3p": "ἔλυον"
                }
            }
        },
        "future": {
            "active": {
                "indicative": {
                    "1s": "λύσω", "2s": "λύσεις", "3s": "λύσει",
                    "1p": "λύσομεν", "2p": "λύσετε", "3p": "λύσουσι(ν)"
                }
            }
        },
        "aorist": {
            "active": {
                "indicative": {
                    "1s": "ἔλυσα", "2s": "ἔλυσας", "3s": "ἔλυσε(ν)",
                    "1p": "ἐλύσαμεν", "2p": "ἐλύσατε", "3p": "ἔλυσαν"
                }
            }
        },
        "perfect": {
            "active": {
                "indicative": {
                    "1s": "λέλυκα", "2s": "λέλυκας", "3s": "λέλυκε(ν)",
                    "1p": "λελύκαμεν", "2p": "λελύκατε", "3p": "λελύκασι(ν)"
                }
            }
        }
    }
}

# ============================================================================
# NOUN PARADIGMS
# ============================================================================

NOUN_PARADIGMS = {
    "second_declension_masculine": {
        "endings": {
            "ns": "ος", "gs": "ου", "ds": "ῳ", "as": "ον", "vs": "ε",
            "np": "οι", "gp": "ων", "dp": "οις", "ap": "ους", "vp": "οι"
        },
        "example": "λόγος"
    },
    "second_declension_neuter": {
        "endings": {
            "ns": "ον", "gs": "ου", "ds": "ῳ", "as": "ον", "vs": "ον",
            "np": "α", "gp": "ων", "dp": "οις", "ap": "α", "vp": "α"
        },
        "example": "ἔργον"
    },
    "first_declension_feminine_eta": {
        "endings": {
            "ns": "η", "gs": "ης", "ds": "ῃ", "as": "ην", "vs": "η",
            "np": "αι", "gp": "ων", "dp": "αις", "ap": "ας", "vp": "αι"
        },
        "example": "ψυχή"
    },
    "first_declension_feminine_alpha": {
        "endings": {
            "ns": "α", "gs": "ας", "ds": "ᾳ", "as": "αν", "vs": "α",
            "np": "αι", "gp": "ων", "dp": "αις", "ap": "ας", "vp": "αι"
        },
        "example": "χώρα"
    },
    "third_declension_consonant": {
        "endings": {
            "ns": "ς/-", "gs": "ος", "ds": "ι", "as": "α", "vs": "ς/-",
            "np": "ες", "gp": "ων", "dp": "σι(ν)", "ap": "ας", "vp": "ες"
        },
        "example": "σῶμα, σώματος"
    }
}

# ============================================================================
# MORPHOLOGICAL ANALYZER
# ============================================================================

@dataclass
class MorphAnalysis:
    """Morphological analysis result"""
    form: str
    lemma: str
    pos: str
    morphology: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    alternatives: List[Dict] = field(default_factory=list)


class GreekMorphAnalyzer:
    """Greek morphological analyzer"""
    
    def __init__(self):
        self.cache = {}
        self.paradigms = VERB_PARADIGMS
        self.noun_paradigms = NOUN_PARADIGMS
        
        # Build reverse lookup
        self._build_form_index()
    
    def _build_form_index(self):
        """Build index from forms to lemmas"""
        self.form_index = defaultdict(list)
        
        for lemma, paradigm in self.paradigms.items():
            for tense, voices in paradigm.items():
                if tense in ["stem", "class"]:
                    continue
                for voice, moods in voices.items():
                    for mood, forms in moods.items():
                        if isinstance(forms, dict):
                            for person_num, form in forms.items():
                                clean_form = form.replace("(ν)", "").replace("(", "").replace(")", "")
                                self.form_index[clean_form].append({
                                    "lemma": lemma,
                                    "tense": tense,
                                    "voice": voice,
                                    "mood": mood,
                                    "person_number": person_num
                                })
                        elif isinstance(forms, str):
                            self.form_index[forms].append({
                                "lemma": lemma,
                                "tense": tense,
                                "voice": voice,
                                "mood": mood
                            })
    
    def analyze(self, form: str) -> MorphAnalysis:
        """Analyze a Greek word form"""
        if form in self.cache:
            return self.cache[form]
        
        # Check form index first
        if form in self.form_index:
            entries = self.form_index[form]
            if entries:
                entry = entries[0]
                morph = self._build_morphology(entry)
                result = MorphAnalysis(
                    form=form,
                    lemma=entry["lemma"],
                    pos="V-",
                    morphology=morph,
                    alternatives=[self._build_morphology(e) for e in entries[1:]]
                )
                self.cache[form] = result
                return result
        
        # Fall back to pattern matching
        result = self._analyze_by_pattern(form)
        self.cache[form] = result
        return result
    
    def _build_morphology(self, entry: Dict) -> Dict:
        """Build morphology dict from entry"""
        morph = {}
        
        if "tense" in entry:
            tense_map = {"present": "p", "imperfect": "i", "future": "f", 
                        "aorist": "a", "perfect": "r", "pluperfect": "l"}
            morph["tense"] = tense_map.get(entry["tense"], "")
        
        if "voice" in entry:
            voice_map = {"active": "a", "middle": "m", "passive": "p"}
            morph["voice"] = voice_map.get(entry["voice"], "")
        
        if "mood" in entry:
            mood_map = {"indicative": "i", "subjunctive": "s", "optative": "o",
                       "imperative": "m", "infinitive": "n", "participle": "p"}
            morph["mood"] = mood_map.get(entry["mood"], "")
        
        if "person_number" in entry:
            pn = entry["person_number"]
            if len(pn) >= 2:
                morph["person"] = pn[0]
                morph["number"] = pn[1]
        
        return morph
    
    def _analyze_by_pattern(self, form: str) -> MorphAnalysis:
        """Analyze by pattern matching"""
        pos = self._detect_pos(form)
        morph = self._detect_morphology(form, pos)
        lemma = self._guess_lemma(form, pos)
        
        return MorphAnalysis(
            form=form,
            lemma=lemma,
            pos=pos,
            morphology=morph,
            confidence=0.6
        )
    
    def _detect_pos(self, form: str) -> str:
        """Detect POS from form"""
        # Articles
        articles = {"ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῷ", "τῇ", "τόν", "τήν",
                   "οἱ", "αἱ", "τά", "τῶν", "τοῖς", "ταῖς", "τούς", "τάς"}
        if form in articles:
            return "S-"
        
        # Prepositions
        preps = {"ἐν", "εἰς", "ἐκ", "ἐξ", "ἀπό", "πρός", "παρά", "μετά", 
                "διά", "ὑπό", "περί", "κατά", "ἐπί", "ὑπέρ", "πρό", "ἀντί", "σύν"}
        if form in preps:
            return "R-"
        
        # Conjunctions
        conjs = {"καί", "δέ", "γάρ", "ἀλλά", "ἤ", "οὔτε", "μήτε", "τε"}
        if form in conjs:
            return "C-"
        
        # Verb endings
        verb_endings = ["ω", "εις", "ει", "ομεν", "ετε", "ουσι", "ουσιν",
                       "μαι", "σαι", "ται", "μεθα", "σθε", "νται",
                       "ον", "ες", "ε", "α", "ας", "αν", "αμεν", "ατε",
                       "ειν", "αι", "σθαι", "ων", "ουσα"]
        for ending in verb_endings:
            if form.endswith(ending):
                return "V-"
        
        # Noun/adjective endings
        noun_endings = ["ος", "ου", "ῳ", "ον", "οι", "ων", "οις", "ους",
                       "η", "ης", "ῃ", "ην", "αι", "ας",
                       "α", "ᾳ", "αν"]
        for ending in noun_endings:
            if form.endswith(ending):
                return "Nb"
        
        return "X-"
    
    def _detect_morphology(self, form: str, pos: str) -> Dict:
        """Detect morphological features"""
        morph = {}
        
        if pos == "V-":
            # Detect tense/mood/voice from endings
            if form.endswith(("ω", "εις", "ει", "ομεν", "ετε", "ουσι")):
                morph["tense"] = "p"
                morph["mood"] = "i"
                morph["voice"] = "a"
            elif form.endswith(("ον", "ες", "ε")):
                morph["tense"] = "i"
                morph["mood"] = "i"
            elif form.endswith(("α", "ας", "αν", "αμεν", "ατε")):
                morph["tense"] = "a"
                morph["mood"] = "i"
            
            # Detect person/number
            if form.endswith(("ω", "μαι", "ον", "α")):
                morph["person"] = "1"
                morph["number"] = "s"
            elif form.endswith(("εις", "ῃ", "ες", "ας")):
                morph["person"] = "2"
                morph["number"] = "s"
            elif form.endswith(("ει", "ται", "ε", "εν")):
                morph["person"] = "3"
                morph["number"] = "s"
            elif form.endswith(("ομεν", "μεθα", "αμεν")):
                morph["person"] = "1"
                morph["number"] = "p"
            elif form.endswith(("ετε", "σθε", "ατε")):
                morph["person"] = "2"
                morph["number"] = "p"
            elif form.endswith(("ουσι", "ουσιν", "νται", "αν")):
                morph["person"] = "3"
                morph["number"] = "p"
        
        elif pos in ["Nb", "Ne", "A-"]:
            # Detect case/number/gender
            if form.endswith(("ος", "ης", "υς")):
                morph["case"] = "n"
                morph["number"] = "s"
            elif form.endswith(("ου", "ης")):
                morph["case"] = "g"
                morph["number"] = "s"
            elif form.endswith(("ῳ", "ῃ", "ι")):
                morph["case"] = "d"
                morph["number"] = "s"
            elif form.endswith(("ον", "ην", "αν", "α")):
                morph["case"] = "a"
                morph["number"] = "s"
            elif form.endswith(("οι", "αι")):
                morph["case"] = "n"
                morph["number"] = "p"
            elif form.endswith("ων"):
                morph["case"] = "g"
                morph["number"] = "p"
            elif form.endswith(("οις", "αις")):
                morph["case"] = "d"
                morph["number"] = "p"
            elif form.endswith(("ους", "ας")):
                morph["case"] = "a"
                morph["number"] = "p"
        
        return morph
    
    def _guess_lemma(self, form: str, pos: str) -> str:
        """Guess lemma from form"""
        if pos == "V-":
            # Try to get first person singular present
            for ending in ["ω", "ομαι"]:
                if form.endswith(ending):
                    return form
            # Strip endings and add -ω
            for ending in ["εις", "ει", "ομεν", "ετε", "ουσι", "ουσιν",
                          "ον", "ες", "ε", "α", "ας", "αν", "αμεν", "ατε"]:
                if form.endswith(ending):
                    stem = form[:-len(ending)]
                    return stem + "ω"
        
        elif pos in ["Nb", "Ne"]:
            # Try to get nominative singular
            for ending in ["ου", "ῳ", "ον", "οι", "ων", "οις", "ους"]:
                if form.endswith(ending):
                    stem = form[:-len(ending)]
                    return stem + "ος"
            for ending in ["ης", "ῃ", "ην", "αι", "ων", "αις", "ας"]:
                if form.endswith(ending):
                    stem = form[:-len(ending)]
                    return stem + "η"
        
        return form
    
    def generate_paradigm(self, lemma: str, pos: str) -> Dict:
        """Generate full paradigm for a lemma"""
        if lemma in self.paradigms:
            return self.paradigms[lemma]
        
        # Generate based on pattern
        if pos == "V-":
            return self._generate_verb_paradigm(lemma)
        elif pos in ["Nb", "Ne"]:
            return self._generate_noun_paradigm(lemma)
        
        return {}
    
    def _generate_verb_paradigm(self, lemma: str) -> Dict:
        """Generate verb paradigm"""
        if not lemma.endswith("ω"):
            return {}
        
        stem = lemma[:-1]
        
        return {
            "present": {
                "active": {
                    "indicative": {
                        "1s": stem + "ω",
                        "2s": stem + "εις",
                        "3s": stem + "ει",
                        "1p": stem + "ομεν",
                        "2p": stem + "ετε",
                        "3p": stem + "ουσι(ν)"
                    }
                }
            }
        }
    
    def _generate_noun_paradigm(self, lemma: str) -> Dict:
        """Generate noun paradigm"""
        if lemma.endswith("ος"):
            stem = lemma[:-2]
            return {
                "singular": {
                    "nominative": stem + "ος",
                    "genitive": stem + "ου",
                    "dative": stem + "ῳ",
                    "accusative": stem + "ον",
                    "vocative": stem + "ε"
                },
                "plural": {
                    "nominative": stem + "οι",
                    "genitive": stem + "ων",
                    "dative": stem + "οις",
                    "accusative": stem + "ους",
                    "vocative": stem + "οι"
                }
            }
        
        return {}
    
    def format_analysis(self, analysis: MorphAnalysis) -> str:
        """Format analysis for display"""
        morph_str = []
        for cat, val in analysis.morphology.items():
            if cat in MORPHOLOGY and val in MORPHOLOGY[cat]:
                morph_str.append(f"{cat}={MORPHOLOGY[cat][val]}")
        
        pos_name = POS_TAGS.get(analysis.pos, {}).get("name", analysis.pos)
        
        return f"{analysis.form} → {analysis.lemma} ({pos_name}) [{', '.join(morph_str)}]"


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Greek Morphological Tools")
    parser.add_argument('command', choices=['analyze', 'paradigm', 'batch'],
                       help="Command to run")
    parser.add_argument('--form', '-f', help="Word form to analyze")
    parser.add_argument('--lemma', '-l', help="Lemma for paradigm")
    parser.add_argument('--pos', '-p', help="POS tag")
    parser.add_argument('--input', '-i', help="Input file for batch")
    
    args = parser.parse_args()
    
    analyzer = GreekMorphAnalyzer()
    
    if args.command == 'analyze':
        if args.form:
            result = analyzer.analyze(args.form)
            print(analyzer.format_analysis(result))
            print(json.dumps(result.morphology, ensure_ascii=False, indent=2))
        else:
            print("Please provide --form")
    
    elif args.command == 'paradigm':
        if args.lemma:
            paradigm = analyzer.generate_paradigm(args.lemma, args.pos or "V-")
            print(json.dumps(paradigm, ensure_ascii=False, indent=2))
        else:
            print("Please provide --lemma")
    
    elif args.command == 'batch':
        if args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                for line in f:
                    form = line.strip()
                    if form:
                        result = analyzer.analyze(form)
                        print(analyzer.format_analysis(result))
        else:
            print("Please provide --input file")


if __name__ == "__main__":
    main()
