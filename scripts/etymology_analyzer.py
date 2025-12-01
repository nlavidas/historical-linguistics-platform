#!/usr/bin/env python3
"""
Etymology Analyzer for Greek
Proto-Indo-European reconstruction and cognate tracking

Features:
- PIE root reconstruction
- Cognate identification across IE languages
- Semantic development tracking
- Loanword detection
- Sound change rules
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
# PIE ROOTS DATABASE
# ============================================================================

PIE_ROOTS = {
    # Being/Existence
    "*h₁es-": {
        "meaning": "to be",
        "cognates": {
            "grc": ["εἰμί", "ἐστί", "ὤν"],
            "la": ["sum", "est", "esse"],
            "sa": ["asti", "sánti"],
            "got": ["ist", "sind"],
            "en": ["is", "am", "are"]
        }
    },
    "*bʰuH-": {
        "meaning": "to be, become, grow",
        "cognates": {
            "grc": ["φύω", "φύσις"],
            "la": ["fui", "futurus"],
            "sa": ["bhávati"],
            "en": ["be", "been"]
        }
    },
    
    # Motion
    "*h₁ey-": {
        "meaning": "to go",
        "cognates": {
            "grc": ["εἶμι", "ἰέναι"],
            "la": ["eo", "ire", "iter"],
            "sa": ["éti", "áyati"],
            "got": ["iddja"]
        }
    },
    "*gʷem-": {
        "meaning": "to come, go",
        "cognates": {
            "grc": ["βαίνω", "βάσις"],
            "la": ["venio", "ventus"],
            "sa": ["gámati"],
            "got": ["qiman"],
            "en": ["come"]
        }
    },
    "*steh₂-": {
        "meaning": "to stand",
        "cognates": {
            "grc": ["ἵστημι", "στάσις"],
            "la": ["sto", "stare", "status"],
            "sa": ["tíṣṭhati"],
            "got": ["standan"],
            "en": ["stand", "state"]
        }
    },
    
    # Perception
    "*weid-": {
        "meaning": "to see, know",
        "cognates": {
            "grc": ["εἶδον", "οἶδα", "ἰδέα"],
            "la": ["video", "videre"],
            "sa": ["véda", "vidyā"],
            "got": ["witan"],
            "en": ["wit", "wise", "vision"]
        }
    },
    "*ǵneh₃-": {
        "meaning": "to know",
        "cognates": {
            "grc": ["γιγνώσκω", "γνῶσις"],
            "la": ["nosco", "cognosco"],
            "sa": ["jānā́ti"],
            "got": ["kunnan"],
            "en": ["know", "can"]
        }
    },
    "*h₂ew-": {
        "meaning": "to perceive",
        "cognates": {
            "grc": ["αἰσθάνομαι", "ἀκούω"],
            "la": ["audio", "audire"],
            "sa": ["ávas"]
        }
    },
    
    # Speech
    "*wekʷ-": {
        "meaning": "to speak",
        "cognates": {
            "grc": ["ἔπος", "εἶπον"],
            "la": ["vox", "vocare"],
            "sa": ["vákti", "vā́c"],
            "en": ["voice"]
        }
    },
    "*bʰeh₂-": {
        "meaning": "to speak",
        "cognates": {
            "grc": ["φημί", "φάσις", "φωνή"],
            "la": ["fari", "fama"],
            "en": ["fame"]
        }
    },
    "*leg-": {
        "meaning": "to collect, speak",
        "cognates": {
            "grc": ["λέγω", "λόγος"],
            "la": ["lego", "legere"],
            "en": ["lecture", "legend"]
        }
    },
    
    # Giving/Taking
    "*deh₃-": {
        "meaning": "to give",
        "cognates": {
            "grc": ["δίδωμι", "δόσις", "δῶρον"],
            "la": ["do", "dare", "donum"],
            "sa": ["dádāti", "dā́nam"],
            "en": ["donate"]
        }
    },
    "*kap-": {
        "meaning": "to grasp",
        "cognates": {
            "grc": ["κάπτω"],
            "la": ["capio", "capere"],
            "got": ["hafjan"],
            "en": ["have", "capture"]
        }
    },
    
    # Thinking
    "*men-": {
        "meaning": "to think",
        "cognates": {
            "grc": ["μένος", "μαίνομαι", "μνήμη"],
            "la": ["mens", "memini"],
            "sa": ["mánas", "mányate"],
            "en": ["mind", "memory"]
        }
    },
    
    # Family
    "*ph₂tḗr": {
        "meaning": "father",
        "cognates": {
            "grc": ["πατήρ"],
            "la": ["pater"],
            "sa": ["pitár"],
            "got": ["fadar"],
            "en": ["father"]
        }
    },
    "*méh₂tēr": {
        "meaning": "mother",
        "cognates": {
            "grc": ["μήτηρ"],
            "la": ["mater"],
            "sa": ["mātár"],
            "en": ["mother"]
        }
    },
    "*bʰréh₂tēr": {
        "meaning": "brother",
        "cognates": {
            "grc": ["φράτηρ", "φράτρα"],
            "la": ["frater"],
            "sa": ["bhrā́tar"],
            "got": ["broþar"],
            "en": ["brother"]
        }
    },
    
    # Nature
    "*h₂ékʷeh₂": {
        "meaning": "water",
        "cognates": {
            "grc": ["ὕδωρ"],
            "la": ["aqua"],
            "got": ["wato"],
            "en": ["water"]
        }
    },
    "*péh₂wr̥": {
        "meaning": "fire",
        "cognates": {
            "grc": ["πῦρ"],
            "la": ["ignis"],
            "got": ["fon"],
            "en": ["fire"]
        }
    },
    
    # Body
    "*ḱerd-": {
        "meaning": "heart",
        "cognates": {
            "grc": ["καρδία", "κῆρ"],
            "la": ["cor", "cordis"],
            "got": ["hairto"],
            "en": ["heart"]
        }
    },
    "*h₃ekʷ-": {
        "meaning": "eye, see",
        "cognates": {
            "grc": ["ὄψ", "ὄμμα", "ὀφθαλμός"],
            "la": ["oculus"],
            "sa": ["ákṣi"],
            "en": ["eye"]
        }
    },
    
    # Numbers
    "*óynos": {
        "meaning": "one",
        "cognates": {
            "grc": ["οἶνος (ace)"],
            "la": ["unus"],
            "got": ["ains"],
            "en": ["one"]
        }
    },
    "*dwóh₁": {
        "meaning": "two",
        "cognates": {
            "grc": ["δύο"],
            "la": ["duo"],
            "sa": ["dvā́"],
            "got": ["twai"],
            "en": ["two"]
        }
    },
    "*tréyes": {
        "meaning": "three",
        "cognates": {
            "grc": ["τρεῖς"],
            "la": ["tres"],
            "sa": ["tráyas"],
            "got": ["þreis"],
            "en": ["three"]
        }
    }
}

# ============================================================================
# SOUND CHANGES
# ============================================================================

SOUND_CHANGES = {
    "pie_to_greek": [
        {"from": "*bʰ", "to": "φ", "position": "initial"},
        {"from": "*dʰ", "to": "θ", "position": "initial"},
        {"from": "*gʰ", "to": "χ", "position": "initial"},
        {"from": "*gʷ", "to": "β", "position": "initial"},
        {"from": "*kʷ", "to": "π/τ/κ", "position": "initial"},
        {"from": "*s", "to": "h/∅", "position": "initial"},
        {"from": "*y", "to": "ζ/h", "position": "initial"},
        {"from": "*w", "to": "∅/F", "position": "initial"},
        {"from": "*h₁", "to": "ε", "position": "any"},
        {"from": "*h₂", "to": "α", "position": "any"},
        {"from": "*h₃", "to": "ο", "position": "any"}
    ],
    "pie_to_latin": [
        {"from": "*bʰ", "to": "f/b", "position": "initial/medial"},
        {"from": "*dʰ", "to": "f/d", "position": "initial/medial"},
        {"from": "*gʰ", "to": "h/g", "position": "initial/medial"},
        {"from": "*gʷ", "to": "v/gu", "position": "initial"},
        {"from": "*kʷ", "to": "qu", "position": "initial"}
    ]
}

# ============================================================================
# LOANWORDS
# ============================================================================

GREEK_LOANWORDS = {
    "semitic": {
        "χιτών": {"source": "Semitic *kittān", "meaning": "tunic"},
        "σάκκος": {"source": "Semitic *śaq", "meaning": "sack"},
        "κάμηλος": {"source": "Semitic *gamal", "meaning": "camel"},
        "σάπφειρος": {"source": "Semitic *sappīr", "meaning": "sapphire"},
        "μύρρα": {"source": "Semitic *murr", "meaning": "myrrh"},
        "κύμινον": {"source": "Semitic *kammōn", "meaning": "cumin"}
    },
    "egyptian": {
        "ἔβενος": {"source": "Egyptian hbnj", "meaning": "ebony"},
        "βύβλος": {"source": "Egyptian pr-ḥꜥpj (Byblos)", "meaning": "papyrus"},
        "νίτρον": {"source": "Egyptian nṯrj", "meaning": "natron"}
    },
    "anatolian": {
        "τύραννος": {"source": "Lydian?", "meaning": "tyrant"},
        "λαβύρινθος": {"source": "Pre-Greek", "meaning": "labyrinth"},
        "θάλασσα": {"source": "Pre-Greek", "meaning": "sea"},
        "κυπάρισσος": {"source": "Pre-Greek", "meaning": "cypress"}
    },
    "persian": {
        "παράδεισος": {"source": "Old Persian *paridaida", "meaning": "paradise"},
        "σατράπης": {"source": "Old Persian xšaθrapāvan", "meaning": "satrap"},
        "ἄγγαρος": {"source": "Old Persian *hangāra", "meaning": "courier"}
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EtymologyEntry:
    """Etymology entry for a word"""
    lemma: str
    language: str
    pie_root: str = ""
    meaning: str = ""
    cognates: Dict[str, List[str]] = field(default_factory=dict)
    loanword_source: str = ""
    semantic_development: List[str] = field(default_factory=list)
    bibliography: List[str] = field(default_factory=list)
    notes: str = ""
    confidence: float = 1.0


# ============================================================================
# ETYMOLOGY ANALYZER
# ============================================================================

class EtymologyAnalyzer:
    """Analyze word etymologies"""
    
    def __init__(self):
        self.pie_roots = PIE_ROOTS
        self.loanwords = GREEK_LOANWORDS
        self.cache = {}
        
        # Build reverse index
        self._build_cognate_index()
    
    def _build_cognate_index(self):
        """Build index from forms to PIE roots"""
        self.cognate_index = defaultdict(list)
        
        for root, data in self.pie_roots.items():
            for lang, forms in data.get("cognates", {}).items():
                for form in forms:
                    self.cognate_index[form].append({
                        "root": root,
                        "meaning": data["meaning"],
                        "language": lang
                    })
    
    def analyze(self, lemma: str, language: str = "grc") -> EtymologyEntry:
        """Analyze etymology of a word"""
        cache_key = f"{language}:{lemma}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        entry = EtymologyEntry(lemma=lemma, language=language)
        
        # Check cognate index
        if lemma in self.cognate_index:
            matches = self.cognate_index[lemma]
            if matches:
                match = matches[0]
                entry.pie_root = match["root"]
                entry.meaning = match["meaning"]
                
                # Get all cognates for this root
                if match["root"] in self.pie_roots:
                    entry.cognates = self.pie_roots[match["root"]]["cognates"]
        
        # Check loanwords
        for source, words in self.loanwords.items():
            if lemma in words:
                entry.loanword_source = source
                entry.meaning = words[lemma]["meaning"]
                entry.notes = f"Borrowed from {words[lemma]['source']}"
                break
        
        self.cache[cache_key] = entry
        return entry
    
    def find_cognates(self, lemma: str, language: str = "grc") -> Dict[str, List[str]]:
        """Find cognates in other languages"""
        entry = self.analyze(lemma, language)
        return entry.cognates
    
    def get_pie_root(self, lemma: str) -> Optional[str]:
        """Get PIE root for a word"""
        if lemma in self.cognate_index:
            matches = self.cognate_index[lemma]
            if matches:
                return matches[0]["root"]
        return None
    
    def is_loanword(self, lemma: str) -> Tuple[bool, str]:
        """Check if word is a loanword"""
        for source, words in self.loanwords.items():
            if lemma in words:
                return True, source
        return False, ""
    
    def get_semantic_field(self, pie_root: str) -> List[str]:
        """Get words in same semantic field"""
        if pie_root not in self.pie_roots:
            return []
        
        return self.pie_roots[pie_root].get("cognates", {}).get("grc", [])
    
    def compare_languages(self, lemma: str, languages: List[str]) -> Dict:
        """Compare cognates across languages"""
        entry = self.analyze(lemma)
        
        result = {
            "lemma": lemma,
            "pie_root": entry.pie_root,
            "meaning": entry.meaning,
            "cognates": {}
        }
        
        for lang in languages:
            if lang in entry.cognates:
                result["cognates"][lang] = entry.cognates[lang]
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get etymology statistics"""
        stats = {
            "pie_roots": len(self.pie_roots),
            "total_cognates": 0,
            "by_language": defaultdict(int),
            "loanword_sources": {}
        }
        
        for root, data in self.pie_roots.items():
            for lang, forms in data.get("cognates", {}).items():
                stats["total_cognates"] += len(forms)
                stats["by_language"][lang] += len(forms)
        
        for source, words in self.loanwords.items():
            stats["loanword_sources"][source] = len(words)
        
        return stats
    
    def export_database(self, output_path: str):
        """Export etymology database"""
        data = {
            "pie_roots": self.pie_roots,
            "loanwords": self.loanwords,
            "sound_changes": SOUND_CHANGES
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported etymology database to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Etymology Analyzer")
    parser.add_argument('command', choices=['analyze', 'cognates', 'root', 'stats', 'export'],
                       help="Command to run")
    parser.add_argument('--lemma', '-l', help="Word to analyze")
    parser.add_argument('--language', default='grc', help="Language code")
    parser.add_argument('--output', '-o', help="Output file")
    
    args = parser.parse_args()
    
    analyzer = EtymologyAnalyzer()
    
    if args.command == 'analyze':
        if args.lemma:
            entry = analyzer.analyze(args.lemma, args.language)
            print(f"Lemma: {entry.lemma}")
            print(f"PIE Root: {entry.pie_root or 'Unknown'}")
            print(f"Meaning: {entry.meaning or 'Unknown'}")
            if entry.loanword_source:
                print(f"Loanword from: {entry.loanword_source}")
            if entry.cognates:
                print("Cognates:")
                for lang, forms in entry.cognates.items():
                    print(f"  {lang}: {', '.join(forms)}")
        else:
            print("Please provide --lemma")
    
    elif args.command == 'cognates':
        if args.lemma:
            cognates = analyzer.find_cognates(args.lemma, args.language)
            print(json.dumps(cognates, ensure_ascii=False, indent=2))
        else:
            print("Please provide --lemma")
    
    elif args.command == 'root':
        if args.lemma:
            root = analyzer.get_pie_root(args.lemma)
            print(f"PIE root for {args.lemma}: {root or 'Unknown'}")
        else:
            print("Please provide --lemma")
    
    elif args.command == 'stats':
        stats = analyzer.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif args.command == 'export':
        output = args.output or "etymology_database.json"
        analyzer.export_database(output)
        print(f"Exported to {output}")


if __name__ == "__main__":
    main()
