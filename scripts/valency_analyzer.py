#!/usr/bin/env python3
"""
Valency Analyzer - PROIEL-style Valency Analysis
Comprehensive verbal argument structure analysis for Greek

Based on:
- PROIEL Treebank methodology
- Vonatus valency lexicon approach
- University of Athens linguistic research
"""

import os
import sys
import re
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# VALENCY CONFIGURATION
# ============================================================================

VALENCY_CONFIG = {
    # Case patterns
    "case_labels": {
        "n": "Nominative",
        "g": "Genitive", 
        "d": "Dative",
        "a": "Accusative",
        "v": "Vocative",
        "i": "Instrumental",
        "l": "Locative"
    },
    
    # Semantic roles (PROIEL style)
    "semantic_roles": {
        "A": {"name": "Agent", "description": "Volitional causer of event"},
        "P": {"name": "Patient", "description": "Entity undergoing change"},
        "T": {"name": "Theme", "description": "Entity in motion or located"},
        "E": {"name": "Experiencer", "description": "Entity experiencing state"},
        "R": {"name": "Recipient", "description": "Entity receiving something"},
        "B": {"name": "Beneficiary", "description": "Entity benefiting from action"},
        "G": {"name": "Goal", "description": "Endpoint of motion"},
        "S": {"name": "Source", "description": "Starting point of motion"},
        "L": {"name": "Location", "description": "Place where event occurs"},
        "I": {"name": "Instrument", "description": "Means by which action performed"},
        "C": {"name": "Cause", "description": "Non-volitional causer"},
        "M": {"name": "Manner", "description": "Way action is performed"},
        "X": {"name": "Extent", "description": "Measure or degree"}
    },
    
    # Dependency relations for arguments
    "argument_relations": {
        "sub": "Subject",
        "obj": "Object",
        "obl": "Oblique",
        "ag": "Agent",
        "comp": "Complement",
        "xobj": "External object",
        "xsub": "External subject",
        "atr": "Attribute"
    },
    
    # Valency classes
    "valency_classes": {
        "0": {"name": "Avalent", "pattern": "∅", "example": "ὕει (it rains)"},
        "1": {"name": "Monovalent", "pattern": "NOM", "example": "βαίνω (I go)"},
        "2a": {"name": "Bivalent-ACC", "pattern": "NOM+ACC", "example": "ὁράω (I see)"},
        "2g": {"name": "Bivalent-GEN", "pattern": "NOM+GEN", "example": "ἀκούω (I hear)"},
        "2d": {"name": "Bivalent-DAT", "pattern": "NOM+DAT", "example": "πείθομαι (I obey)"},
        "3": {"name": "Trivalent", "pattern": "NOM+ACC+DAT", "example": "δίδωμι (I give)"},
        "3g": {"name": "Trivalent-GEN", "pattern": "NOM+ACC+GEN", "example": "πληρόω (I fill)"},
        "2x2": {"name": "Double-ACC", "pattern": "NOM+ACC+ACC", "example": "διδάσκω (I teach)"},
        "cop": {"name": "Copular", "pattern": "NOM+NOM", "example": "εἰμί (I am)"}
    }
}

# ============================================================================
# GREEK VERB LEXICON
# ============================================================================

GREEK_VERB_LEXICON = {
    # Movement verbs
    "βαίνω": {"class": "1", "pattern": "NOM", "gloss": "go, walk", "semantic": "motion"},
    "ἔρχομαι": {"class": "1", "pattern": "NOM", "gloss": "come, go", "semantic": "motion"},
    "πορεύομαι": {"class": "1", "pattern": "NOM", "gloss": "travel, journey", "semantic": "motion"},
    "φεύγω": {"class": "1", "pattern": "NOM", "gloss": "flee", "semantic": "motion"},
    "τρέχω": {"class": "1", "pattern": "NOM", "gloss": "run", "semantic": "motion"},
    
    # Perception verbs
    "ὁράω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "see", "semantic": "perception"},
    "βλέπω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "look at", "semantic": "perception"},
    "ἀκούω": {"class": "2g", "pattern": "NOM+GEN", "gloss": "hear", "semantic": "perception"},
    "αἰσθάνομαι": {"class": "2g", "pattern": "NOM+GEN", "gloss": "perceive", "semantic": "perception"},
    
    # Communication verbs
    "λέγω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "say, speak", "semantic": "communication"},
    "φημί": {"class": "2a", "pattern": "NOM+ACC", "gloss": "say, affirm", "semantic": "communication"},
    "ἐρωτάω": {"class": "2x2", "pattern": "NOM+ACC+ACC", "gloss": "ask", "semantic": "communication"},
    "κελεύω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "order, command", "semantic": "communication"},
    
    # Transfer verbs
    "δίδωμι": {"class": "3", "pattern": "NOM+ACC+DAT", "gloss": "give", "semantic": "transfer"},
    "λαμβάνω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "take, receive", "semantic": "transfer"},
    "φέρω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "carry, bring", "semantic": "transfer"},
    "πέμπω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "send", "semantic": "transfer"},
    
    # Cognitive verbs
    "γιγνώσκω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "know, recognize", "semantic": "cognition"},
    "οἶδα": {"class": "2a", "pattern": "NOM+ACC", "gloss": "know", "semantic": "cognition"},
    "νομίζω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "think, believe", "semantic": "cognition"},
    "δοκέω": {"class": "2d", "pattern": "NOM+DAT", "gloss": "seem", "semantic": "cognition"},
    "μανθάνω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "learn", "semantic": "cognition"},
    
    # Causative verbs
    "ποιέω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "make, do", "semantic": "causation"},
    "τίθημι": {"class": "2a", "pattern": "NOM+ACC", "gloss": "put, place", "semantic": "causation"},
    "ἵστημι": {"class": "2a", "pattern": "NOM+ACC", "gloss": "set up, stand", "semantic": "causation"},
    
    # Emotional verbs
    "φιλέω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "love", "semantic": "emotion"},
    "μισέω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "hate", "semantic": "emotion"},
    "φοβέομαι": {"class": "2a", "pattern": "NOM+ACC", "gloss": "fear", "semantic": "emotion"},
    "θαυμάζω": {"class": "2a", "pattern": "NOM+ACC", "gloss": "wonder at", "semantic": "emotion"},
    
    # Copular verbs
    "εἰμί": {"class": "cop", "pattern": "NOM+NOM", "gloss": "be", "semantic": "state"},
    "γίγνομαι": {"class": "cop", "pattern": "NOM+NOM", "gloss": "become", "semantic": "change"},
    
    # Genitive-taking verbs
    "ἅπτομαι": {"class": "2g", "pattern": "NOM+GEN", "gloss": "touch", "semantic": "contact"},
    "μέμνημαι": {"class": "2g", "pattern": "NOM+GEN", "gloss": "remember", "semantic": "cognition"},
    "ἐπιμελέομαι": {"class": "2g", "pattern": "NOM+GEN", "gloss": "care for", "semantic": "attention"},
    "ἄρχω": {"class": "2g", "pattern": "NOM+GEN", "gloss": "rule, begin", "semantic": "control"},
    "κρατέω": {"class": "2g", "pattern": "NOM+GEN", "gloss": "rule, control", "semantic": "control"},
    
    # Dative-taking verbs
    "πείθομαι": {"class": "2d", "pattern": "NOM+DAT", "gloss": "obey", "semantic": "social"},
    "ἕπομαι": {"class": "2d", "pattern": "NOM+DAT", "gloss": "follow", "semantic": "motion"},
    "βοηθέω": {"class": "2d", "pattern": "NOM+DAT", "gloss": "help", "semantic": "assistance"},
    "πιστεύω": {"class": "2d", "pattern": "NOM+DAT", "gloss": "trust, believe", "semantic": "cognition"},
    
    # Double accusative verbs
    "διδάσκω": {"class": "2x2", "pattern": "NOM+ACC+ACC", "gloss": "teach", "semantic": "transfer"},
    "αἰτέω": {"class": "2x2", "pattern": "NOM+ACC+ACC", "gloss": "ask for", "semantic": "request"},
    
    # Accusative + Genitive verbs
    "πληρόω": {"class": "3g", "pattern": "NOM+ACC+GEN", "gloss": "fill", "semantic": "filling"},
    "ἀξιόω": {"class": "3g", "pattern": "NOM+ACC+GEN", "gloss": "deem worthy", "semantic": "evaluation"}
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValencyFrame:
    """Valency frame for a verb"""
    lemma: str
    pattern: str
    valency_class: str
    arguments: List[Dict] = field(default_factory=list)
    frequency: int = 1
    examples: List[str] = field(default_factory=list)
    period: str = ""
    semantic_class: str = ""
    alternations: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ArgumentSlot:
    """Argument slot in valency frame"""
    case: str
    role: str
    obligatory: bool = True
    preposition: str = ""
    semantic_restrictions: List[str] = field(default_factory=list)


# ============================================================================
# VALENCY ANALYZER
# ============================================================================

class ValencyAnalyzer:
    """Analyze verbal valency patterns"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self.lexicon = GREEK_VERB_LEXICON.copy()
        self.extracted_frames = defaultdict(list)
    
    def analyze_sentence(self, tokens: List[Dict]) -> List[ValencyFrame]:
        """Analyze valency in a sentence"""
        frames = []
        
        for token in tokens:
            if token.get("pos", "").startswith("V"):
                frame = self._extract_frame(token, tokens)
                if frame:
                    frames.append(frame)
        
        return frames
    
    def _extract_frame(self, verb: Dict, tokens: List[Dict]) -> Optional[ValencyFrame]:
        """Extract valency frame for a verb"""
        verb_id = verb.get("id", 0)
        lemma = verb.get("lemma", verb.get("form", ""))
        
        # Find dependents
        arguments = []
        for token in tokens:
            if token.get("head") == verb_id:
                rel = token.get("relation", token.get("deprel", ""))
                if rel in ["sub", "obj", "obl", "ag", "comp", "nsubj", "obj", "iobj"]:
                    arg = {
                        "form": token.get("form", ""),
                        "lemma": token.get("lemma", ""),
                        "case": token.get("morphology", {}).get("case", ""),
                        "relation": rel,
                        "role": self._infer_role(rel, token)
                    }
                    arguments.append(arg)
        
        # Determine pattern
        cases = [a["case"] for a in arguments if a["case"]]
        pattern = self._determine_pattern(cases)
        valency_class = self._determine_class(pattern)
        
        # Check lexicon
        if lemma in self.lexicon:
            lex_entry = self.lexicon[lemma]
            semantic_class = lex_entry.get("semantic", "")
        else:
            semantic_class = ""
        
        return ValencyFrame(
            lemma=lemma,
            pattern=pattern,
            valency_class=valency_class,
            arguments=arguments,
            semantic_class=semantic_class
        )
    
    def _determine_pattern(self, cases: List[str]) -> str:
        """Determine valency pattern from cases"""
        if not cases:
            return "∅"
        
        case_map = {"n": "NOM", "g": "GEN", "d": "DAT", "a": "ACC", "v": "VOC"}
        mapped = [case_map.get(c, c.upper()) for c in cases]
        
        # Sort in standard order
        order = ["NOM", "ACC", "GEN", "DAT", "VOC"]
        sorted_cases = sorted(set(mapped), key=lambda x: order.index(x) if x in order else 99)
        
        return "+".join(sorted_cases)
    
    def _determine_class(self, pattern: str) -> str:
        """Determine valency class from pattern"""
        for cls, info in VALENCY_CONFIG["valency_classes"].items():
            if info["pattern"] == pattern:
                return cls
        return "?"
    
    def _infer_role(self, relation: str, token: Dict) -> str:
        """Infer semantic role from relation"""
        role_map = {
            "sub": "A",
            "nsubj": "A",
            "obj": "P",
            "iobj": "R",
            "obl": "L",
            "ag": "A"
        }
        return role_map.get(relation, "")
    
    def get_lexicon_entry(self, lemma: str) -> Optional[Dict]:
        """Get lexicon entry for verb"""
        return self.lexicon.get(lemma)
    
    def add_to_lexicon(self, frame: ValencyFrame):
        """Add frame to lexicon"""
        if frame.lemma not in self.lexicon:
            self.lexicon[frame.lemma] = {
                "class": frame.valency_class,
                "pattern": frame.pattern,
                "semantic": frame.semantic_class
            }
        
        self.extracted_frames[frame.lemma].append(frame)
    
    def get_statistics(self) -> Dict:
        """Get valency statistics"""
        stats = {
            "lexicon_size": len(self.lexicon),
            "extracted_verbs": len(self.extracted_frames),
            "by_class": Counter(),
            "by_pattern": Counter(),
            "by_semantic": Counter()
        }
        
        for lemma, info in self.lexicon.items():
            stats["by_class"][info.get("class", "?")] += 1
            stats["by_pattern"][info.get("pattern", "?")] += 1
            stats["by_semantic"][info.get("semantic", "?")] += 1
        
        return stats
    
    def export_lexicon(self, output_path: str):
        """Export lexicon to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported lexicon to {output_path}")


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

class ValencyDatabase:
    """Database for valency data"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize valency tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                valency_class TEXT,
                arguments TEXT,
                frequency INTEGER DEFAULT 1,
                examples TEXT,
                period TEXT,
                semantic_class TEXT,
                alternations TEXT,
                notes TEXT,
                source_sentence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_alternations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern_a TEXT NOT NULL,
                pattern_b TEXT NOT NULL,
                alternation_type TEXT,
                frequency_a INTEGER DEFAULT 0,
                frequency_b INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_lemma ON valency_frames(lemma)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_pattern ON valency_frames(pattern)")
        
        conn.commit()
        conn.close()
    
    def save_frame(self, frame: ValencyFrame, sentence_id: str = "") -> bool:
        """Save valency frame"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO valency_frames 
                (lemma, pattern, valency_class, arguments, frequency, examples,
                 period, semantic_class, alternations, notes, source_sentence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                frame.lemma, frame.pattern, frame.valency_class,
                json.dumps(frame.arguments), frame.frequency,
                json.dumps(frame.examples), frame.period,
                frame.semantic_class, json.dumps(frame.alternations),
                frame.notes, sentence_id
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def get_frames_for_verb(self, lemma: str) -> List[Dict]:
        """Get all frames for a verb"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM valency_frames WHERE lemma = ?
        """, (lemma,))
        
        frames = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return frames
    
    def get_pattern_statistics(self) -> Dict:
        """Get pattern statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pattern, COUNT(DISTINCT lemma) as verbs, SUM(frequency) as total
            FROM valency_frames
            GROUP BY pattern
            ORDER BY verbs DESC
        """)
        
        stats = {row["pattern"]: {"verbs": row["verbs"], "total": row["total"]} 
                 for row in cursor.fetchall()}
        conn.close()
        return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Valency Analyzer")
    parser.add_argument('command', choices=['analyze', 'lookup', 'stats', 'export'],
                       help="Command to run")
    parser.add_argument('--lemma', help="Verb lemma to look up")
    parser.add_argument('--output', '-o', help="Output file")
    
    args = parser.parse_args()
    
    analyzer = ValencyAnalyzer()
    
    if args.command == 'lookup':
        if args.lemma:
            entry = analyzer.get_lexicon_entry(args.lemma)
            if entry:
                print(json.dumps(entry, ensure_ascii=False, indent=2))
            else:
                print(f"Verb '{args.lemma}' not found in lexicon")
        else:
            print("Please provide --lemma")
    
    elif args.command == 'stats':
        stats = analyzer.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif args.command == 'export':
        output = args.output or "valency_lexicon.json"
        analyzer.export_lexicon(output)
        print(f"Exported to {output}")
    
    elif args.command == 'analyze':
        print("Valency analysis ready")
        print(f"Lexicon contains {len(analyzer.lexicon)} verbs")


if __name__ == "__main__":
    main()
