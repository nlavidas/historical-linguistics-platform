#!/usr/bin/env python3
"""
English Comparison Module
Cross-linguistic analysis between Greek and English

Features:
- Parallel text alignment
- Translation comparison
- Cognate tracking
- Semantic shift analysis
- Syntactic pattern comparison
"""

import re
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GREEK-ENGLISH COGNATES
# ============================================================================

GREEK_ENGLISH_COGNATES = {
    # Direct borrowings from Greek
    "philosophy": {"greek": "φιλοσοφία", "meaning": "love of wisdom", "type": "direct"},
    "democracy": {"greek": "δημοκρατία", "meaning": "rule of the people", "type": "direct"},
    "theology": {"greek": "θεολογία", "meaning": "study of god", "type": "direct"},
    "biology": {"greek": "βιολογία", "meaning": "study of life", "type": "direct"},
    "psychology": {"greek": "ψυχολογία", "meaning": "study of soul/mind", "type": "direct"},
    "anthropology": {"greek": "ἀνθρωπολογία", "meaning": "study of humans", "type": "direct"},
    "archaeology": {"greek": "ἀρχαιολογία", "meaning": "study of ancient things", "type": "direct"},
    "etymology": {"greek": "ἐτυμολογία", "meaning": "study of true meanings", "type": "direct"},
    "syntax": {"greek": "σύνταξις", "meaning": "arrangement together", "type": "direct"},
    "morphology": {"greek": "μορφολογία", "meaning": "study of form", "type": "direct"},
    "phonology": {"greek": "φωνολογία", "meaning": "study of sound", "type": "direct"},
    "semantics": {"greek": "σημαντικός", "meaning": "significant", "type": "direct"},
    "pragmatics": {"greek": "πραγματικός", "meaning": "practical", "type": "direct"},
    
    # PIE cognates
    "mother": {"greek": "μήτηρ", "pie": "*méh₂tēr", "type": "cognate"},
    "father": {"greek": "πατήρ", "pie": "*ph₂tḗr", "type": "cognate"},
    "brother": {"greek": "φράτηρ", "pie": "*bʰréh₂tēr", "type": "cognate"},
    "daughter": {"greek": "θυγάτηρ", "pie": "*dʰugh₂tḗr", "type": "cognate"},
    "is": {"greek": "ἐστί", "pie": "*h₁ésti", "type": "cognate"},
    "know": {"greek": "γιγνώσκω", "pie": "*ǵneh₃-", "type": "cognate"},
    "stand": {"greek": "ἵστημι", "pie": "*steh₂-", "type": "cognate"},
    "two": {"greek": "δύο", "pie": "*dwóh₁", "type": "cognate"},
    "three": {"greek": "τρεῖς", "pie": "*tréyes", "type": "cognate"},
    "new": {"greek": "νέος", "pie": "*néwos", "type": "cognate"},
    "night": {"greek": "νύξ", "pie": "*nókʷts", "type": "cognate"},
    "name": {"greek": "ὄνομα", "pie": "*h₁nómn̥", "type": "cognate"},
    "heart": {"greek": "καρδία", "pie": "*ḱerd-", "type": "cognate"},
    "fire": {"greek": "πῦρ", "pie": "*péh₂wr̥", "type": "cognate"},
    "water": {"greek": "ὕδωρ", "pie": "*wódr̥", "type": "cognate"},
    
    # Scientific/medical terms
    "cardiac": {"greek": "καρδιακός", "meaning": "of the heart", "type": "scientific"},
    "hepatic": {"greek": "ἡπατικός", "meaning": "of the liver", "type": "scientific"},
    "renal": {"greek": "νεφρός (via Latin)", "meaning": "of the kidney", "type": "scientific"},
    "dermal": {"greek": "δέρμα", "meaning": "of the skin", "type": "scientific"},
    "neural": {"greek": "νεῦρον", "meaning": "of the nerve", "type": "scientific"},
    "optical": {"greek": "ὀπτικός", "meaning": "of sight", "type": "scientific"},
    "acoustic": {"greek": "ἀκουστικός", "meaning": "of hearing", "type": "scientific"},
    
    # Philosophical terms
    "logos": {"greek": "λόγος", "meaning": "word, reason, account", "type": "philosophical"},
    "ethos": {"greek": "ἦθος", "meaning": "character, custom", "type": "philosophical"},
    "pathos": {"greek": "πάθος", "meaning": "suffering, emotion", "type": "philosophical"},
    "telos": {"greek": "τέλος", "meaning": "end, purpose", "type": "philosophical"},
    "cosmos": {"greek": "κόσμος", "meaning": "order, world", "type": "philosophical"},
    "chaos": {"greek": "χάος", "meaning": "void, disorder", "type": "philosophical"},
    "psyche": {"greek": "ψυχή", "meaning": "soul, mind", "type": "philosophical"},
    "nous": {"greek": "νοῦς", "meaning": "mind, intellect", "type": "philosophical"},
    "sophia": {"greek": "σοφία", "meaning": "wisdom", "type": "philosophical"},
    "arete": {"greek": "ἀρετή", "meaning": "excellence, virtue", "type": "philosophical"},
    "eudaimonia": {"greek": "εὐδαιμονία", "meaning": "flourishing, happiness", "type": "philosophical"}
}

# ============================================================================
# PARALLEL TEXTS
# ============================================================================

PARALLEL_TEXTS = {
    "new_testament": {
        "john_1_1": {
            "greek": "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
            "english": "In the beginning was the Word, and the Word was with God, and the Word was God.",
            "alignment": [
                ("Ἐν", "In"),
                ("ἀρχῇ", "the beginning"),
                ("ἦν", "was"),
                ("ὁ λόγος", "the Word"),
                ("καὶ", "and"),
                ("πρὸς", "with"),
                ("τὸν θεόν", "God"),
                ("θεὸς", "God"),
                ("ἦν", "was")
            ]
        },
        "matthew_5_3": {
            "greek": "Μακάριοι οἱ πτωχοὶ τῷ πνεύματι, ὅτι αὐτῶν ἐστιν ἡ βασιλεία τῶν οὐρανῶν.",
            "english": "Blessed are the poor in spirit, for theirs is the kingdom of heaven.",
            "alignment": [
                ("Μακάριοι", "Blessed"),
                ("οἱ πτωχοὶ", "the poor"),
                ("τῷ πνεύματι", "in spirit"),
                ("ὅτι", "for"),
                ("αὐτῶν", "theirs"),
                ("ἐστιν", "is"),
                ("ἡ βασιλεία", "the kingdom"),
                ("τῶν οὐρανῶν", "of heaven")
            ]
        }
    },
    "homer": {
        "iliad_1_1": {
            "greek": "Μῆνιν ἄειδε, θεά, Πηληϊάδεω Ἀχιλῆος",
            "english": "Sing, goddess, the wrath of Achilles, son of Peleus",
            "alignment": [
                ("Μῆνιν", "the wrath"),
                ("ἄειδε", "sing"),
                ("θεά", "goddess"),
                ("Πηληϊάδεω", "son of Peleus"),
                ("Ἀχιλῆος", "of Achilles")
            ]
        },
        "odyssey_1_1": {
            "greek": "Ἄνδρα μοι ἔννεπε, Μοῦσα, πολύτροπον, ὃς μάλα πολλὰ",
            "english": "Tell me, Muse, of the man of many ways, who",
            "alignment": [
                ("Ἄνδρα", "the man"),
                ("μοι", "me"),
                ("ἔννεπε", "tell"),
                ("Μοῦσα", "Muse"),
                ("πολύτροπον", "of many ways"),
                ("ὃς", "who"),
                ("μάλα πολλὰ", "very much")
            ]
        }
    },
    "plato": {
        "republic_514a": {
            "greek": "Μετὰ ταῦτα δή, εἶπον, ἀπείκασον τοιούτῳ πάθει τὴν ἡμετέραν φύσιν παιδείας τε πέρι καὶ ἀπαιδευσίας.",
            "english": "After this, I said, compare our nature in respect of education and its lack to such an experience as this.",
            "alignment": [
                ("Μετὰ ταῦτα", "After this"),
                ("εἶπον", "I said"),
                ("ἀπείκασον", "compare"),
                ("τὴν ἡμετέραν φύσιν", "our nature"),
                ("παιδείας", "education"),
                ("ἀπαιδευσίας", "lack of education")
            ]
        }
    }
}

# ============================================================================
# SYNTACTIC PATTERNS
# ============================================================================

SYNTACTIC_PATTERNS = {
    "word_order": {
        "greek": {
            "default": "SOV/free",
            "description": "Relatively free word order with SOV tendency",
            "examples": [
                {"pattern": "SOV", "example": "ὁ ἄνθρωπος τὸν λόγον λέγει"},
                {"pattern": "VSO", "example": "λέγει ὁ ἄνθρωπος τὸν λόγον"},
                {"pattern": "OVS", "example": "τὸν λόγον λέγει ὁ ἄνθρωπος"}
            ]
        },
        "english": {
            "default": "SVO",
            "description": "Fixed SVO word order",
            "examples": [
                {"pattern": "SVO", "example": "The man speaks the word"}
            ]
        }
    },
    "case_system": {
        "greek": {
            "cases": ["nominative", "genitive", "dative", "accusative", "vocative"],
            "functions": {
                "nominative": "subject",
                "genitive": "possession, source, partitive",
                "dative": "indirect object, instrument, location",
                "accusative": "direct object, extent",
                "vocative": "address"
            }
        },
        "english": {
            "cases": ["nominative", "accusative", "genitive"],
            "notes": "Case only preserved in pronouns (I/me, he/him, etc.)"
        }
    },
    "article_usage": {
        "greek": {
            "definite": "ὁ, ἡ, τό",
            "indefinite": "τις (enclitic)",
            "notes": "Article can nominalize any part of speech"
        },
        "english": {
            "definite": "the",
            "indefinite": "a, an",
            "notes": "Articles cannot nominalize"
        }
    },
    "participle_usage": {
        "greek": {
            "types": ["present", "aorist", "perfect", "future"],
            "functions": ["attributive", "circumstantial", "supplementary"],
            "notes": "Participles fully declined, very common"
        },
        "english": {
            "types": ["present (-ing)", "past (-ed)"],
            "functions": ["attributive", "adverbial"],
            "notes": "Less common, often replaced by relative clauses"
        }
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AlignedSentence:
    """Aligned Greek-English sentence pair"""
    id: str
    greek_text: str
    english_text: str
    alignments: List[Tuple[str, str]] = field(default_factory=list)
    source: str = ""
    notes: str = ""


@dataclass
class CognateEntry:
    """Cognate relationship entry"""
    english: str
    greek: str
    relationship_type: str  # direct, cognate, scientific, philosophical
    pie_root: str = ""
    meaning: str = ""
    semantic_shift: str = ""


@dataclass
class SyntacticComparison:
    """Syntactic pattern comparison"""
    feature: str
    greek_pattern: str
    english_pattern: str
    examples: List[Dict] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# COMPARISON ANALYZER
# ============================================================================

class EnglishComparisonAnalyzer:
    """Analyze Greek-English comparisons"""
    
    def __init__(self):
        self.cognates = GREEK_ENGLISH_COGNATES
        self.parallel_texts = PARALLEL_TEXTS
        self.syntactic_patterns = SYNTACTIC_PATTERNS
    
    def find_cognate(self, word: str, language: str = "english") -> Optional[Dict]:
        """Find cognate relationship"""
        word_lower = word.lower()
        
        if language == "english":
            if word_lower in self.cognates:
                return self.cognates[word_lower]
        else:
            # Search by Greek form
            for eng, data in self.cognates.items():
                if data.get("greek", "").lower() == word_lower:
                    return {"english": eng, **data}
        
        return None
    
    def get_parallel_text(self, source: str, passage: str) -> Optional[Dict]:
        """Get parallel text"""
        if source in self.parallel_texts:
            if passage in self.parallel_texts[source]:
                return self.parallel_texts[source][passage]
        return None
    
    def compare_syntax(self, feature: str) -> Optional[Dict]:
        """Compare syntactic feature"""
        if feature in self.syntactic_patterns:
            return self.syntactic_patterns[feature]
        return None
    
    def analyze_translation(self, greek_text: str, english_text: str) -> Dict:
        """Analyze translation differences"""
        analysis = {
            "greek_words": len(greek_text.split()),
            "english_words": len(english_text.split()),
            "word_ratio": len(english_text.split()) / max(len(greek_text.split()), 1),
            "cognates_found": [],
            "notes": []
        }
        
        # Find cognates in English text
        for word in english_text.lower().split():
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in self.cognates:
                analysis["cognates_found"].append({
                    "english": word_clean,
                    "greek": self.cognates[word_clean].get("greek", "")
                })
        
        # Note word order differences
        if analysis["word_ratio"] > 1.2:
            analysis["notes"].append("English uses more words (analytic structure)")
        elif analysis["word_ratio"] < 0.8:
            analysis["notes"].append("Greek uses more words")
        
        return analysis
    
    def get_cognate_statistics(self) -> Dict:
        """Get cognate statistics"""
        stats = {
            "total": len(self.cognates),
            "by_type": defaultdict(int)
        }
        
        for word, data in self.cognates.items():
            stats["by_type"][data.get("type", "unknown")] += 1
        
        return stats
    
    def export_cognates(self, output_path: str):
        """Export cognate database"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.cognates, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported cognates to {output_path}")


# ============================================================================
# DATABASE
# ============================================================================

class ComparisonDatabase:
    """Database for comparison data"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize comparison tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Parallel texts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parallel_texts (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                passage TEXT NOT NULL,
                greek_text TEXT NOT NULL,
                english_text TEXT NOT NULL,
                alignments TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cognates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                english TEXT NOT NULL,
                greek TEXT NOT NULL,
                relationship_type TEXT,
                pie_root TEXT,
                meaning TEXT,
                semantic_shift TEXT,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_parallel_text(self, aligned: AlignedSentence) -> bool:
        """Save parallel text"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO parallel_texts 
                (id, source, passage, greek_text, english_text, alignments, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                aligned.id, aligned.source, "",
                aligned.greek_text, aligned.english_text,
                json.dumps(aligned.alignments), aligned.notes
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving parallel text: {e}")
            return False
    
    def search_parallel(self, query: str) -> List[Dict]:
        """Search parallel texts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM parallel_texts 
            WHERE greek_text LIKE ? OR english_text LIKE ?
            LIMIT 50
        """, (f"%{query}%", f"%{query}%"))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="English Comparison Module")
    parser.add_argument('command', choices=['cognate', 'parallel', 'syntax', 'stats', 'export'],
                       help="Command to run")
    parser.add_argument('--word', '-w', help="Word to look up")
    parser.add_argument('--source', '-s', help="Text source")
    parser.add_argument('--passage', '-p', help="Passage ID")
    parser.add_argument('--feature', '-f', help="Syntactic feature")
    parser.add_argument('--output', '-o', help="Output file")
    
    args = parser.parse_args()
    
    analyzer = EnglishComparisonAnalyzer()
    
    if args.command == 'cognate':
        if args.word:
            result = analyzer.find_cognate(args.word)
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"No cognate found for '{args.word}'")
        else:
            print("Please provide --word")
    
    elif args.command == 'parallel':
        if args.source and args.passage:
            result = analyzer.get_parallel_text(args.source, args.passage)
            if result:
                print(f"Greek: {result['greek']}")
                print(f"English: {result['english']}")
                print("\nAlignments:")
                for gr, en in result.get('alignment', []):
                    print(f"  {gr} → {en}")
            else:
                print("Passage not found")
        else:
            print("Available sources:", list(PARALLEL_TEXTS.keys()))
    
    elif args.command == 'syntax':
        if args.feature:
            result = analyzer.compare_syntax(args.feature)
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Feature '{args.feature}' not found")
        else:
            print("Available features:", list(SYNTACTIC_PATTERNS.keys()))
    
    elif args.command == 'stats':
        stats = analyzer.get_cognate_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif args.command == 'export':
        output = args.output or "cognates.json"
        analyzer.export_cognates(output)
        print(f"Exported to {output}")


if __name__ == "__main__":
    main()
