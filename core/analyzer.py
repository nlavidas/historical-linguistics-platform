"""
Linguistic Analysis Engine
Comprehensive morphological, syntactic, and semantic analysis
"""

import re
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import unicodedata

from .database import DatabaseManager, Token, Sentence, Document, ValencyFrame

logger = logging.getLogger(__name__)

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================

LANGUAGE_CONFIG = {
    "grc": {
        "name": "Ancient Greek",
        "script": "greek",
        "word_pattern": r'[\u0370-\u03FF\u1F00-\u1FFF]+',
        "sentence_delimiters": r'[.;:!?]',
        "morphology_rules": {
            "noun_endings": {
                "ος": {"Case": "Nom", "Number": "Sing", "Gender": "Masc"},
                "ον": {"Case": "Nom", "Number": "Sing", "Gender": "Neut"},
                "η": {"Case": "Nom", "Number": "Sing", "Gender": "Fem"},
                "α": {"Case": "Nom", "Number": "Sing", "Gender": "Fem"},
                "ου": {"Case": "Gen", "Number": "Sing"},
                "ης": {"Case": "Gen", "Number": "Sing", "Gender": "Fem"},
                "ας": {"Case": "Gen", "Number": "Sing", "Gender": "Fem"},
                "ω": {"Case": "Dat", "Number": "Sing"},
                "η": {"Case": "Dat", "Number": "Sing", "Gender": "Fem"},
                "α": {"Case": "Dat", "Number": "Sing", "Gender": "Fem"},
                "οι": {"Case": "Nom", "Number": "Plur", "Gender": "Masc"},
                "αι": {"Case": "Nom", "Number": "Plur", "Gender": "Fem"},
                "ων": {"Case": "Gen", "Number": "Plur"},
                "οις": {"Case": "Dat", "Number": "Plur", "Gender": "Masc"},
                "αις": {"Case": "Dat", "Number": "Plur", "Gender": "Fem"},
                "ους": {"Case": "Acc", "Number": "Plur", "Gender": "Masc"},
            },
            "verb_endings": {
                "ω": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "εις": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "2", "Number": "Sing"},
                "ει": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Sing"},
                "ομεν": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Plur"},
                "ετε": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "2", "Number": "Plur"},
                "ουσι": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Plur"},
                "ον": {"Mood": "Ind", "Tense": "Impf", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "ες": {"Mood": "Ind", "Tense": "Impf", "Voice": "Act", "Person": "2", "Number": "Sing"},
                "ε": {"Mood": "Ind", "Tense": "Impf", "Voice": "Act", "Person": "3", "Number": "Sing"},
                "σα": {"Mood": "Ind", "Tense": "Aor", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "σας": {"Mood": "Ind", "Tense": "Aor", "Voice": "Act", "Person": "2", "Number": "Sing"},
                "σε": {"Mood": "Ind", "Tense": "Aor", "Voice": "Act", "Person": "3", "Number": "Sing"},
                "ειν": {"Mood": "Inf", "Tense": "Pres", "Voice": "Act"},
                "σαι": {"Mood": "Inf", "Tense": "Aor", "Voice": "Act"},
                "ων": {"Mood": "Part", "Tense": "Pres", "Voice": "Act", "Case": "Nom", "Gender": "Masc"},
                "ουσα": {"Mood": "Part", "Tense": "Pres", "Voice": "Act", "Case": "Nom", "Gender": "Fem"},
                "ομαι": {"Mood": "Ind", "Tense": "Pres", "Voice": "Mid", "Person": "1", "Number": "Sing"},
                "εται": {"Mood": "Ind", "Tense": "Pres", "Voice": "Mid", "Person": "3", "Number": "Sing"},
            }
        },
        "pos_patterns": {
            "article": ["ο", "η", "το", "οι", "αι", "τα", "του", "της", "των", "τω", "τη", "τοις", "ταις", "τον", "την"],
            "preposition": ["εν", "εις", "εκ", "απο", "προς", "παρα", "μετα", "δια", "υπο", "περι", "κατα", "αντι"],
            "conjunction": ["και", "δε", "γαρ", "αλλα", "ουν", "ει", "οτι", "ως", "ινα", "μη", "ουδε", "μηδε"],
            "particle": ["μεν", "δη", "αν", "γε", "τε", "ουκ", "ου", "μη"],
            "pronoun": ["εγω", "συ", "αυτος", "ουτος", "εκεινος", "ος", "τις", "τι"],
        }
    },
    "la": {
        "name": "Latin",
        "script": "latin",
        "word_pattern": r'[a-zA-Z]+',
        "sentence_delimiters": r'[.;:!?]',
        "morphology_rules": {
            "noun_endings": {
                "us": {"Case": "Nom", "Number": "Sing", "Gender": "Masc"},
                "um": {"Case": "Nom", "Number": "Sing", "Gender": "Neut"},
                "a": {"Case": "Nom", "Number": "Sing", "Gender": "Fem"},
                "i": {"Case": "Gen", "Number": "Sing"},
                "ae": {"Case": "Gen", "Number": "Sing", "Gender": "Fem"},
                "o": {"Case": "Dat", "Number": "Sing"},
                "is": {"Case": "Dat", "Number": "Sing"},
                "am": {"Case": "Acc", "Number": "Sing", "Gender": "Fem"},
                "em": {"Case": "Acc", "Number": "Sing"},
                "e": {"Case": "Abl", "Number": "Sing"},
                "orum": {"Case": "Gen", "Number": "Plur", "Gender": "Masc"},
                "arum": {"Case": "Gen", "Number": "Plur", "Gender": "Fem"},
                "os": {"Case": "Acc", "Number": "Plur", "Gender": "Masc"},
                "as": {"Case": "Acc", "Number": "Plur", "Gender": "Fem"},
            },
            "verb_endings": {
                "o": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "s": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "2", "Number": "Sing"},
                "t": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Sing"},
                "mus": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Plur"},
                "tis": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "2", "Number": "Plur"},
                "nt": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Plur"},
                "bam": {"Mood": "Ind", "Tense": "Impf", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "vi": {"Mood": "Ind", "Tense": "Perf", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "re": {"Mood": "Inf", "Tense": "Pres", "Voice": "Act"},
                "ns": {"Mood": "Part", "Tense": "Pres", "Voice": "Act"},
                "tus": {"Mood": "Part", "Tense": "Perf", "Voice": "Pass"},
            }
        },
        "pos_patterns": {
            "preposition": ["in", "ex", "de", "ad", "ab", "cum", "pro", "per", "sub", "inter", "ante", "post"],
            "conjunction": ["et", "sed", "aut", "vel", "nec", "neque", "atque", "ac", "nam", "enim", "quod", "ut", "si"],
            "pronoun": ["ego", "tu", "is", "ea", "id", "hic", "ille", "qui", "quis", "quid"],
        }
    },
    "sa": {
        "name": "Sanskrit",
        "script": "devanagari",
        "word_pattern": r'[\u0900-\u097F]+',
        "sentence_delimiters": r'[।॥]',
        "morphology_rules": {
            "noun_endings": {
                "ः": {"Case": "Nom", "Number": "Sing"},
                "म्": {"Case": "Acc", "Number": "Sing"},
                "स्य": {"Case": "Gen", "Number": "Sing"},
                "आय": {"Case": "Dat", "Number": "Sing"},
                "आत्": {"Case": "Abl", "Number": "Sing"},
                "ए": {"Case": "Loc", "Number": "Sing"},
            },
            "verb_endings": {
                "ति": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Sing"},
                "न्ति": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Plur"},
                "मि": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Sing"},
            }
        },
        "pos_patterns": {}
    },
    "got": {
        "name": "Gothic",
        "script": "gothic",
        "word_pattern": r'[a-zA-Z]+',
        "sentence_delimiters": r'[.;:!?]',
        "morphology_rules": {
            "noun_endings": {
                "s": {"Case": "Nom", "Number": "Sing", "Gender": "Masc"},
                "is": {"Case": "Gen", "Number": "Sing"},
                "a": {"Case": "Dat", "Number": "Sing"},
            },
            "verb_endings": {
                "a": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "1", "Number": "Sing"},
                "is": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "2", "Number": "Sing"},
                "ip": {"Mood": "Ind", "Tense": "Pres", "Voice": "Act", "Person": "3", "Number": "Sing"},
            }
        },
        "pos_patterns": {}
    }
}

# Default configuration for unsupported languages
DEFAULT_CONFIG = {
    "name": "Unknown",
    "script": "latin",
    "word_pattern": r'\b\w+\b',
    "sentence_delimiters": r'[.;:!?]',
    "morphology_rules": {"noun_endings": {}, "verb_endings": {}},
    "pos_patterns": {}
}

# ============================================================================
# MORPHOLOGICAL ANALYZER
# ============================================================================

class MorphologicalAnalyzer:
    """Morphological analysis engine"""
    
    def __init__(self, language: str):
        self.language = language
        self.config = LANGUAGE_CONFIG.get(language, DEFAULT_CONFIG)
    
    def analyze_word(self, word: str) -> Dict[str, Any]:
        """Analyze a single word"""
        result = {
            "form": word,
            "lemma": self._lemmatize(word),
            "pos": self._get_pos(word),
            "morphology": self._get_morphology(word)
        }
        return result
    
    def _lemmatize(self, word: str) -> str:
        """Get lemma for word"""
        # Remove common endings to approximate lemma
        word_lower = word.lower()
        
        # Try verb endings first
        for ending in sorted(self.config["morphology_rules"].get("verb_endings", {}).keys(), 
                            key=len, reverse=True):
            if word_lower.endswith(ending) and len(word_lower) > len(ending):
                return word_lower[:-len(ending)] + "ω" if self.language == "grc" else word_lower[:-len(ending)] + "re"
        
        # Try noun endings
        for ending in sorted(self.config["morphology_rules"].get("noun_endings", {}).keys(), 
                            key=len, reverse=True):
            if word_lower.endswith(ending) and len(word_lower) > len(ending):
                return word_lower[:-len(ending)]
        
        return word_lower
    
    def _get_pos(self, word: str) -> str:
        """Determine part of speech"""
        word_lower = word.lower()
        
        # Check closed class words
        for pos, words in self.config.get("pos_patterns", {}).items():
            if word_lower in words:
                pos_map = {
                    "article": "DET",
                    "preposition": "ADP",
                    "conjunction": "CCONJ",
                    "particle": "PART",
                    "pronoun": "PRON"
                }
                return pos_map.get(pos, pos.upper())
        
        # Check verb endings
        for ending in self.config["morphology_rules"].get("verb_endings", {}).keys():
            if word_lower.endswith(ending):
                return "VERB"
        
        # Check noun endings
        for ending in self.config["morphology_rules"].get("noun_endings", {}).keys():
            if word_lower.endswith(ending):
                return "NOUN"
        
        return "X"
    
    def _get_morphology(self, word: str) -> Dict[str, str]:
        """Get morphological features"""
        word_lower = word.lower()
        features = {}
        
        # Check verb endings
        for ending, feats in sorted(
            self.config["morphology_rules"].get("verb_endings", {}).items(),
            key=lambda x: len(x[0]), reverse=True
        ):
            if word_lower.endswith(ending):
                features.update(feats)
                break
        
        # If no verb match, check noun endings
        if not features:
            for ending, feats in sorted(
                self.config["morphology_rules"].get("noun_endings", {}).items(),
                key=lambda x: len(x[0]), reverse=True
            ):
                if word_lower.endswith(ending):
                    features.update(feats)
                    break
        
        return features


# ============================================================================
# SYNTACTIC ANALYZER
# ============================================================================

class SyntacticAnalyzer:
    """Syntactic analysis engine"""
    
    def __init__(self, language: str):
        self.language = language
        self.config = LANGUAGE_CONFIG.get(language, DEFAULT_CONFIG)
    
    def analyze_sentence(self, tokens: List[Token]) -> List[Token]:
        """Perform dependency parsing on tokens"""
        if not tokens:
            return tokens
        
        # Find main verb (predicate)
        verb_indices = [i for i, t in enumerate(tokens) if t.pos == "VERB"]
        
        if verb_indices:
            main_verb_idx = verb_indices[0]
            tokens[main_verb_idx].head = 0
            tokens[main_verb_idx].deprel = "pred"
            
            # Assign dependencies to other tokens
            for i, token in enumerate(tokens):
                if i == main_verb_idx:
                    continue
                
                # Default: attach to main verb
                token.head = main_verb_idx + 1
                
                # Determine relation based on POS and morphology
                if token.pos == "NOUN":
                    case = token.morphology.get("Case", "")
                    if case == "Nom":
                        token.deprel = "sub"
                    elif case == "Acc":
                        token.deprel = "obj"
                    elif case == "Gen":
                        token.deprel = "obl"
                    elif case == "Dat":
                        token.deprel = "obl"
                    else:
                        token.deprel = "narg"
                
                elif token.pos == "DET":
                    # Attach to nearest noun
                    nearest_noun = self._find_nearest_noun(tokens, i)
                    if nearest_noun is not None:
                        token.head = nearest_noun + 1
                        token.deprel = "atr"
                    else:
                        token.deprel = "det"
                
                elif token.pos == "ADJ":
                    # Attach to nearest noun
                    nearest_noun = self._find_nearest_noun(tokens, i)
                    if nearest_noun is not None:
                        token.head = nearest_noun + 1
                        token.deprel = "atr"
                    else:
                        token.deprel = "amod"
                
                elif token.pos == "ADV":
                    token.deprel = "adv"
                
                elif token.pos == "ADP":
                    token.deprel = "case"
                
                elif token.pos == "CCONJ":
                    token.deprel = "cc"
                
                elif token.pos == "PART":
                    token.deprel = "advmod"
                
                elif token.pos == "PRON":
                    case = token.morphology.get("Case", "")
                    if case == "Nom":
                        token.deprel = "sub"
                    elif case == "Acc":
                        token.deprel = "obj"
                    else:
                        token.deprel = "obl"
                
                else:
                    token.deprel = "dep"
        
        else:
            # No verb found - use first token as root
            if tokens:
                tokens[0].head = 0
                tokens[0].deprel = "root"
                
                for i, token in enumerate(tokens[1:], 1):
                    token.head = 1
                    token.deprel = "dep"
        
        return tokens
    
    def _find_nearest_noun(self, tokens: List[Token], current_idx: int) -> Optional[int]:
        """Find nearest noun to current token"""
        # Look right first
        for i in range(current_idx + 1, len(tokens)):
            if tokens[i].pos == "NOUN":
                return i
        
        # Then look left
        for i in range(current_idx - 1, -1, -1):
            if tokens[i].pos == "NOUN":
                return i
        
        return None
    
    def get_dependency_tree(self, tokens: List[Token]) -> Dict:
        """Build dependency tree structure"""
        tree = {"root": [], "children": defaultdict(list)}
        
        for token in tokens:
            if token.head == 0:
                tree["root"].append(token.id)
            else:
                tree["children"][token.head].append(token.id)
        
        return tree
    
    def calculate_tree_metrics(self, tokens: List[Token]) -> Dict:
        """Calculate syntactic tree metrics"""
        if not tokens:
            return {}
        
        # Arc lengths
        arc_lengths = []
        for token in tokens:
            if token.head > 0:
                arc_lengths.append(abs(token.id - token.head))
        
        # Tree depth
        def get_depth(token_id: int, depth: int = 0) -> int:
            token = next((t for t in tokens if t.id == token_id), None)
            if token is None or token.head == 0:
                return depth
            return get_depth(token.head, depth + 1)
        
        depths = [get_depth(t.id) for t in tokens]
        
        # Dependency relation distribution
        deprel_counts = Counter(t.deprel for t in tokens)
        
        return {
            "avg_arc_length": sum(arc_lengths) / len(arc_lengths) if arc_lengths else 0,
            "max_arc_length": max(arc_lengths) if arc_lengths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "max_depth": max(depths) if depths else 0,
            "deprel_distribution": dict(deprel_counts)
        }


# ============================================================================
# VALENCY EXTRACTOR
# ============================================================================

class ValencyExtractor:
    """Extract verbal valency patterns"""
    
    def __init__(self, language: str):
        self.language = language
    
    def extract_patterns(self, tokens: List[Token]) -> List[ValencyFrame]:
        """Extract valency patterns from analyzed tokens"""
        patterns = []
        
        for token in tokens:
            if token.pos == "VERB":
                # Find arguments
                arguments = []
                
                for t in tokens:
                    if t.head == token.id:
                        arg = self._classify_argument(t)
                        if arg:
                            arguments.append(arg)
                
                # Build pattern string
                pattern = self._build_pattern(arguments)
                
                # Create valency frame
                frame = ValencyFrame(
                    verb_lemma=token.lemma,
                    language=self.language,
                    pattern=pattern,
                    arguments=arguments,
                    frequency=1,
                    examples=[" ".join(t.form for t in tokens)]
                )
                
                patterns.append(frame)
        
        return patterns
    
    def _classify_argument(self, token: Token) -> Optional[Dict]:
        """Classify argument by role and case"""
        if token.deprel not in ["sub", "obj", "obl", "xobj", "comp"]:
            return None
        
        case = token.morphology.get("Case", "")
        
        return {
            "role": token.deprel,
            "case": case,
            "form": token.form,
            "lemma": token.lemma,
            "pos": token.pos
        }
    
    def _build_pattern(self, arguments: List[Dict]) -> str:
        """Build pattern string from arguments"""
        cases = ["NOM"]  # Subject is always nominative
        
        for arg in arguments:
            case = arg.get("case", "").upper()
            if case and case not in cases:
                cases.append(case)
        
        return "+".join(sorted(cases, key=lambda x: ["NOM", "ACC", "GEN", "DAT", "ABL", "LOC", "INS"].index(x) if x in ["NOM", "ACC", "GEN", "DAT", "ABL", "LOC", "INS"] else 99))


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class LinguisticAnalyzer:
    """Main linguistic analysis engine"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.morphological_analyzers = {}
        self.syntactic_analyzers = {}
        self.valency_extractors = {}
    
    def _get_morphological_analyzer(self, language: str) -> MorphologicalAnalyzer:
        if language not in self.morphological_analyzers:
            self.morphological_analyzers[language] = MorphologicalAnalyzer(language)
        return self.morphological_analyzers[language]
    
    def _get_syntactic_analyzer(self, language: str) -> SyntacticAnalyzer:
        if language not in self.syntactic_analyzers:
            self.syntactic_analyzers[language] = SyntacticAnalyzer(language)
        return self.syntactic_analyzers[language]
    
    def _get_valency_extractor(self, language: str) -> ValencyExtractor:
        if language not in self.valency_extractors:
            self.valency_extractors[language] = ValencyExtractor(language)
        return self.valency_extractors[language]
    
    def analyze(self, text: str, language: str, analyses: List[str]) -> Dict:
        """Perform comprehensive linguistic analysis"""
        start_time = time.time()
        
        # Check cache
        text_hash = hashlib.md5(f"{text}:{language}:{','.join(sorted(analyses))}".encode()).hexdigest()
        cached = self.db.get_cached_analysis(text_hash)
        if cached:
            logger.info(f"Cache hit for analysis")
            return cached
        
        # Get language config
        config = LANGUAGE_CONFIG.get(language, DEFAULT_CONFIG)
        
        # Initialize result
        result = {
            "language": language,
            "text": text,
            "sentences": [],
            "statistics": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Tokenize into sentences
        sentences = self._tokenize_sentences(text, config)
        
        # Get analyzers
        morph_analyzer = self._get_morphological_analyzer(language)
        syntax_analyzer = self._get_syntactic_analyzer(language)
        valency_extractor = self._get_valency_extractor(language)
        
        # Process each sentence
        for sent_idx, sent_text in enumerate(sentences):
            sent_result = {
                "id": f"s{sent_idx + 1}",
                "text": sent_text,
                "tokens": []
            }
            
            # Tokenize words
            words = self._tokenize_words(sent_text, config)
            
            # Morphological analysis
            if "morphological" in analyses or "syntactic" in analyses or "valency" in analyses:
                tokens = []
                for idx, word in enumerate(words, 1):
                    morph_result = morph_analyzer.analyze_word(word)
                    token = Token(
                        id=idx,
                        form=word,
                        lemma=morph_result["lemma"],
                        pos=morph_result["pos"],
                        morphology=morph_result["morphology"]
                    )
                    tokens.append(token)
                
                # Syntactic analysis
                if "syntactic" in analyses or "valency" in analyses:
                    tokens = syntax_analyzer.analyze_sentence(tokens)
                
                # Valency extraction
                if "valency" in analyses:
                    valency_patterns = valency_extractor.extract_patterns(tokens)
                    sent_result["valency_patterns"] = [p.to_dict() for p in valency_patterns]
                
                sent_result["tokens"] = [t.to_dict() for t in tokens]
            
            result["sentences"].append(sent_result)
        
        # Calculate statistics
        result["statistics"] = self._calculate_statistics(result["sentences"])
        
        # Log performance
        elapsed = time.time() - start_time
        self.db.log_metric("analysis_time_ms", elapsed * 1000, {
            "language": language,
            "text_length": len(text),
            "sentence_count": len(sentences),
            "analyses": analyses
        })
        
        # Cache result
        self.db.cache_analysis(text_hash, language, analyses, result)
        
        logger.info(f"Analysis completed in {elapsed:.3f}s")
        return result
    
    def _tokenize_sentences(self, text: str, config: Dict) -> List[str]:
        """Tokenize text into sentences"""
        delimiter_pattern = config.get("sentence_delimiters", r'[.;:!?]')
        sentences = re.split(f'(?<={delimiter_pattern})\\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize_words(self, sentence: str, config: Dict) -> List[str]:
        """Tokenize sentence into words"""
        word_pattern = config.get("word_pattern", r'\b\w+\b')
        words = re.findall(word_pattern, sentence)
        return words
    
    def _calculate_statistics(self, sentences: List[Dict]) -> Dict:
        """Calculate text statistics"""
        all_tokens = []
        for sent in sentences:
            all_tokens.extend(sent.get("tokens", []))
        
        if not all_tokens:
            return {}
        
        # Basic counts
        token_count = len(all_tokens)
        forms = [t["form"] for t in all_tokens]
        lemmas = [t["lemma"] for t in all_tokens if t.get("lemma")]
        
        # POS distribution
        pos_counts = Counter(t["pos"] for t in all_tokens if t.get("pos"))
        
        # Morphological feature distribution
        feature_counts = defaultdict(Counter)
        for token in all_tokens:
            for feat, value in token.get("morphology", {}).items():
                feature_counts[feat][value] += 1
        
        # Dependency relation distribution
        deprel_counts = Counter(t.get("deprel", "") for t in all_tokens if t.get("deprel"))
        
        return {
            "sentence_count": len(sentences),
            "token_count": token_count,
            "type_count": len(set(forms)),
            "lemma_count": len(set(lemmas)),
            "type_token_ratio": len(set(forms)) / token_count if token_count > 0 else 0,
            "avg_sentence_length": token_count / len(sentences) if sentences else 0,
            "avg_word_length": sum(len(f) for f in forms) / token_count if token_count > 0 else 0,
            "pos_distribution": dict(pos_counts),
            "deprel_distribution": dict(deprel_counts),
            "morphological_features": {k: dict(v) for k, v in feature_counts.items()}
        }
    
    def analyze_document(self, doc: Document, analyses: List[str]) -> Document:
        """Analyze entire document"""
        for sentence in doc.sentences:
            result = self.analyze(sentence.text, doc.language, analyses)
            
            if result.get("sentences"):
                sent_result = result["sentences"][0]
                sentence.tokens = [Token.from_dict(t) for t in sent_result.get("tokens", [])]
                sentence.valency_patterns = sent_result.get("valency_patterns", [])
        
        doc.annotation_status = "complete"
        return doc
    
    def batch_analyze(self, texts: List[Tuple[str, str]], analyses: List[str]) -> List[Dict]:
        """Batch analyze multiple texts"""
        results = []
        for text, language in texts:
            result = self.analyze(text, language, analyses)
            results.append(result)
        return results
