"""
Treebank Management Module
PROIEL, Syntacticus, and Universal Dependencies treebank support
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
import json

from .database import Token, Sentence, Document, DatabaseManager

logger = logging.getLogger(__name__)

# ============================================================================
# PROIEL MORPHOLOGY ENCODING
# ============================================================================

PROIEL_PERSON = {"1": "1", "2": "2", "3": "3", "-": None}
PROIEL_NUMBER = {"s": "Sing", "d": "Dual", "p": "Plur", "-": None}
PROIEL_TENSE = {"p": "Pres", "i": "Impf", "f": "Fut", "s": "Fut", "a": "Aor", "r": "Perf", "l": "Pqp", "t": "FutPerf", "-": None}
PROIEL_MOOD = {"i": "Ind", "s": "Sub", "o": "Opt", "m": "Imp", "n": "Inf", "p": "Part", "d": "Ger", "g": "Gdv", "u": "Sup", "-": None}
PROIEL_VOICE = {"a": "Act", "m": "Mid", "p": "Pass", "e": "MidPass", "-": None}
PROIEL_GENDER = {"m": "Masc", "f": "Fem", "n": "Neut", "o": "MascNeut", "p": "MascFem", "r": "FemNeut", "-": None}
PROIEL_CASE = {"n": "Nom", "g": "Gen", "d": "Dat", "a": "Acc", "v": "Voc", "b": "Abl", "l": "Loc", "i": "Ins", "-": None}
PROIEL_DEGREE = {"p": "Pos", "c": "Cmp", "s": "Sup", "-": None}
PROIEL_STRENGTH = {"w": "Weak", "s": "Strong", "t": "Strong", "-": None}
PROIEL_INFLECTION = {"n": "Nominal", "i": "Pronominal", "-": None}

PROIEL_POS_MAP = {
    "A-": "ADJ",
    "Df": "ADV",
    "S-": "ART",
    "Ma": "NUM",
    "Mo": "NUM",
    "Nb": "NOUN",
    "Ne": "PROPN",
    "Pc": "PRON",
    "Pd": "PRON",
    "Pi": "PRON",
    "Pk": "PRON",
    "Pp": "PRON",
    "Pr": "PRON",
    "Ps": "PRON",
    "Pt": "PRON",
    "Px": "PRON",
    "Py": "PRON",
    "R-": "ADP",
    "C-": "CCONJ",
    "G-": "SCONJ",
    "Du": "ADV",
    "Dq": "ADV",
    "I-": "INTJ",
    "V-": "VERB",
    "N-": "NOUN",
    "F-": "PUNCT",
    "X-": "X"
}

PROIEL_DEPREL_MAP = {
    "pred": "root",
    "sub": "nsubj",
    "obj": "obj",
    "obl": "obl",
    "ag": "obl:agent",
    "atr": "amod",
    "atv": "acl",
    "adv": "advmod",
    "apos": "appos",
    "aux": "aux",
    "comp": "ccomp",
    "expl": "expl",
    "narg": "dep",
    "nonsub": "dep",
    "parpred": "parataxis",
    "per": "discourse",
    "voc": "vocative",
    "xadv": "advcl",
    "xobj": "xcomp",
    "xsub": "nsubj"
}

# ============================================================================
# TREEBANK PARSER
# ============================================================================

class TreebankParser:
    """Parse various treebank formats"""
    
    def __init__(self):
        pass
    
    def parse_proiel_xml(self, xml_content: str) -> List[Sentence]:
        """Parse PROIEL XML format"""
        sentences = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Find all sentences
            for sent_elem in root.findall(".//sentence"):
                sent_id = sent_elem.get("id", "")
                sent_status = sent_elem.get("status", "")
                presentation_before = sent_elem.get("presentation-before", "")
                presentation_after = sent_elem.get("presentation-after", "")
                
                tokens = []
                
                for token_elem in sent_elem.findall("token"):
                    token = self._parse_proiel_token(token_elem)
                    if token:
                        tokens.append(token)
                
                # Build sentence text
                text = " ".join(t.form for t in tokens)
                
                sentence = Sentence(
                    id=sent_id,
                    text=text,
                    tokens=tokens,
                    metadata={
                        "status": sent_status,
                        "presentation_before": presentation_before,
                        "presentation_after": presentation_after
                    }
                )
                sentences.append(sentence)
        
        except ET.ParseError as e:
            logger.error(f"Error parsing PROIEL XML: {e}")
        
        return sentences
    
    def _parse_proiel_token(self, elem: ET.Element) -> Optional[Token]:
        """Parse a single PROIEL token element"""
        try:
            token_id = int(elem.get("id", 0))
            form = elem.get("form", "")
            lemma = elem.get("lemma", "")
            pos = elem.get("part-of-speech", "")
            morph_str = elem.get("morphology", "")
            head_id = elem.get("head-id", "")
            relation = elem.get("relation", "")
            
            # Parse morphology
            morphology = self._parse_proiel_morphology(morph_str)
            
            # Map POS to UD
            upos = PROIEL_POS_MAP.get(pos, "X")
            
            # Parse head
            head = int(head_id) if head_id and head_id.isdigit() else 0
            
            # Map relation
            deprel = relation
            
            return Token(
                id=token_id,
                form=form,
                lemma=lemma,
                pos=upos,
                xpos=pos,
                morphology=morphology,
                head=head,
                deprel=deprel
            )
        
        except Exception as e:
            logger.warning(f"Error parsing PROIEL token: {e}")
            return None
    
    def _parse_proiel_morphology(self, morph_str: str) -> Dict[str, str]:
        """Parse PROIEL positional morphology string"""
        features = {}
        
        if not morph_str or morph_str == "---------":
            return features
        
        # Pad to 10 characters
        morph_str = morph_str.ljust(10, "-")
        
        # Position 0: Person
        if len(morph_str) > 0 and morph_str[0] in PROIEL_PERSON:
            val = PROIEL_PERSON[morph_str[0]]
            if val:
                features["Person"] = val
        
        # Position 1: Number
        if len(morph_str) > 1 and morph_str[1] in PROIEL_NUMBER:
            val = PROIEL_NUMBER[morph_str[1]]
            if val:
                features["Number"] = val
        
        # Position 2: Tense
        if len(morph_str) > 2 and morph_str[2] in PROIEL_TENSE:
            val = PROIEL_TENSE[morph_str[2]]
            if val:
                features["Tense"] = val
        
        # Position 3: Mood
        if len(morph_str) > 3 and morph_str[3] in PROIEL_MOOD:
            val = PROIEL_MOOD[morph_str[3]]
            if val:
                features["Mood"] = val
        
        # Position 4: Voice
        if len(morph_str) > 4 and morph_str[4] in PROIEL_VOICE:
            val = PROIEL_VOICE[morph_str[4]]
            if val:
                features["Voice"] = val
        
        # Position 5: Gender
        if len(morph_str) > 5 and morph_str[5] in PROIEL_GENDER:
            val = PROIEL_GENDER[morph_str[5]]
            if val:
                features["Gender"] = val
        
        # Position 6: Case
        if len(morph_str) > 6 and morph_str[6] in PROIEL_CASE:
            val = PROIEL_CASE[morph_str[6]]
            if val:
                features["Case"] = val
        
        # Position 7: Degree
        if len(morph_str) > 7 and morph_str[7] in PROIEL_DEGREE:
            val = PROIEL_DEGREE[morph_str[7]]
            if val:
                features["Degree"] = val
        
        # Position 8: Strength (for adjectives)
        if len(morph_str) > 8 and morph_str[8] in PROIEL_STRENGTH:
            val = PROIEL_STRENGTH[morph_str[8]]
            if val:
                features["Strength"] = val
        
        # Position 9: Inflection
        if len(morph_str) > 9 and morph_str[9] in PROIEL_INFLECTION:
            val = PROIEL_INFLECTION[morph_str[9]]
            if val:
                features["Inflection"] = val
        
        return features
    
    def parse_conllu(self, content: str) -> List[Sentence]:
        """Parse CoNLL-U format"""
        sentences = []
        current_tokens = []
        sent_id = ""
        sent_text = ""
        metadata = {}
        
        for line in content.split("\n"):
            line = line.rstrip()
            
            if line.startswith("# sent_id"):
                if "=" in line:
                    sent_id = line.split("=", 1)[1].strip()
            
            elif line.startswith("# text"):
                if "=" in line:
                    sent_text = line.split("=", 1)[1].strip()
            
            elif line.startswith("#"):
                # Other metadata
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    metadata[key.strip()] = value.strip()
            
            elif not line:
                # End of sentence
                if current_tokens:
                    sentence = Sentence(
                        id=sent_id or f"s{len(sentences)+1}",
                        text=sent_text or " ".join(t.form for t in current_tokens),
                        tokens=current_tokens,
                        metadata=metadata
                    )
                    sentences.append(sentence)
                    current_tokens = []
                    sent_id = ""
                    sent_text = ""
                    metadata = {}
            
            else:
                # Token line
                token = Token.from_conllu(line)
                if token:
                    current_tokens.append(token)
        
        # Handle last sentence
        if current_tokens:
            sentence = Sentence(
                id=sent_id or f"s{len(sentences)+1}",
                text=sent_text or " ".join(t.form for t in current_tokens),
                tokens=current_tokens,
                metadata=metadata
            )
            sentences.append(sentence)
        
        return sentences
    
    def parse_agdt(self, xml_content: str) -> List[Sentence]:
        """Parse Ancient Greek Dependency Treebank XML format"""
        sentences = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for sent_elem in root.findall(".//sentence"):
                sent_id = sent_elem.get("id", "")
                sent_subdoc = sent_elem.get("subdoc", "")
                
                tokens = []
                
                for word_elem in sent_elem.findall("word"):
                    token = self._parse_agdt_word(word_elem)
                    if token:
                        tokens.append(token)
                
                text = " ".join(t.form for t in tokens)
                
                sentence = Sentence(
                    id=sent_id,
                    text=text,
                    tokens=tokens,
                    metadata={"subdoc": sent_subdoc}
                )
                sentences.append(sentence)
        
        except ET.ParseError as e:
            logger.error(f"Error parsing AGDT XML: {e}")
        
        return sentences
    
    def _parse_agdt_word(self, elem: ET.Element) -> Optional[Token]:
        """Parse AGDT word element"""
        try:
            token_id = int(elem.get("id", 0))
            form = elem.get("form", "")
            lemma = elem.get("lemma", "")
            postag = elem.get("postag", "")
            head = int(elem.get("head", 0))
            relation = elem.get("relation", "")
            
            # Parse AGDT postag (9 positions)
            morphology = self._parse_agdt_postag(postag)
            
            # Map POS
            pos = self._agdt_pos_to_upos(postag[0] if postag else "")
            
            return Token(
                id=token_id,
                form=form,
                lemma=lemma,
                pos=pos,
                xpos=postag,
                morphology=morphology,
                head=head,
                deprel=relation
            )
        
        except Exception as e:
            logger.warning(f"Error parsing AGDT word: {e}")
            return None
    
    def _parse_agdt_postag(self, postag: str) -> Dict[str, str]:
        """Parse AGDT 9-position postag"""
        features = {}
        
        if not postag or len(postag) < 9:
            return features
        
        # Position 1: Person
        person_map = {"1": "1", "2": "2", "3": "3"}
        if postag[1] in person_map:
            features["Person"] = person_map[postag[1]]
        
        # Position 2: Number
        number_map = {"s": "Sing", "p": "Plur", "d": "Dual"}
        if postag[2] in number_map:
            features["Number"] = number_map[postag[2]]
        
        # Position 3: Tense
        tense_map = {"p": "Pres", "i": "Impf", "r": "Perf", "l": "Pqp", "t": "FutPerf", "f": "Fut", "a": "Aor"}
        if postag[3] in tense_map:
            features["Tense"] = tense_map[postag[3]]
        
        # Position 4: Mood
        mood_map = {"i": "Ind", "s": "Sub", "o": "Opt", "n": "Inf", "m": "Imp", "p": "Part"}
        if postag[4] in mood_map:
            features["Mood"] = mood_map[postag[4]]
        
        # Position 5: Voice
        voice_map = {"a": "Act", "p": "Pass", "m": "Mid", "e": "MidPass"}
        if postag[5] in voice_map:
            features["Voice"] = voice_map[postag[5]]
        
        # Position 6: Gender
        gender_map = {"m": "Masc", "f": "Fem", "n": "Neut"}
        if postag[6] in gender_map:
            features["Gender"] = gender_map[postag[6]]
        
        # Position 7: Case
        case_map = {"n": "Nom", "g": "Gen", "d": "Dat", "a": "Acc", "v": "Voc"}
        if postag[7] in case_map:
            features["Case"] = case_map[postag[7]]
        
        # Position 8: Degree
        degree_map = {"p": "Pos", "c": "Cmp", "s": "Sup"}
        if postag[8] in degree_map:
            features["Degree"] = degree_map[postag[8]]
        
        return features
    
    def _agdt_pos_to_upos(self, pos_char: str) -> str:
        """Map AGDT POS character to UPOS"""
        pos_map = {
            "n": "NOUN",
            "v": "VERB",
            "t": "VERB",  # participle
            "a": "ADJ",
            "d": "ADV",
            "l": "DET",
            "g": "PART",
            "c": "CCONJ",
            "r": "ADP",
            "p": "PRON",
            "m": "NUM",
            "i": "INTJ",
            "e": "INTJ",
            "u": "PUNCT",
            "x": "X"
        }
        return pos_map.get(pos_char.lower(), "X")


# ============================================================================
# TREEBANK EXPORTER
# ============================================================================

class TreebankExporter:
    """Export to various treebank formats"""
    
    def __init__(self):
        pass
    
    def to_conllu(self, sentences: List[Sentence]) -> str:
        """Export to CoNLL-U format"""
        lines = []
        
        for sentence in sentences:
            lines.append(f"# sent_id = {sentence.id}")
            lines.append(f"# text = {sentence.text}")
            
            for key, value in sentence.metadata.items():
                if key not in ["sent_id", "text"]:
                    lines.append(f"# {key} = {value}")
            
            for token in sentence.tokens:
                lines.append(token.to_conllu())
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_proiel_xml(self, sentences: List[Sentence], source_info: Dict = None) -> str:
        """Export to PROIEL XML format"""
        root = ET.Element("proiel")
        root.set("export-time", "")
        root.set("schema-version", "2.1")
        
        source = ET.SubElement(root, "source")
        if source_info:
            source.set("id", source_info.get("id", ""))
            source.set("language", source_info.get("language", ""))
        
        div = ET.SubElement(source, "div")
        
        for sentence in sentences:
            sent_elem = ET.SubElement(div, "sentence")
            sent_elem.set("id", sentence.id)
            
            if sentence.metadata.get("status"):
                sent_elem.set("status", sentence.metadata["status"])
            
            for token in sentence.tokens:
                token_elem = ET.SubElement(sent_elem, "token")
                token_elem.set("id", str(token.id))
                token_elem.set("form", token.form)
                
                if token.lemma:
                    token_elem.set("lemma", token.lemma)
                
                if token.xpos:
                    token_elem.set("part-of-speech", token.xpos)
                
                # Convert morphology to PROIEL format
                morph_str = self._morphology_to_proiel(token.morphology)
                if morph_str:
                    token_elem.set("morphology", morph_str)
                
                if token.head > 0:
                    token_elem.set("head-id", str(token.head))
                
                if token.deprel:
                    token_elem.set("relation", token.deprel)
        
        return ET.tostring(root, encoding="unicode")
    
    def _morphology_to_proiel(self, morphology: Dict[str, str]) -> str:
        """Convert morphology dict to PROIEL positional string"""
        result = ["-"] * 10
        
        # Person
        person_map = {"1": "1", "2": "2", "3": "3"}
        if "Person" in morphology:
            result[0] = person_map.get(morphology["Person"], "-")
        
        # Number
        number_map = {"Sing": "s", "Dual": "d", "Plur": "p"}
        if "Number" in morphology:
            result[1] = number_map.get(morphology["Number"], "-")
        
        # Tense
        tense_map = {"Pres": "p", "Impf": "i", "Fut": "f", "Aor": "a", "Perf": "r", "Pqp": "l"}
        if "Tense" in morphology:
            result[2] = tense_map.get(morphology["Tense"], "-")
        
        # Mood
        mood_map = {"Ind": "i", "Sub": "s", "Opt": "o", "Imp": "m", "Inf": "n", "Part": "p"}
        if "Mood" in morphology:
            result[3] = mood_map.get(morphology["Mood"], "-")
        
        # Voice
        voice_map = {"Act": "a", "Mid": "m", "Pass": "p", "MidPass": "e"}
        if "Voice" in morphology:
            result[4] = voice_map.get(morphology["Voice"], "-")
        
        # Gender
        gender_map = {"Masc": "m", "Fem": "f", "Neut": "n"}
        if "Gender" in morphology:
            result[5] = gender_map.get(morphology["Gender"], "-")
        
        # Case
        case_map = {"Nom": "n", "Gen": "g", "Dat": "d", "Acc": "a", "Voc": "v", "Abl": "b", "Loc": "l", "Ins": "i"}
        if "Case" in morphology:
            result[6] = case_map.get(morphology["Case"], "-")
        
        # Degree
        degree_map = {"Pos": "p", "Cmp": "c", "Sup": "s"}
        if "Degree" in morphology:
            result[7] = degree_map.get(morphology["Degree"], "-")
        
        return "".join(result)
    
    def to_json(self, sentences: List[Sentence]) -> str:
        """Export to JSON format"""
        data = {
            "sentences": [s.to_dict() for s in sentences]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def to_csv(self, sentences: List[Sentence]) -> str:
        """Export to CSV format"""
        lines = ["sent_id,token_id,form,lemma,pos,xpos,morphology,head,deprel"]
        
        for sentence in sentences:
            for token in sentence.tokens:
                morph_str = "|".join(f"{k}={v}" for k, v in token.morphology.items())
                lines.append(f"{sentence.id},{token.id},{token.form},{token.lemma},{token.pos},{token.xpos},{morph_str},{token.head},{token.deprel}")
        
        return "\n".join(lines)


# ============================================================================
# TREEBANK STATISTICS
# ============================================================================

class TreebankStatistics:
    """Calculate treebank statistics"""
    
    def __init__(self):
        pass
    
    def calculate(self, sentences: List[Sentence]) -> Dict:
        """Calculate comprehensive statistics"""
        if not sentences:
            return {}
        
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence.tokens)
        
        stats = {
            "sentence_count": len(sentences),
            "token_count": len(all_tokens),
            "avg_sentence_length": len(all_tokens) / len(sentences) if sentences else 0
        }
        
        # Unique lemmas
        lemmas = [t.lemma for t in all_tokens if t.lemma]
        stats["unique_lemmas"] = len(set(lemmas))
        
        # POS distribution
        pos_counts = Counter(t.pos for t in all_tokens if t.pos)
        stats["pos_distribution"] = dict(pos_counts.most_common())
        
        # Dependency relation distribution
        deprel_counts = Counter(t.deprel for t in all_tokens if t.deprel)
        stats["deprel_distribution"] = dict(deprel_counts.most_common())
        
        # Arc length statistics
        arc_lengths = []
        for token in all_tokens:
            if token.head > 0:
                arc_lengths.append(abs(token.id - token.head))
        
        if arc_lengths:
            stats["avg_arc_length"] = sum(arc_lengths) / len(arc_lengths)
            stats["max_arc_length"] = max(arc_lengths)
            stats["arc_length_distribution"] = dict(Counter(arc_lengths).most_common(10))
        
        # Tree depth statistics
        depths = []
        for sentence in sentences:
            depth = self._calculate_tree_depth(sentence.tokens)
            depths.append(depth)
        
        if depths:
            stats["avg_tree_depth"] = sum(depths) / len(depths)
            stats["max_tree_depth"] = max(depths)
        
        # Morphological feature statistics
        feature_counts = defaultdict(Counter)
        for token in all_tokens:
            for feat, value in token.morphology.items():
                feature_counts[feat][value] += 1
        
        stats["morphological_features"] = {
            feat: dict(counts.most_common())
            for feat, counts in feature_counts.items()
        }
        
        # Non-projective arcs
        non_proj_count = 0
        for sentence in sentences:
            non_proj_count += self._count_non_projective(sentence.tokens)
        
        stats["non_projective_arcs"] = non_proj_count
        stats["non_projectivity_rate"] = non_proj_count / len(all_tokens) if all_tokens else 0
        
        return stats
    
    def _calculate_tree_depth(self, tokens: List[Token]) -> int:
        """Calculate maximum tree depth"""
        if not tokens:
            return 0
        
        def get_depth(token_id: int, visited: set = None) -> int:
            if visited is None:
                visited = set()
            
            if token_id in visited:
                return 0  # Cycle detected
            
            visited.add(token_id)
            
            token = next((t for t in tokens if t.id == token_id), None)
            if token is None or token.head == 0:
                return 0
            
            return 1 + get_depth(token.head, visited)
        
        return max(get_depth(t.id) for t in tokens)
    
    def _count_non_projective(self, tokens: List[Token]) -> int:
        """Count non-projective arcs"""
        count = 0
        
        for token in tokens:
            if token.head == 0:
                continue
            
            # Get arc span
            arc_start = min(token.id, token.head)
            arc_end = max(token.id, token.head)
            
            # Check for crossing arcs
            for other in tokens:
                if other.head == 0 or other.id == token.id:
                    continue
                
                other_start = min(other.id, other.head)
                other_end = max(other.id, other.head)
                
                # Check if arcs cross
                if (arc_start < other_start < arc_end < other_end or
                    other_start < arc_start < other_end < arc_end):
                    count += 1
                    break
        
        return count
    
    def compare(self, treebank1: List[Sentence], treebank2: List[Sentence]) -> Dict:
        """Compare two treebanks"""
        stats1 = self.calculate(treebank1)
        stats2 = self.calculate(treebank2)
        
        comparison = {
            "treebank1": stats1,
            "treebank2": stats2,
            "differences": {}
        }
        
        # Calculate differences
        for key in ["sentence_count", "token_count", "unique_lemmas", "avg_sentence_length", "avg_arc_length"]:
            if key in stats1 and key in stats2:
                comparison["differences"][key] = stats2.get(key, 0) - stats1.get(key, 0)
        
        return comparison


# ============================================================================
# TREEBANK MANAGER
# ============================================================================

class TreebankManager:
    """Main treebank management class"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.parser = TreebankParser()
        self.exporter = TreebankExporter()
        self.statistics = TreebankStatistics()
    
    def import_treebank(self, content: str, format: str, source_info: Dict = None) -> List[Sentence]:
        """Import treebank from various formats"""
        if format.lower() == "proiel":
            sentences = self.parser.parse_proiel_xml(content)
        elif format.lower() == "conllu":
            sentences = self.parser.parse_conllu(content)
        elif format.lower() == "agdt":
            sentences = self.parser.parse_agdt(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Imported {len(sentences)} sentences from {format} format")
        return sentences
    
    def export_treebank(self, sentences: List[Sentence], format: str, source_info: Dict = None) -> str:
        """Export treebank to various formats"""
        if format.lower() == "conllu":
            return self.exporter.to_conllu(sentences)
        elif format.lower() == "proiel":
            return self.exporter.to_proiel_xml(sentences, source_info)
        elif format.lower() == "json":
            return self.exporter.to_json(sentences)
        elif format.lower() == "csv":
            return self.exporter.to_csv(sentences)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def convert(self, content: str, from_format: str, to_format: str, source_info: Dict = None) -> str:
        """Convert between treebank formats"""
        sentences = self.import_treebank(content, from_format, source_info)
        return self.export_treebank(sentences, to_format, source_info)
    
    def get_statistics(self, sentences: List[Sentence]) -> Dict:
        """Get treebank statistics"""
        return self.statistics.calculate(sentences)
    
    def validate(self, sentences: List[Sentence]) -> List[Dict]:
        """Validate treebank data"""
        errors = []
        
        for sent_idx, sentence in enumerate(sentences):
            # Check for empty sentences
            if not sentence.tokens:
                errors.append({
                    "sentence": sentence.id,
                    "type": "empty_sentence",
                    "message": "Sentence has no tokens"
                })
                continue
            
            # Check token IDs
            token_ids = [t.id for t in sentence.tokens]
            if len(token_ids) != len(set(token_ids)):
                errors.append({
                    "sentence": sentence.id,
                    "type": "duplicate_id",
                    "message": "Duplicate token IDs found"
                })
            
            # Check head references
            for token in sentence.tokens:
                if token.head > 0 and token.head not in token_ids:
                    errors.append({
                        "sentence": sentence.id,
                        "token": token.id,
                        "type": "invalid_head",
                        "message": f"Token {token.id} references non-existent head {token.head}"
                    })
            
            # Check for cycles
            if self._has_cycle(sentence.tokens):
                errors.append({
                    "sentence": sentence.id,
                    "type": "cycle",
                    "message": "Dependency cycle detected"
                })
            
            # Check for multiple roots
            roots = [t for t in sentence.tokens if t.head == 0]
            if len(roots) > 1:
                errors.append({
                    "sentence": sentence.id,
                    "type": "multiple_roots",
                    "message": f"Multiple roots found: {[r.id for r in roots]}"
                })
            elif len(roots) == 0:
                errors.append({
                    "sentence": sentence.id,
                    "type": "no_root",
                    "message": "No root token found"
                })
        
        return errors
    
    def _has_cycle(self, tokens: List[Token]) -> bool:
        """Check for cycles in dependency tree"""
        for token in tokens:
            visited = set()
            current = token
            
            while current.head > 0:
                if current.id in visited:
                    return True
                visited.add(current.id)
                current = next((t for t in tokens if t.id == current.head), None)
                if current is None:
                    break
        
        return False
    
    def query(self, sentences: List[Sentence], query: Dict) -> List[Tuple[Sentence, List[Token]]]:
        """Query treebank with pattern matching"""
        results = []
        
        for sentence in sentences:
            matches = self._match_query(sentence.tokens, query)
            if matches:
                results.append((sentence, matches))
        
        return results
    
    def _match_query(self, tokens: List[Token], query: Dict) -> List[Token]:
        """Match query pattern against tokens"""
        matches = []
        
        for token in tokens:
            match = True
            
            # Check form
            if "form" in query:
                if not re.match(query["form"], token.form, re.IGNORECASE):
                    match = False
            
            # Check lemma
            if "lemma" in query and match:
                if not re.match(query["lemma"], token.lemma, re.IGNORECASE):
                    match = False
            
            # Check POS
            if "pos" in query and match:
                if token.pos != query["pos"]:
                    match = False
            
            # Check deprel
            if "deprel" in query and match:
                if token.deprel != query["deprel"]:
                    match = False
            
            # Check morphological features
            if "morphology" in query and match:
                for feat, value in query["morphology"].items():
                    if token.morphology.get(feat) != value:
                        match = False
                        break
            
            if match:
                matches.append(token)
        
        return matches
