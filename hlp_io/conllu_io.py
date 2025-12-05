"""
HLP IO CoNLL-U - CoNLL-U Format Support

This module provides comprehensive support for reading and writing
CoNLL-U format, the standard format for Universal Dependencies treebanks.

Supports:
- CoNLL-U 2.0 format
- Multi-word tokens
- Empty nodes
- Enhanced dependencies
- Sentence-level metadata
- Document boundaries

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Union, 
    Iterator, TextIO, Generator
)
from dataclasses import dataclass, field
from datetime import datetime

from hlp_core.models import (
    Corpus, Document, Sentence, Token,
    MorphologicalFeatures, SyntacticRelation,
    Language, Period, Genre, AnnotationStatus,
    PartOfSpeech, DependencyRelation, Case, Number, Gender,
    Person, Tense, Mood, Voice, Degree,
    SourceMetadata
)

logger = logging.getLogger(__name__)


CONLLU_FIELD_COUNT = 10
CONLLU_FIELDS = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]

UPOS_MAP = {
    "ADJ": PartOfSpeech.ADJ,
    "ADP": PartOfSpeech.ADP,
    "ADV": PartOfSpeech.ADV,
    "AUX": PartOfSpeech.AUX,
    "CCONJ": PartOfSpeech.CCONJ,
    "DET": PartOfSpeech.DET,
    "INTJ": PartOfSpeech.INTJ,
    "NOUN": PartOfSpeech.NOUN,
    "NUM": PartOfSpeech.NUM,
    "PART": PartOfSpeech.PART,
    "PRON": PartOfSpeech.PRON,
    "PROPN": PartOfSpeech.PROPN,
    "PUNCT": PartOfSpeech.PUNCT,
    "SCONJ": PartOfSpeech.SCONJ,
    "SYM": PartOfSpeech.SYM,
    "VERB": PartOfSpeech.VERB,
    "X": PartOfSpeech.X,
}

DEPREL_MAP = {
    "acl": DependencyRelation.ACL,
    "acl:relcl": DependencyRelation.ACL_RELCL,
    "advcl": DependencyRelation.ADVCL,
    "advmod": DependencyRelation.ADVMOD,
    "amod": DependencyRelation.AMOD,
    "appos": DependencyRelation.APPOS,
    "aux": DependencyRelation.AUX,
    "aux:pass": DependencyRelation.AUX_PASS,
    "case": DependencyRelation.CASE,
    "cc": DependencyRelation.CC,
    "ccomp": DependencyRelation.CCOMP,
    "clf": DependencyRelation.CLF,
    "compound": DependencyRelation.COMPOUND,
    "conj": DependencyRelation.CONJ,
    "cop": DependencyRelation.COP,
    "csubj": DependencyRelation.CSUBJ,
    "csubj:pass": DependencyRelation.CSUBJ_PASS,
    "dep": DependencyRelation.DEP,
    "det": DependencyRelation.DET,
    "discourse": DependencyRelation.DISCOURSE,
    "dislocated": DependencyRelation.DISLOCATED,
    "expl": DependencyRelation.EXPL,
    "fixed": DependencyRelation.FIXED,
    "flat": DependencyRelation.FLAT,
    "flat:name": DependencyRelation.FLAT_NAME,
    "goeswith": DependencyRelation.GOESWITH,
    "iobj": DependencyRelation.IOBJ,
    "list": DependencyRelation.LIST,
    "mark": DependencyRelation.MARK,
    "nmod": DependencyRelation.NMOD,
    "nsubj": DependencyRelation.NSUBJ,
    "nsubj:pass": DependencyRelation.NSUBJ_PASS,
    "nummod": DependencyRelation.NUMMOD,
    "obj": DependencyRelation.OBJ,
    "obl": DependencyRelation.OBL,
    "orphan": DependencyRelation.ORPHAN,
    "parataxis": DependencyRelation.PARATAXIS,
    "punct": DependencyRelation.PUNCT,
    "reparandum": DependencyRelation.REPARANDUM,
    "root": DependencyRelation.ROOT,
    "vocative": DependencyRelation.VOCATIVE,
    "xcomp": DependencyRelation.XCOMP,
}


@dataclass
class MultiWordToken:
    """Multi-word token span"""
    start_id: int
    end_id: int
    form: str
    misc: Dict[str, str] = field(default_factory=dict)


@dataclass
class EmptyNode:
    """Empty node (enhanced dependencies)"""
    main_id: int
    sub_id: int
    form: str
    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: Dict[str, str] = field(default_factory=dict)
    deps: List[Tuple[str, str]] = field(default_factory=list)
    misc: Dict[str, str] = field(default_factory=dict)


class CoNLLUReader:
    """Reader for CoNLL-U format"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self._current_doc_id: Optional[str] = None
        self._current_sent_id: Optional[str] = None
        self._line_number = 0
    
    def read_file(self, file_path: Union[str, Path]) -> Corpus:
        """Read CoNLL-U file and return Corpus"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.read_string(content, source_path=str(file_path))
    
    def read_string(self, conllu_string: str, source_path: Optional[str] = None) -> Corpus:
        """Read CoNLL-U string and return Corpus"""
        self._line_number = 0
        self._current_doc_id = None
        
        corpus_id = f"conllu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if source_path:
            corpus_id = Path(source_path).stem
        
        corpus = Corpus(
            id=corpus_id,
            name=corpus_id,
            description="CoNLL-U corpus",
            metadata={"source_path": source_path}
        )
        
        current_document: Optional[Document] = None
        sentence_index = 0
        
        for sentence in self._iter_sentences(conllu_string):
            if sentence.newdoc:
                if current_document:
                    corpus.documents.append(current_document)
                
                doc_id = sentence.newdoc if isinstance(sentence.newdoc, str) else f"doc_{len(corpus.documents) + 1}"
                current_document = Document(
                    id=doc_id,
                    title=doc_id,
                    source=SourceMetadata(source_type="conllu")
                )
                sentence_index = 0
            
            if current_document is None:
                current_document = Document(
                    id="doc_1",
                    title="Document 1",
                    source=SourceMetadata(source_type="conllu")
                )
            
            sentence.document_id = current_document.id
            sentence.sentence_index = sentence_index
            current_document.sentences.append(sentence)
            sentence_index += 1
        
        if current_document:
            corpus.documents.append(current_document)
        
        return corpus
    
    def _iter_sentences(self, conllu_string: str) -> Generator[Sentence, None, None]:
        """Iterate over sentences in CoNLL-U string"""
        lines = conllu_string.split("\n")
        current_lines: List[str] = []
        
        for line in lines:
            self._line_number += 1
            
            if line.strip() == "":
                if current_lines:
                    sentence = self._parse_sentence(current_lines)
                    if sentence:
                        yield sentence
                    current_lines = []
            else:
                current_lines.append(line)
        
        if current_lines:
            sentence = self._parse_sentence(current_lines)
            if sentence:
                yield sentence
    
    def _parse_sentence(self, lines: List[str]) -> Optional[Sentence]:
        """Parse a single sentence from lines"""
        metadata: Dict[str, Any] = {}
        token_lines: List[str] = []
        mwt_lines: List[str] = []
        empty_lines: List[str] = []
        
        for line in lines:
            if line.startswith("#"):
                key, value = self._parse_comment(line)
                if key:
                    metadata[key] = value
            else:
                fields = line.split("\t")
                if len(fields) >= 1:
                    token_id = fields[0]
                    if "-" in token_id:
                        mwt_lines.append(line)
                    elif "." in token_id:
                        empty_lines.append(line)
                    else:
                        token_lines.append(line)
        
        sent_id = metadata.get("sent_id", f"s{self._line_number}")
        text = metadata.get("text", "")
        newdoc = metadata.get("newdoc")
        newpar = metadata.get("newpar")
        
        sentence = Sentence(
            id=sent_id,
            sent_id=sent_id,
            text=text,
            newdoc=newdoc,
            newpar=newpar,
            metadata=metadata
        )
        
        mwt_map = self._parse_mwt_lines(mwt_lines)
        
        for token_line in token_lines:
            token = self._parse_token_line(token_line, mwt_map)
            if token:
                sentence.tokens.append(token)
        
        if not sentence.text and sentence.tokens:
            sentence.text = " ".join(t.form for t in sentence.tokens if not t.is_empty)
        
        return sentence
    
    def _parse_comment(self, line: str) -> Tuple[Optional[str], Any]:
        """Parse comment line"""
        line = line[1:].strip()
        
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "newdoc":
                return "newdoc", value if value else True
            elif key == "newpar":
                return "newpar", value if value else True
            
            return key, value
        
        return None, line
    
    def _parse_mwt_lines(self, mwt_lines: List[str]) -> Dict[int, MultiWordToken]:
        """Parse multi-word token lines"""
        mwt_map: Dict[int, MultiWordToken] = {}
        
        for line in mwt_lines:
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            
            id_range = fields[0]
            form = fields[1]
            
            if "-" in id_range:
                start, end = id_range.split("-")
                try:
                    start_id = int(start)
                    end_id = int(end)
                    
                    misc = {}
                    if len(fields) >= 10 and fields[9] != "_":
                        misc = self._parse_misc(fields[9])
                    
                    mwt = MultiWordToken(
                        start_id=start_id,
                        end_id=end_id,
                        form=form,
                        misc=misc
                    )
                    
                    for i in range(start_id, end_id + 1):
                        mwt_map[i] = mwt
                except ValueError:
                    pass
        
        return mwt_map
    
    def _parse_token_line(
        self, 
        line: str, 
        mwt_map: Dict[int, MultiWordToken]
    ) -> Optional[Token]:
        """Parse a single token line"""
        fields = line.split("\t")
        
        if len(fields) != CONLLU_FIELD_COUNT:
            if self.strict:
                raise ValueError(f"Invalid field count at line {self._line_number}: expected {CONLLU_FIELD_COUNT}, got {len(fields)}")
            while len(fields) < CONLLU_FIELD_COUNT:
                fields.append("_")
        
        try:
            token_id = int(fields[0])
        except ValueError:
            if self.strict:
                raise ValueError(f"Invalid token ID at line {self._line_number}: {fields[0]}")
            return None
        
        form = fields[1]
        lemma = fields[2] if fields[2] != "_" else None
        upos = fields[3] if fields[3] != "_" else None
        xpos = fields[4] if fields[4] != "_" else None
        feats_str = fields[5]
        head_str = fields[6]
        deprel = fields[7] if fields[7] != "_" else None
        deps_str = fields[8]
        misc_str = fields[9]
        
        morphology = self._parse_morphology(upos, xpos, feats_str)
        
        syntax = None
        if head_str != "_":
            try:
                head_id = int(head_str)
                relation = DEPREL_MAP.get(deprel, DependencyRelation.DEP) if deprel else DependencyRelation.DEP
                
                enhanced_deps = []
                if deps_str != "_":
                    enhanced_deps = self._parse_enhanced_deps(deps_str)
                
                syntax = SyntacticRelation(
                    head_id=head_id,
                    relation=relation,
                    enhanced_deps=enhanced_deps
                )
            except ValueError:
                pass
        
        misc = self._parse_misc(misc_str) if misc_str != "_" else {}
        
        is_multiword = token_id in mwt_map
        multiword_id = None
        if is_multiword:
            mwt = mwt_map[token_id]
            multiword_id = f"{mwt.start_id}-{mwt.end_id}"
        
        span_start = misc.get("SpanStart")
        span_end = misc.get("SpanEnd")
        
        token = Token(
            id=token_id,
            form=form,
            lemma=lemma,
            morphology=morphology,
            syntax=syntax,
            misc=misc,
            is_multiword=is_multiword,
            multiword_id=multiword_id,
            span_start=int(span_start) if span_start else None,
            span_end=int(span_end) if span_end else None
        )
        
        return token
    
    def _parse_morphology(
        self, 
        upos: Optional[str], 
        xpos: Optional[str], 
        feats_str: str
    ) -> MorphologicalFeatures:
        """Parse morphological features"""
        features = MorphologicalFeatures()
        
        if upos:
            features.pos = UPOS_MAP.get(upos, PartOfSpeech.X)
        
        if xpos:
            features.proiel_morph = xpos
        
        if feats_str and feats_str != "_":
            feats = self._parse_feats(feats_str)
            
            case_map = {
                "Nom": Case.NOMINATIVE, "Gen": Case.GENITIVE,
                "Dat": Case.DATIVE, "Acc": Case.ACCUSATIVE,
                "Voc": Case.VOCATIVE, "Abl": Case.ABLATIVE,
                "Loc": Case.LOCATIVE, "Ins": Case.INSTRUMENTAL
            }
            if "Case" in feats:
                features.case = case_map.get(feats["Case"])
            
            number_map = {"Sing": Number.SINGULAR, "Dual": Number.DUAL, "Plur": Number.PLURAL}
            if "Number" in feats:
                features.number = number_map.get(feats["Number"])
            
            gender_map = {"Masc": Gender.MASCULINE, "Fem": Gender.FEMININE, "Neut": Gender.NEUTER}
            if "Gender" in feats:
                features.gender = gender_map.get(feats["Gender"])
            
            person_map = {"1": Person.FIRST, "2": Person.SECOND, "3": Person.THIRD}
            if "Person" in feats:
                features.person = person_map.get(feats["Person"])
            
            tense_map = {
                "Pres": Tense.PRESENT, "Past": Tense.PAST, "Fut": Tense.FUTURE,
                "Imp": Tense.IMPERFECT, "Pqp": Tense.PLUPERFECT
            }
            if "Tense" in feats:
                features.tense = tense_map.get(feats["Tense"])
            
            mood_map = {
                "Ind": Mood.INDICATIVE, "Sub": Mood.SUBJUNCTIVE,
                "Imp": Mood.IMPERATIVE, "Opt": Mood.OPTATIVE,
                "Inf": Mood.INFINITIVE, "Part": Mood.PARTICIPLE
            }
            if "Mood" in feats:
                features.mood = mood_map.get(feats["Mood"])
            
            voice_map = {"Act": Voice.ACTIVE, "Pass": Voice.PASSIVE, "Mid": Voice.MIDDLE}
            if "Voice" in feats:
                features.voice = voice_map.get(feats["Voice"])
            
            degree_map = {"Pos": Degree.POSITIVE, "Cmp": Degree.COMPARATIVE, "Sup": Degree.SUPERLATIVE}
            if "Degree" in feats:
                features.degree = degree_map.get(feats["Degree"])
        
        return features
    
    def _parse_feats(self, feats_str: str) -> Dict[str, str]:
        """Parse feature string"""
        feats = {}
        if feats_str and feats_str != "_":
            for feat in feats_str.split("|"):
                if "=" in feat:
                    key, value = feat.split("=", 1)
                    feats[key] = value
        return feats
    
    def _parse_enhanced_deps(self, deps_str: str) -> List[Tuple[int, str]]:
        """Parse enhanced dependencies string"""
        deps = []
        if deps_str and deps_str != "_":
            for dep in deps_str.split("|"):
                if ":" in dep:
                    head_str, rel = dep.split(":", 1)
                    try:
                        head_id = int(head_str.split(".")[0])
                        deps.append((head_id, rel))
                    except ValueError:
                        pass
        return deps
    
    def _parse_misc(self, misc_str: str) -> Dict[str, str]:
        """Parse MISC field"""
        misc = {}
        if misc_str and misc_str != "_":
            for item in misc_str.split("|"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    misc[key] = value
                else:
                    misc[item] = "true"
        return misc


class CoNLLUWriter:
    """Writer for CoNLL-U format"""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
    
    def write_file(self, corpus: Corpus, file_path: Union[str, Path]):
        """Write corpus to CoNLL-U file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        conllu_string = self.write_string(corpus)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conllu_string)
    
    def write_string(self, corpus: Corpus) -> str:
        """Write corpus to CoNLL-U string"""
        lines: List[str] = []
        
        for doc_idx, document in enumerate(corpus.documents):
            for sent_idx, sentence in enumerate(document.sentences):
                is_first_in_doc = sent_idx == 0
                sent_lines = self._write_sentence(sentence, is_first_in_doc, document.id)
                lines.extend(sent_lines)
                lines.append("")
        
        return "\n".join(lines)
    
    def write_sentence(self, sentence: Sentence) -> str:
        """Write single sentence to CoNLL-U string"""
        lines = self._write_sentence(sentence, False, None)
        return "\n".join(lines)
    
    def _write_sentence(
        self, 
        sentence: Sentence, 
        is_first_in_doc: bool,
        doc_id: Optional[str]
    ) -> List[str]:
        """Write sentence to list of lines"""
        lines: List[str] = []
        
        if self.include_metadata:
            if is_first_in_doc and doc_id:
                lines.append(f"# newdoc id = {doc_id}")
            
            if sentence.newdoc:
                if isinstance(sentence.newdoc, str):
                    lines.append(f"# newdoc id = {sentence.newdoc}")
                else:
                    lines.append("# newdoc")
            
            if sentence.newpar:
                if isinstance(sentence.newpar, str):
                    lines.append(f"# newpar id = {sentence.newpar}")
                else:
                    lines.append("# newpar")
            
            sent_id = sentence.sent_id or sentence.id
            lines.append(f"# sent_id = {sent_id}")
            
            if sentence.text:
                lines.append(f"# text = {sentence.text}")
            
            for key, value in sentence.metadata.items():
                if key not in ("sent_id", "text", "newdoc", "newpar"):
                    lines.append(f"# {key} = {value}")
        
        mwt_tokens: Dict[str, List[Token]] = {}
        for token in sentence.tokens:
            if token.is_multiword and token.multiword_id:
                if token.multiword_id not in mwt_tokens:
                    mwt_tokens[token.multiword_id] = []
                mwt_tokens[token.multiword_id].append(token)
        
        written_mwt: set = set()
        
        for token in sentence.tokens:
            if token.is_multiword and token.multiword_id and token.multiword_id not in written_mwt:
                mwt_line = self._write_mwt_line(token.multiword_id, mwt_tokens.get(token.multiword_id, []))
                if mwt_line:
                    lines.append(mwt_line)
                written_mwt.add(token.multiword_id)
            
            token_line = self._write_token_line(token)
            lines.append(token_line)
        
        return lines
    
    def _write_mwt_line(self, mwt_id: str, tokens: List[Token]) -> Optional[str]:
        """Write multi-word token line"""
        if not tokens:
            return None
        
        form = "".join(t.form for t in tokens)
        
        fields = [
            mwt_id,
            form,
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_"
        ]
        
        return "\t".join(fields)
    
    def _write_token_line(self, token: Token) -> str:
        """Write single token line"""
        token_id = str(token.id)
        form = token.form
        lemma = token.lemma or "_"
        
        upos = "_"
        if token.morphology and token.morphology.pos:
            upos = token.morphology.pos.value
        
        xpos = "_"
        if token.morphology and token.morphology.proiel_morph:
            xpos = token.morphology.proiel_morph
        
        feats = "_"
        if token.morphology:
            feats_str = token.morphology.to_ud_string()
            if feats_str:
                feats = feats_str
        
        head = "_"
        deprel = "_"
        if token.syntax:
            head = str(token.syntax.head_id)
            if token.syntax.relation:
                deprel = token.syntax.relation.value
        
        deps = "_"
        if token.syntax and token.syntax.enhanced_deps:
            deps_parts = [f"{h}:{r}" for h, r in token.syntax.enhanced_deps]
            deps = "|".join(deps_parts)
        
        misc = "_"
        if token.misc:
            misc_parts = []
            for key, value in token.misc.items():
                if value == "true":
                    misc_parts.append(key)
                else:
                    misc_parts.append(f"{key}={value}")
            if misc_parts:
                misc = "|".join(misc_parts)
        
        fields = [token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc]
        return "\t".join(fields)


class CoNLLUValidator:
    """Validator for CoNLL-U format"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate CoNLL-U file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self._errors.append(f"File not found: {file_path}")
            return False
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.validate_string(content)
    
    def validate_string(self, conllu_string: str) -> bool:
        """Validate CoNLL-U string"""
        self._errors.clear()
        self._warnings.clear()
        
        lines = conllu_string.split("\n")
        line_number = 0
        sentence_count = 0
        current_sentence_lines: List[Tuple[int, str]] = []
        
        for line in lines:
            line_number += 1
            
            if line.strip() == "":
                if current_sentence_lines:
                    self._validate_sentence(current_sentence_lines, sentence_count)
                    sentence_count += 1
                    current_sentence_lines = []
            else:
                current_sentence_lines.append((line_number, line))
        
        if current_sentence_lines:
            self._validate_sentence(current_sentence_lines, sentence_count)
            sentence_count += 1
        
        if sentence_count == 0:
            self._errors.append("No sentences found in file")
        
        return len(self._errors) == 0
    
    def validate_corpus(self, corpus: Corpus) -> bool:
        """Validate corpus object"""
        self._errors.clear()
        self._warnings.clear()
        
        if not corpus.id:
            self._errors.append("Corpus missing ID")
        
        if not corpus.documents:
            self._warnings.append("Corpus has no documents")
        
        for doc in corpus.documents:
            self._validate_document(doc)
        
        return len(self._errors) == 0
    
    def _validate_sentence(self, lines: List[Tuple[int, str]], sent_num: int):
        """Validate a single sentence"""
        has_sent_id = False
        has_text = False
        token_ids: List[int] = []
        expected_id = 1
        
        for line_num, line in lines:
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    has_sent_id = True
                elif line.startswith("# text"):
                    has_text = True
            else:
                fields = line.split("\t")
                
                if len(fields) != CONLLU_FIELD_COUNT:
                    self._errors.append(
                        f"Line {line_num}: Invalid field count (expected {CONLLU_FIELD_COUNT}, got {len(fields)})"
                    )
                    continue
                
                token_id_str = fields[0]
                
                if "-" in token_id_str:
                    continue
                elif "." in token_id_str:
                    continue
                else:
                    try:
                        token_id = int(token_id_str)
                        
                        if token_id != expected_id:
                            self._warnings.append(
                                f"Line {line_num}: Token ID {token_id} does not match expected {expected_id}"
                            )
                        
                        token_ids.append(token_id)
                        expected_id = token_id + 1
                        
                        self._validate_token_fields(fields, line_num, token_ids)
                        
                    except ValueError:
                        self._errors.append(f"Line {line_num}: Invalid token ID '{token_id_str}'")
        
        if not has_sent_id:
            self._warnings.append(f"Sentence {sent_num + 1}: Missing sent_id comment")
        
        if not has_text:
            self._warnings.append(f"Sentence {sent_num + 1}: Missing text comment")
        
        if not token_ids:
            self._errors.append(f"Sentence {sent_num + 1}: No tokens found")
    
    def _validate_token_fields(self, fields: List[str], line_num: int, token_ids: List[int]):
        """Validate token fields"""
        form = fields[1]
        if not form:
            self._errors.append(f"Line {line_num}: Empty FORM field")
        
        upos = fields[3]
        if upos != "_" and upos not in UPOS_MAP:
            self._warnings.append(f"Line {line_num}: Unknown UPOS tag '{upos}'")
        
        feats = fields[5]
        if feats != "_":
            for feat in feats.split("|"):
                if "=" not in feat:
                    self._warnings.append(f"Line {line_num}: Invalid feature format '{feat}'")
        
        head = fields[6]
        if head != "_":
            try:
                head_id = int(head)
                if head_id < 0:
                    self._errors.append(f"Line {line_num}: Negative HEAD value")
                elif head_id > 0 and head_id not in token_ids:
                    pass
            except ValueError:
                self._errors.append(f"Line {line_num}: Invalid HEAD value '{head}'")
        
        deprel = fields[7]
        if deprel != "_" and deprel.split(":")[0] not in DEPREL_MAP:
            self._warnings.append(f"Line {line_num}: Unknown DEPREL '{deprel}'")
    
    def _validate_document(self, document: Document):
        """Validate document object"""
        if not document.id:
            self._errors.append("Document missing ID")
        
        if not document.sentences:
            self._warnings.append(f"Document {document.id} has no sentences")
        
        for sentence in document.sentences:
            self._validate_sentence_object(sentence, document.id)
    
    def _validate_sentence_object(self, sentence: Sentence, doc_id: str):
        """Validate sentence object"""
        if not sentence.id:
            self._errors.append(f"Sentence in {doc_id} missing ID")
        
        if not sentence.tokens:
            self._warnings.append(f"Sentence {sentence.id} has no tokens")
        
        token_ids = set()
        for token in sentence.tokens:
            if token.id in token_ids:
                self._errors.append(f"Duplicate token id {token.id} in sentence {sentence.id}")
            token_ids.add(token.id)
    
    @property
    def errors(self) -> List[str]:
        """Get validation errors"""
        return self._errors.copy()
    
    @property
    def warnings(self) -> List[str]:
        """Get validation warnings"""
        return self._warnings.copy()
    
    def get_report(self) -> str:
        """Get validation report"""
        lines = ["CoNLL-U Validation Report", "=" * 40]
        
        if self._errors:
            lines.append(f"\nErrors ({len(self._errors)}):")
            for error in self._errors:
                lines.append(f"  - {error}")
        
        if self._warnings:
            lines.append(f"\nWarnings ({len(self._warnings)}):")
            for warning in self._warnings:
                lines.append(f"  - {warning}")
        
        if not self._errors and not self._warnings:
            lines.append("\nNo issues found.")
        
        return "\n".join(lines)


def parse_conllu_file(file_path: Union[str, Path]) -> Corpus:
    """Parse CoNLL-U file and return Corpus"""
    reader = CoNLLUReader()
    return reader.read_file(file_path)


def parse_conllu_string(conllu_string: str) -> Corpus:
    """Parse CoNLL-U string and return Corpus"""
    reader = CoNLLUReader()
    return reader.read_string(conllu_string)


def write_conllu_file(corpus: Corpus, file_path: Union[str, Path]):
    """Write Corpus to CoNLL-U file"""
    writer = CoNLLUWriter()
    writer.write_file(corpus, file_path)


def write_conllu_string(corpus: Corpus) -> str:
    """Write Corpus to CoNLL-U string"""
    writer = CoNLLUWriter()
    return writer.write_string(corpus)


def validate_conllu_file(file_path: Union[str, Path]) -> Tuple[bool, List[str], List[str]]:
    """Validate CoNLL-U file"""
    validator = CoNLLUValidator()
    is_valid = validator.validate_file(file_path)
    return is_valid, validator.errors, validator.warnings


def validate_conllu_string(conllu_string: str) -> Tuple[bool, List[str], List[str]]:
    """Validate CoNLL-U string"""
    validator = CoNLLUValidator()
    is_valid = validator.validate_string(conllu_string)
    return is_valid, validator.errors, validator.warnings
