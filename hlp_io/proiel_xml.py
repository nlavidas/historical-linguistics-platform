"""
HLP IO PROIEL XML - PROIEL 3.0 XML Format Support

This module provides comprehensive support for reading and writing
PROIEL XML format, the standard format used by the PROIEL treebank
and Syntacticus annotation platform.

Supports:
- PROIEL XML 3.0 schema
- Full document structure (source, div, sentence, token)
- Morphological annotation
- Syntactic annotation (dependency relations, slashes)
- Information structure annotation
- Lemma and POS information

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Union, 
    Iterator, TextIO, BinaryIO
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    from lxml import etree as lxml_etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

from hlp_core.models import (
    Corpus, Document, Sentence, Token,
    MorphologicalFeatures, SyntacticRelation,
    Language, Period, Genre, AnnotationStatus,
    PartOfSpeech, DependencyRelation, Case, Number, Gender,
    Person, Tense, Mood, Voice, Degree,
    InformationStructure, InformationStatusType, TopicFocusType,
    SourceMetadata, DiachronicStage
)

logger = logging.getLogger(__name__)


PROIEL_NAMESPACE = "http://proiel.eu/xml/3.0"
PROIEL_SCHEMA_LOCATION = "http://proiel.eu/xml/3.0/proiel-3.0.xsd"

PROIEL_POS_MAP = {
    "A-": PartOfSpeech.ADJ,
    "Df": PartOfSpeech.ADV,
    "Dq": PartOfSpeech.ADV,
    "S-": PartOfSpeech.ADP,
    "C-": PartOfSpeech.CCONJ,
    "G-": PartOfSpeech.SCONJ,
    "I-": PartOfSpeech.INTJ,
    "Ma": PartOfSpeech.NUM,
    "Mo": PartOfSpeech.NUM,
    "Nb": PartOfSpeech.NOUN,
    "Nc": PartOfSpeech.NOUN,
    "Ne": PartOfSpeech.PROPN,
    "Np": PartOfSpeech.PROPN,
    "Pc": PartOfSpeech.PRON,
    "Pd": PartOfSpeech.PRON,
    "Pi": PartOfSpeech.PRON,
    "Pk": PartOfSpeech.PRON,
    "Pp": PartOfSpeech.PRON,
    "Pq": PartOfSpeech.PRON,
    "Pr": PartOfSpeech.PRON,
    "Ps": PartOfSpeech.PRON,
    "Pt": PartOfSpeech.PRON,
    "Px": PartOfSpeech.PRON,
    "Py": PartOfSpeech.PRON,
    "R-": PartOfSpeech.ADP,
    "V-": PartOfSpeech.VERB,
    "X-": PartOfSpeech.X,
    "F-": PartOfSpeech.PUNCT,
}

PROIEL_RELATION_MAP = {
    "PRED": DependencyRelation.PRED,
    "SUB": DependencyRelation.SUB,
    "OBJ": DependencyRelation.OBJ_PROIEL,
    "OBL": DependencyRelation.OBL_PROIEL,
    "ADV": DependencyRelation.ADV,
    "ATR": DependencyRelation.ATR,
    "APOS": DependencyRelation.APOS,
    "AUX": DependencyRelation.AUX_PROIEL,
    "COMP": DependencyRelation.COMP,
    "EXPL": DependencyRelation.EXPL_PROIEL,
    "NARG": DependencyRelation.NARG,
    "NONSUB": DependencyRelation.NONSUB,
    "PARPRED": DependencyRelation.PARPRED,
    "PER": DependencyRelation.PER,
    "PART": DependencyRelation.PART_PROIEL,
    "XADV": DependencyRelation.XADV,
    "XOBJ": DependencyRelation.XOBJ,
    "XSUB": DependencyRelation.XSUB,
    "VOC": DependencyRelation.VOC,
    "AG": DependencyRelation.AG,
}

UD_TO_PROIEL_RELATION = {
    DependencyRelation.NSUBJ: "SUB",
    DependencyRelation.OBJ: "OBJ",
    DependencyRelation.IOBJ: "OBJ",
    DependencyRelation.OBL: "OBL",
    DependencyRelation.ADVMOD: "ADV",
    DependencyRelation.AMOD: "ATR",
    DependencyRelation.NMOD: "ATR",
    DependencyRelation.DET: "ATR",
    DependencyRelation.APPOS: "APOS",
    DependencyRelation.AUX: "AUX",
    DependencyRelation.COP: "AUX",
    DependencyRelation.CCOMP: "COMP",
    DependencyRelation.XCOMP: "XOBJ",
    DependencyRelation.ADVCL: "XADV",
    DependencyRelation.VOCATIVE: "VOC",
    DependencyRelation.EXPL: "EXPL",
    DependencyRelation.MARK: "AUX",
    DependencyRelation.CASE: "AUX",
    DependencyRelation.CC: "AUX",
    DependencyRelation.CONJ: "PRED",
    DependencyRelation.PUNCT: "AUX",
    DependencyRelation.ROOT: "PRED",
}

PROIEL_INFO_STATUS_MAP = {
    "new": InformationStatusType.NEW,
    "old": InformationStatusType.OLD,
    "acc_gen": InformationStatusType.ACCESSIBLE,
    "acc_sit": InformationStatusType.ACCESSIBLE,
    "acc_inf": InformationStatusType.INFERABLE,
    "kind": InformationStatusType.NEW,
    "non_spec": InformationStatusType.NEW,
    "quant": InformationStatusType.NEW,
}


@dataclass
class PROIELSource:
    """PROIEL source metadata"""
    id: str
    language: str
    
    title: Optional[str] = None
    author: Optional[str] = None
    citation_part: Optional[str] = None
    
    edition: Optional[str] = None
    editor: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    
    url: Optional[str] = None
    license: Optional[str] = None
    
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PROIELDiv:
    """PROIEL division (chapter, section, etc.)"""
    id: Optional[str] = None
    title: Optional[str] = None
    presentation_before: Optional[str] = None
    presentation_after: Optional[str] = None
    alignment_id: Optional[str] = None


class PROIELReader:
    """Reader for PROIEL XML format"""
    
    def __init__(self, use_lxml: bool = True):
        self.use_lxml = use_lxml and HAS_LXML
        self._id_map: Dict[str, int] = {}
        self._current_source: Optional[PROIELSource] = None
    
    def read_file(self, file_path: Union[str, Path]) -> Corpus:
        """Read PROIEL XML file and return Corpus"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.read_string(content, source_path=str(file_path))
    
    def read_string(self, xml_string: str, source_path: Optional[str] = None) -> Corpus:
        """Read PROIEL XML string and return Corpus"""
        self._id_map.clear()
        
        if self.use_lxml:
            root = lxml_etree.fromstring(xml_string.encode("utf-8"))
        else:
            root = ET.fromstring(xml_string)
        
        corpus = self._parse_proiel_root(root, source_path)
        return corpus
    
    def _parse_proiel_root(self, root: Any, source_path: Optional[str] = None) -> Corpus:
        """Parse PROIEL root element"""
        ns = {"p": PROIEL_NAMESPACE}
        
        schema_version = root.get("schema-version", "3.0")
        export_time = root.get("export-time")
        
        corpus_id = f"proiel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if source_path:
            corpus_id = Path(source_path).stem
        
        corpus = Corpus(
            id=corpus_id,
            name=corpus_id,
            description=f"PROIEL corpus (schema {schema_version})",
            metadata={
                "schema_version": schema_version,
                "export_time": export_time,
                "source_path": source_path
            }
        )
        
        if self.use_lxml:
            sources = root.findall(".//p:source", ns)
        else:
            sources = root.findall(".//{%s}source" % PROIEL_NAMESPACE)
            if not sources:
                sources = root.findall(".//source")
        
        for source_elem in sources:
            document = self._parse_source(source_elem, ns)
            if document:
                corpus.documents.append(document)
                if document.language and document.language not in corpus.languages:
                    corpus.languages.append(document.language)
        
        if corpus.languages:
            corpus.language = corpus.languages[0]
        
        return corpus
    
    def _parse_source(self, source_elem: Any, ns: Dict[str, str]) -> Optional[Document]:
        """Parse PROIEL source element"""
        source_id = source_elem.get("id", "")
        language_code = source_elem.get("language", "grc")
        
        try:
            language = Language(language_code)
        except ValueError:
            language = Language.ANCIENT_GREEK
        
        self._current_source = PROIELSource(
            id=source_id,
            language=language_code
        )
        
        title_elem = self._find_child(source_elem, "title", ns)
        title = title_elem.text if title_elem is not None and title_elem.text else source_id
        
        author_elem = self._find_child(source_elem, "author", ns)
        author = author_elem.text if author_elem is not None and author_elem.text else None
        
        citation_elem = self._find_child(source_elem, "citation-part", ns)
        citation = citation_elem.text if citation_elem is not None and citation_elem.text else None
        
        document = Document(
            id=source_id,
            title=title,
            author=author,
            language=language,
            proiel_id=source_id,
            source=SourceMetadata(
                source_type="proiel",
                source_id=source_id
            ),
            metadata={
                "citation_part": citation,
                "proiel_language": language_code
            }
        )
        
        if self.use_lxml:
            divs = source_elem.findall(".//p:div", ns)
        else:
            divs = source_elem.findall(".//{%s}div" % PROIEL_NAMESPACE)
            if not divs:
                divs = source_elem.findall(".//div")
        
        sentence_index = 0
        for div_elem in divs:
            div_info = self._parse_div(div_elem, ns)
            
            if self.use_lxml:
                sentences = div_elem.findall("p:sentence", ns)
            else:
                sentences = div_elem.findall("{%s}sentence" % PROIEL_NAMESPACE)
                if not sentences:
                    sentences = div_elem.findall("sentence")
            
            for sent_elem in sentences:
                sentence = self._parse_sentence(sent_elem, ns, sentence_index, div_info)
                if sentence:
                    sentence.document_id = document.id
                    document.sentences.append(sentence)
                    sentence_index += 1
        
        if not divs:
            if self.use_lxml:
                sentences = source_elem.findall(".//p:sentence", ns)
            else:
                sentences = source_elem.findall(".//{%s}sentence" % PROIEL_NAMESPACE)
                if not sentences:
                    sentences = source_elem.findall(".//sentence")
            
            for sent_elem in sentences:
                sentence = self._parse_sentence(sent_elem, ns, sentence_index, None)
                if sentence:
                    sentence.document_id = document.id
                    document.sentences.append(sentence)
                    sentence_index += 1
        
        return document
    
    def _parse_div(self, div_elem: Any, ns: Dict[str, str]) -> PROIELDiv:
        """Parse PROIEL div element"""
        return PROIELDiv(
            id=div_elem.get("id"),
            title=div_elem.get("title"),
            presentation_before=div_elem.get("presentation-before"),
            presentation_after=div_elem.get("presentation-after"),
            alignment_id=div_elem.get("alignment-id")
        )
    
    def _parse_sentence(
        self, 
        sent_elem: Any, 
        ns: Dict[str, str],
        sentence_index: int,
        div_info: Optional[PROIELDiv]
    ) -> Optional[Sentence]:
        """Parse PROIEL sentence element"""
        sent_id = sent_elem.get("id", f"s{sentence_index}")
        status = sent_elem.get("status", "")
        presentation_before = sent_elem.get("presentation-before", "")
        presentation_after = sent_elem.get("presentation-after", "")
        
        sentence = Sentence(
            id=sent_id,
            sentence_index=sentence_index,
            proiel_id=sent_id,
            proiel_status=status,
            proiel_presentation_before=presentation_before,
            proiel_presentation_after=presentation_after,
            metadata={}
        )
        
        if div_info:
            sentence.metadata["div_id"] = div_info.id
            sentence.metadata["div_title"] = div_info.title
        
        if self.use_lxml:
            tokens = sent_elem.findall("p:token", ns)
        else:
            tokens = sent_elem.findall("{%s}token" % PROIEL_NAMESPACE)
            if not tokens:
                tokens = sent_elem.findall("token")
        
        self._id_map.clear()
        for idx, token_elem in enumerate(tokens, start=1):
            proiel_id = token_elem.get("id", "")
            self._id_map[proiel_id] = idx
        
        for idx, token_elem in enumerate(tokens, start=1):
            token = self._parse_token(token_elem, ns, idx)
            if token:
                sentence.tokens.append(token)
        
        if sentence.tokens:
            text_parts = []
            for token in sentence.tokens:
                if token.proiel_presentation_before:
                    text_parts.append(token.proiel_presentation_before)
                text_parts.append(token.form)
                if token.proiel_presentation_after:
                    text_parts.append(token.proiel_presentation_after)
            sentence.text = "".join(text_parts).strip()
        
        if status == "reviewed":
            sentence.annotation_status = AnnotationStatus.REVIEWED
        elif status == "annotated":
            sentence.annotation_status = AnnotationStatus.MANUAL
        else:
            sentence.annotation_status = AnnotationStatus.AUTOMATIC
        
        return sentence
    
    def _parse_token(self, token_elem: Any, ns: Dict[str, str], token_index: int) -> Optional[Token]:
        """Parse PROIEL token element"""
        proiel_id = token_elem.get("id", "")
        form = token_elem.get("form", "")
        lemma = token_elem.get("lemma")
        pos = token_elem.get("part-of-speech", "")
        morph = token_elem.get("morphology", "")
        
        head_id_str = token_elem.get("head-id", "")
        relation = token_elem.get("relation", "")
        slashes = token_elem.get("slashes", "")
        
        presentation_before = token_elem.get("presentation-before", "")
        presentation_after = token_elem.get("presentation-after", "")
        empty_token_sort = token_elem.get("empty-token-sort")
        citation_part = token_elem.get("citation-part")
        
        antecedent_id = token_elem.get("antecedent-id")
        information_status = token_elem.get("information-status")
        contrast = token_elem.get("contrast")
        
        morphology = self._parse_morphology(pos, morph)
        
        syntax = None
        if head_id_str or relation:
            head_id = self._id_map.get(head_id_str, 0)
            
            try:
                dep_relation = PROIEL_RELATION_MAP.get(relation, DependencyRelation.DEP)
            except (ValueError, KeyError):
                dep_relation = DependencyRelation.DEP
            
            slash_list = []
            if slashes:
                slash_list = [s.strip() for s in slashes.split(",") if s.strip()]
            
            syntax = SyntacticRelation(
                head_id=head_id,
                relation=dep_relation,
                proiel_relation=relation,
                proiel_slashes=slash_list
            )
        
        info_structure = None
        if information_status or contrast or antecedent_id:
            info_status = PROIEL_INFO_STATUS_MAP.get(
                information_status, 
                InformationStatusType.UNKNOWN
            )
            info_structure = InformationStructure(
                info_status=info_status,
                contrast=contrast == "true" if contrast else False,
                antecedent_id=self._id_map.get(antecedent_id) if antecedent_id else None
            )
        
        is_empty = empty_token_sort is not None
        
        token = Token(
            id=token_index,
            form=form,
            lemma=lemma,
            morphology=morphology,
            syntax=syntax,
            info_structure=info_structure,
            proiel_id=proiel_id,
            proiel_presentation_before=presentation_before,
            proiel_presentation_after=presentation_after,
            proiel_empty_token_sort=empty_token_sort,
            proiel_citation_part=citation_part,
            proiel_antecedent_id=antecedent_id,
            proiel_information_status=information_status,
            proiel_contrast=contrast,
            is_empty=is_empty
        )
        
        return token
    
    def _parse_morphology(self, pos: str, morph: str) -> MorphologicalFeatures:
        """Parse PROIEL morphology string"""
        features = MorphologicalFeatures()
        features.proiel_morph = morph
        
        if pos:
            features.pos = PROIEL_POS_MAP.get(pos, PartOfSpeech.X)
        
        if morph and len(morph) >= 1:
            person_map = {"1": Person.FIRST, "2": Person.SECOND, "3": Person.THIRD}
            if morph[0] in person_map:
                features.person = person_map[morph[0]]
        
        if morph and len(morph) >= 2:
            number_map = {"s": Number.SINGULAR, "d": Number.DUAL, "p": Number.PLURAL}
            if morph[1] in number_map:
                features.number = number_map[morph[1]]
        
        if morph and len(morph) >= 3:
            tense_map = {
                "p": Tense.PRESENT, "i": Tense.IMPERFECT, "f": Tense.FUTURE,
                "a": Tense.AORIST, "r": Tense.PERFECT, "l": Tense.PLUPERFECT,
                "t": Tense.FUTURE_PERFECT
            }
            if morph[2] in tense_map:
                features.tense = tense_map[morph[2]]
        
        if morph and len(morph) >= 4:
            mood_map = {
                "i": Mood.INDICATIVE, "s": Mood.SUBJUNCTIVE, "o": Mood.OPTATIVE,
                "m": Mood.IMPERATIVE, "n": Mood.INFINITIVE, "p": Mood.PARTICIPLE,
                "g": Mood.GERUND, "d": Mood.GERUNDIVE, "u": Mood.SUPINE
            }
            if morph[3] in mood_map:
                features.mood = mood_map[morph[3]]
        
        if morph and len(morph) >= 5:
            voice_map = {
                "a": Voice.ACTIVE, "m": Voice.MIDDLE, 
                "p": Voice.PASSIVE, "e": Voice.MIDDLE_PASSIVE
            }
            if morph[4] in voice_map:
                features.voice = voice_map[morph[4]]
        
        if morph and len(morph) >= 6:
            gender_map = {"m": Gender.MASCULINE, "f": Gender.FEMININE, "n": Gender.NEUTER}
            if morph[5] in gender_map:
                features.gender = gender_map[morph[5]]
        
        if morph and len(morph) >= 7:
            case_map = {
                "n": Case.NOMINATIVE, "g": Case.GENITIVE, "d": Case.DATIVE,
                "a": Case.ACCUSATIVE, "v": Case.VOCATIVE, "b": Case.ABLATIVE,
                "l": Case.LOCATIVE, "i": Case.INSTRUMENTAL
            }
            if morph[6] in case_map:
                features.case = case_map[morph[6]]
        
        if morph and len(morph) >= 8:
            degree_map = {"p": Degree.POSITIVE, "c": Degree.COMPARATIVE, "s": Degree.SUPERLATIVE}
            if morph[7] in degree_map:
                features.degree = degree_map[morph[7]]
        
        return features
    
    def _find_child(self, elem: Any, tag: str, ns: Dict[str, str]) -> Optional[Any]:
        """Find child element by tag"""
        if self.use_lxml:
            return elem.find(f"p:{tag}", ns)
        else:
            result = elem.find(f"{{{PROIEL_NAMESPACE}}}{tag}")
            if result is None:
                result = elem.find(tag)
            return result


class PROIELWriter:
    """Writer for PROIEL XML format"""
    
    def __init__(self, pretty_print: bool = True, use_lxml: bool = True):
        self.pretty_print = pretty_print
        self.use_lxml = use_lxml and HAS_LXML
        self._token_id_counter = 0
    
    def write_file(self, corpus: Corpus, file_path: Union[str, Path]):
        """Write corpus to PROIEL XML file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        xml_string = self.write_string(corpus)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
    
    def write_string(self, corpus: Corpus) -> str:
        """Write corpus to PROIEL XML string"""
        self._token_id_counter = 0
        
        if self.use_lxml:
            return self._write_lxml(corpus)
        else:
            return self._write_etree(corpus)
    
    def _write_lxml(self, corpus: Corpus) -> str:
        """Write using lxml"""
        nsmap = {None: PROIEL_NAMESPACE}
        
        root = lxml_etree.Element(
            "proiel",
            nsmap=nsmap,
            attrib={
                "schema-version": "3.0",
                "export-time": datetime.now().isoformat()
            }
        )
        
        for document in corpus.documents:
            source_elem = self._create_source_element_lxml(document, nsmap)
            root.append(source_elem)
        
        xml_bytes = lxml_etree.tostring(
            root,
            encoding="unicode",
            pretty_print=self.pretty_print,
            xml_declaration=True
        )
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_bytes}'
    
    def _write_etree(self, corpus: Corpus) -> str:
        """Write using standard ElementTree"""
        root = ET.Element("proiel")
        root.set("xmlns", PROIEL_NAMESPACE)
        root.set("schema-version", "3.0")
        root.set("export-time", datetime.now().isoformat())
        
        for document in corpus.documents:
            source_elem = self._create_source_element_etree(document)
            root.append(source_elem)
        
        xml_string = ET.tostring(root, encoding="unicode")
        
        if self.pretty_print:
            dom = minidom.parseString(xml_string)
            xml_string = dom.toprettyxml(indent="  ")
            lines = xml_string.split("\n")
            xml_string = "\n".join(line for line in lines if line.strip())
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_string}'
    
    def _create_source_element_lxml(self, document: Document, nsmap: Dict) -> Any:
        """Create source element using lxml"""
        source = lxml_etree.Element("source", nsmap=nsmap)
        source.set("id", document.proiel_id or document.id)
        source.set("language", document.language.value if document.language else "grc")
        
        title = lxml_etree.SubElement(source, "title")
        title.text = document.title
        
        if document.author:
            author = lxml_etree.SubElement(source, "author")
            author.text = document.author
        
        div = lxml_etree.SubElement(source, "div")
        
        for sentence in document.sentences:
            sent_elem = self._create_sentence_element_lxml(sentence, nsmap)
            div.append(sent_elem)
        
        return source
    
    def _create_source_element_etree(self, document: Document) -> ET.Element:
        """Create source element using ElementTree"""
        source = ET.Element("source")
        source.set("id", document.proiel_id or document.id)
        source.set("language", document.language.value if document.language else "grc")
        
        title = ET.SubElement(source, "title")
        title.text = document.title
        
        if document.author:
            author = ET.SubElement(source, "author")
            author.text = document.author
        
        div = ET.SubElement(source, "div")
        
        for sentence in document.sentences:
            sent_elem = self._create_sentence_element_etree(sentence)
            div.append(sent_elem)
        
        return source
    
    def _create_sentence_element_lxml(self, sentence: Sentence, nsmap: Dict) -> Any:
        """Create sentence element using lxml"""
        sent = lxml_etree.Element("sentence", nsmap=nsmap)
        sent.set("id", sentence.proiel_id or sentence.id)
        
        if sentence.proiel_status:
            sent.set("status", sentence.proiel_status)
        elif sentence.annotation_status == AnnotationStatus.REVIEWED:
            sent.set("status", "reviewed")
        elif sentence.annotation_status == AnnotationStatus.MANUAL:
            sent.set("status", "annotated")
        
        if sentence.proiel_presentation_before:
            sent.set("presentation-before", sentence.proiel_presentation_before)
        if sentence.proiel_presentation_after:
            sent.set("presentation-after", sentence.proiel_presentation_after)
        
        token_id_map = {}
        for token in sentence.tokens:
            self._token_id_counter += 1
            token_id = token.proiel_id or f"t{self._token_id_counter}"
            token_id_map[token.id] = token_id
        
        for token in sentence.tokens:
            token_elem = self._create_token_element_lxml(token, token_id_map, nsmap)
            sent.append(token_elem)
        
        return sent
    
    def _create_sentence_element_etree(self, sentence: Sentence) -> ET.Element:
        """Create sentence element using ElementTree"""
        sent = ET.Element("sentence")
        sent.set("id", sentence.proiel_id or sentence.id)
        
        if sentence.proiel_status:
            sent.set("status", sentence.proiel_status)
        elif sentence.annotation_status == AnnotationStatus.REVIEWED:
            sent.set("status", "reviewed")
        elif sentence.annotation_status == AnnotationStatus.MANUAL:
            sent.set("status", "annotated")
        
        if sentence.proiel_presentation_before:
            sent.set("presentation-before", sentence.proiel_presentation_before)
        if sentence.proiel_presentation_after:
            sent.set("presentation-after", sentence.proiel_presentation_after)
        
        token_id_map = {}
        for token in sentence.tokens:
            self._token_id_counter += 1
            token_id = token.proiel_id or f"t{self._token_id_counter}"
            token_id_map[token.id] = token_id
        
        for token in sentence.tokens:
            token_elem = self._create_token_element_etree(token, token_id_map)
            sent.append(token_elem)
        
        return sent
    
    def _create_token_element_lxml(
        self, 
        token: Token, 
        token_id_map: Dict[int, str],
        nsmap: Dict
    ) -> Any:
        """Create token element using lxml"""
        tok = lxml_etree.Element("token", nsmap=nsmap)
        
        tok.set("id", token_id_map.get(token.id, token.proiel_id or f"t{token.id}"))
        tok.set("form", token.form)
        
        if token.lemma:
            tok.set("lemma", token.lemma)
        
        if token.morphology and token.morphology.pos:
            pos = self._get_proiel_pos(token.morphology.pos)
            if pos:
                tok.set("part-of-speech", pos)
        
        if token.morphology:
            morph = token.morphology.to_proiel_string()
            if morph and morph != "-----------":
                tok.set("morphology", morph)
        
        if token.syntax:
            if token.syntax.head_id > 0:
                head_token_id = token_id_map.get(token.syntax.head_id, "")
                if head_token_id:
                    tok.set("head-id", head_token_id)
            
            relation = token.syntax.proiel_relation
            if not relation and token.syntax.relation:
                relation = self._get_proiel_relation(token.syntax.relation)
            if relation:
                tok.set("relation", relation)
            
            if token.syntax.proiel_slashes:
                tok.set("slashes", ",".join(token.syntax.proiel_slashes))
        
        if token.proiel_presentation_before:
            tok.set("presentation-before", token.proiel_presentation_before)
        if token.proiel_presentation_after:
            tok.set("presentation-after", token.proiel_presentation_after)
        
        if token.proiel_empty_token_sort:
            tok.set("empty-token-sort", token.proiel_empty_token_sort)
        
        if token.proiel_citation_part:
            tok.set("citation-part", token.proiel_citation_part)
        
        if token.proiel_antecedent_id:
            tok.set("antecedent-id", token.proiel_antecedent_id)
        
        if token.proiel_information_status:
            tok.set("information-status", token.proiel_information_status)
        
        if token.proiel_contrast:
            tok.set("contrast", token.proiel_contrast)
        
        return tok
    
    def _create_token_element_etree(
        self, 
        token: Token, 
        token_id_map: Dict[int, str]
    ) -> ET.Element:
        """Create token element using ElementTree"""
        tok = ET.Element("token")
        
        tok.set("id", token_id_map.get(token.id, token.proiel_id or f"t{token.id}"))
        tok.set("form", token.form)
        
        if token.lemma:
            tok.set("lemma", token.lemma)
        
        if token.morphology and token.morphology.pos:
            pos = self._get_proiel_pos(token.morphology.pos)
            if pos:
                tok.set("part-of-speech", pos)
        
        if token.morphology:
            morph = token.morphology.to_proiel_string()
            if morph and morph != "-----------":
                tok.set("morphology", morph)
        
        if token.syntax:
            if token.syntax.head_id > 0:
                head_token_id = token_id_map.get(token.syntax.head_id, "")
                if head_token_id:
                    tok.set("head-id", head_token_id)
            
            relation = token.syntax.proiel_relation
            if not relation and token.syntax.relation:
                relation = self._get_proiel_relation(token.syntax.relation)
            if relation:
                tok.set("relation", relation)
            
            if token.syntax.proiel_slashes:
                tok.set("slashes", ",".join(token.syntax.proiel_slashes))
        
        if token.proiel_presentation_before:
            tok.set("presentation-before", token.proiel_presentation_before)
        if token.proiel_presentation_after:
            tok.set("presentation-after", token.proiel_presentation_after)
        
        if token.proiel_empty_token_sort:
            tok.set("empty-token-sort", token.proiel_empty_token_sort)
        
        if token.proiel_citation_part:
            tok.set("citation-part", token.proiel_citation_part)
        
        if token.proiel_antecedent_id:
            tok.set("antecedent-id", token.proiel_antecedent_id)
        
        if token.proiel_information_status:
            tok.set("information-status", token.proiel_information_status)
        
        if token.proiel_contrast:
            tok.set("contrast", token.proiel_contrast)
        
        return tok
    
    def _get_proiel_pos(self, pos: PartOfSpeech) -> Optional[str]:
        """Convert UD POS to PROIEL POS"""
        reverse_map = {v: k for k, v in PROIEL_POS_MAP.items()}
        return reverse_map.get(pos)
    
    def _get_proiel_relation(self, relation: DependencyRelation) -> Optional[str]:
        """Convert UD relation to PROIEL relation"""
        return UD_TO_PROIEL_RELATION.get(relation)


class PROIELValidator:
    """Validator for PROIEL XML format"""
    
    def __init__(self):
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate PROIEL XML file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self._errors.append(f"File not found: {file_path}")
            return False
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.validate_string(content)
    
    def validate_string(self, xml_string: str) -> bool:
        """Validate PROIEL XML string"""
        self._errors.clear()
        self._warnings.clear()
        
        try:
            if HAS_LXML:
                root = lxml_etree.fromstring(xml_string.encode("utf-8"))
            else:
                root = ET.fromstring(xml_string)
        except Exception as e:
            self._errors.append(f"XML parsing error: {e}")
            return False
        
        self._validate_structure(root)
        
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
    
    def _validate_structure(self, root: Any):
        """Validate XML structure"""
        ns = {"p": PROIEL_NAMESPACE}
        
        schema_version = root.get("schema-version")
        if not schema_version:
            self._warnings.append("Missing schema-version attribute")
        elif schema_version not in ("3.0", "2.1", "2.0"):
            self._warnings.append(f"Unknown schema version: {schema_version}")
        
        if HAS_LXML:
            sources = root.findall(".//p:source", ns)
        else:
            sources = root.findall(".//{%s}source" % PROIEL_NAMESPACE)
            if not sources:
                sources = root.findall(".//source")
        
        if not sources:
            self._errors.append("No source elements found")
            return
        
        for source in sources:
            self._validate_source(source, ns)
    
    def _validate_source(self, source: Any, ns: Dict[str, str]):
        """Validate source element"""
        source_id = source.get("id")
        if not source_id:
            self._errors.append("Source missing id attribute")
        
        language = source.get("language")
        if not language:
            self._warnings.append(f"Source {source_id} missing language attribute")
        
        if HAS_LXML:
            sentences = source.findall(".//p:sentence", ns)
        else:
            sentences = source.findall(".//{%s}sentence" % PROIEL_NAMESPACE)
            if not sentences:
                sentences = source.findall(".//sentence")
        
        if not sentences:
            self._warnings.append(f"Source {source_id} has no sentences")
        
        for sentence in sentences:
            self._validate_sentence(sentence, ns, source_id)
    
    def _validate_sentence(self, sentence: Any, ns: Dict[str, str], source_id: str):
        """Validate sentence element"""
        sent_id = sentence.get("id")
        if not sent_id:
            self._errors.append(f"Sentence in {source_id} missing id attribute")
        
        if HAS_LXML:
            tokens = sentence.findall("p:token", ns)
        else:
            tokens = sentence.findall("{%s}token" % PROIEL_NAMESPACE)
            if not tokens:
                tokens = sentence.findall("token")
        
        if not tokens:
            self._warnings.append(f"Sentence {sent_id} has no tokens")
        
        token_ids = set()
        for token in tokens:
            token_id = token.get("id")
            if not token_id:
                self._errors.append(f"Token in sentence {sent_id} missing id")
            elif token_id in token_ids:
                self._errors.append(f"Duplicate token id {token_id} in sentence {sent_id}")
            else:
                token_ids.add(token_id)
            
            form = token.get("form")
            if not form and not token.get("empty-token-sort"):
                self._warnings.append(f"Token {token_id} in sentence {sent_id} missing form")
            
            head_id = token.get("head-id")
            relation = token.get("relation")
            
            if head_id and head_id not in token_ids and head_id != token_id:
                pass
            
            if head_id and not relation:
                self._warnings.append(f"Token {token_id} has head-id but no relation")
    
    def _validate_document(self, document: Document):
        """Validate document object"""
        if not document.id:
            self._errors.append("Document missing ID")
        
        if not document.title:
            self._warnings.append(f"Document {document.id} missing title")
        
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
            
            if token.syntax and token.syntax.head_id > 0:
                if token.syntax.head_id not in token_ids and token.syntax.head_id != token.id:
                    pass
    
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
        lines = ["PROIEL Validation Report", "=" * 40]
        
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


def parse_proiel_file(file_path: Union[str, Path]) -> Corpus:
    """Parse PROIEL XML file and return Corpus"""
    reader = PROIELReader()
    return reader.read_file(file_path)


def parse_proiel_string(xml_string: str) -> Corpus:
    """Parse PROIEL XML string and return Corpus"""
    reader = PROIELReader()
    return reader.read_string(xml_string)


def write_proiel_file(corpus: Corpus, file_path: Union[str, Path]):
    """Write Corpus to PROIEL XML file"""
    writer = PROIELWriter()
    writer.write_file(corpus, file_path)


def write_proiel_string(corpus: Corpus) -> str:
    """Write Corpus to PROIEL XML string"""
    writer = PROIELWriter()
    return writer.write_string(corpus)


def validate_proiel_file(file_path: Union[str, Path]) -> Tuple[bool, List[str], List[str]]:
    """Validate PROIEL XML file"""
    validator = PROIELValidator()
    is_valid = validator.validate_file(file_path)
    return is_valid, validator.errors, validator.warnings


def validate_proiel_string(xml_string: str) -> Tuple[bool, List[str], List[str]]:
    """Validate PROIEL XML string"""
    validator = PROIELValidator()
    is_valid = validator.validate_string(xml_string)
    return is_valid, validator.errors, validator.warnings
