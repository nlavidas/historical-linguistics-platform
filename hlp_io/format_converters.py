"""
HLP IO Format Converters - Conversion Between Linguistic Data Formats

This module provides utilities for converting between different
linguistic data formats including PROIEL XML, CoNLL-U, JSON, and others.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Union, 
    Iterator, Callable, Type
)
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from hlp_core.models import (
    Corpus, Document, Sentence, Token,
    MorphologicalFeatures, SyntacticRelation,
    Language, Period, Genre, AnnotationStatus,
    PartOfSpeech, DependencyRelation,
    SemanticRole, NamedEntity, InformationStructure,
    ValencyFrame, ValencyPattern,
    SourceMetadata
)

from hlp_io.proiel_xml import (
    PROIELReader, PROIELWriter,
    PROIEL_POS_MAP, PROIEL_RELATION_MAP,
    UD_TO_PROIEL_RELATION
)

from hlp_io.conllu_io import (
    CoNLLUReader, CoNLLUWriter,
    UPOS_MAP, DEPREL_MAP
)

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    PROIEL_XML = "proiel_xml"
    CONLLU = "conllu"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    TEI_XML = "tei_xml"
    PLAIN_TEXT = "plain_text"


class InputFormat(Enum):
    """Supported input formats"""
    PROIEL_XML = "proiel_xml"
    CONLLU = "conllu"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    AUTO = "auto"


@dataclass
class ConversionOptions:
    """Options for format conversion"""
    preserve_ids: bool = True
    preserve_metadata: bool = True
    preserve_proiel_features: bool = True
    preserve_enhanced_deps: bool = True
    
    include_empty_nodes: bool = True
    include_multiword_tokens: bool = True
    
    normalize_relations: bool = False
    normalize_pos: bool = False
    
    add_sentence_text: bool = True
    add_document_boundaries: bool = True
    
    pretty_print: bool = True
    
    custom_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Result of a format conversion"""
    success: bool
    output_path: Optional[str] = None
    output_string: Optional[str] = None
    
    source_format: Optional[InputFormat] = None
    target_format: Optional[OutputFormat] = None
    
    documents_converted: int = 0
    sentences_converted: int = 0
    tokens_converted: int = 0
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    conversion_time_ms: float = 0.0


class FormatConverter:
    """Main format converter class"""
    
    def __init__(self, options: Optional[ConversionOptions] = None):
        self.options = options or ConversionOptions()
        self._proiel_reader = PROIELReader()
        self._proiel_writer = PROIELWriter(pretty_print=self.options.pretty_print)
        self._conllu_reader = CoNLLUReader()
        self._conllu_writer = CoNLLUWriter(include_metadata=self.options.preserve_metadata)
    
    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: InputFormat = InputFormat.AUTO,
        output_format: Optional[OutputFormat] = None
    ) -> ConversionResult:
        """Convert file from one format to another"""
        import time
        start_time = time.time()
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        result = ConversionResult(success=False)
        
        if not input_path.exists():
            result.errors.append(f"Input file not found: {input_path}")
            return result
        
        if input_format == InputFormat.AUTO:
            input_format = self._detect_format(input_path)
        
        if output_format is None:
            output_format = self._infer_output_format(output_path)
        
        result.source_format = input_format
        result.target_format = output_format
        
        try:
            corpus = self._read_input(input_path, input_format)
            
            if corpus is None:
                result.errors.append("Failed to read input file")
                return result
            
            self._write_output(corpus, output_path, output_format)
            
            result.success = True
            result.output_path = str(output_path)
            result.documents_converted = len(corpus.documents)
            result.sentences_converted = sum(len(d.sentences) for d in corpus.documents)
            result.tokens_converted = sum(
                sum(len(s.tokens) for s in d.sentences) 
                for d in corpus.documents
            )
            
        except Exception as e:
            result.errors.append(f"Conversion error: {str(e)}")
            logger.exception("Conversion failed")
        
        result.conversion_time_ms = (time.time() - start_time) * 1000
        return result
    
    def convert_string(
        self,
        input_string: str,
        input_format: InputFormat,
        output_format: OutputFormat
    ) -> ConversionResult:
        """Convert string from one format to another"""
        import time
        start_time = time.time()
        
        result = ConversionResult(
            success=False,
            source_format=input_format,
            target_format=output_format
        )
        
        try:
            corpus = self._read_input_string(input_string, input_format)
            
            if corpus is None:
                result.errors.append("Failed to parse input string")
                return result
            
            output_string = self._write_output_string(corpus, output_format)
            
            result.success = True
            result.output_string = output_string
            result.documents_converted = len(corpus.documents)
            result.sentences_converted = sum(len(d.sentences) for d in corpus.documents)
            result.tokens_converted = sum(
                sum(len(s.tokens) for s in d.sentences) 
                for d in corpus.documents
            )
            
        except Exception as e:
            result.errors.append(f"Conversion error: {str(e)}")
            logger.exception("Conversion failed")
        
        result.conversion_time_ms = (time.time() - start_time) * 1000
        return result
    
    def convert_corpus(
        self,
        corpus: Corpus,
        output_format: OutputFormat
    ) -> str:
        """Convert corpus to output format string"""
        return self._write_output_string(corpus, output_format)
    
    def _detect_format(self, file_path: Path) -> InputFormat:
        """Detect input format from file extension and content"""
        suffix = file_path.suffix.lower()
        
        if suffix in (".xml", ".proiel"):
            return InputFormat.PROIEL_XML
        elif suffix in (".conllu", ".conll"):
            return InputFormat.CONLLU
        elif suffix == ".json":
            return InputFormat.JSON
        elif suffix == ".txt":
            return InputFormat.PLAIN_TEXT
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            
            if first_line.startswith("<?xml") or first_line.startswith("<proiel"):
                return InputFormat.PROIEL_XML
            elif first_line.startswith("#") or "\t" in first_line:
                return InputFormat.CONLLU
            elif first_line.startswith("{"):
                return InputFormat.JSON
        except Exception:
            pass
        
        return InputFormat.PLAIN_TEXT
    
    def _infer_output_format(self, file_path: Path) -> OutputFormat:
        """Infer output format from file extension"""
        suffix = file_path.suffix.lower()
        
        format_map = {
            ".xml": OutputFormat.PROIEL_XML,
            ".proiel": OutputFormat.PROIEL_XML,
            ".conllu": OutputFormat.CONLLU,
            ".conll": OutputFormat.CONLLU,
            ".json": OutputFormat.JSON,
            ".csv": OutputFormat.CSV,
            ".tsv": OutputFormat.TSV,
            ".txt": OutputFormat.PLAIN_TEXT,
            ".tei": OutputFormat.TEI_XML,
        }
        
        return format_map.get(suffix, OutputFormat.CONLLU)
    
    def _read_input(self, file_path: Path, input_format: InputFormat) -> Optional[Corpus]:
        """Read input file"""
        if input_format == InputFormat.PROIEL_XML:
            return self._proiel_reader.read_file(file_path)
        elif input_format == InputFormat.CONLLU:
            return self._conllu_reader.read_file(file_path)
        elif input_format == InputFormat.JSON:
            return self._read_json_file(file_path)
        elif input_format == InputFormat.PLAIN_TEXT:
            return self._read_plain_text_file(file_path)
        
        return None
    
    def _read_input_string(self, input_string: str, input_format: InputFormat) -> Optional[Corpus]:
        """Read input string"""
        if input_format == InputFormat.PROIEL_XML:
            return self._proiel_reader.read_string(input_string)
        elif input_format == InputFormat.CONLLU:
            return self._conllu_reader.read_string(input_string)
        elif input_format == InputFormat.JSON:
            return self._read_json_string(input_string)
        elif input_format == InputFormat.PLAIN_TEXT:
            return self._read_plain_text_string(input_string)
        
        return None
    
    def _write_output(self, corpus: Corpus, file_path: Path, output_format: OutputFormat):
        """Write output file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == OutputFormat.PROIEL_XML:
            self._proiel_writer.write_file(corpus, file_path)
        elif output_format == OutputFormat.CONLLU:
            self._conllu_writer.write_file(corpus, file_path)
        elif output_format == OutputFormat.JSON:
            self._write_json_file(corpus, file_path)
        elif output_format == OutputFormat.CSV:
            self._write_csv_file(corpus, file_path)
        elif output_format == OutputFormat.TSV:
            self._write_tsv_file(corpus, file_path)
        elif output_format == OutputFormat.PLAIN_TEXT:
            self._write_plain_text_file(corpus, file_path)
        elif output_format == OutputFormat.TEI_XML:
            self._write_tei_xml_file(corpus, file_path)
    
    def _write_output_string(self, corpus: Corpus, output_format: OutputFormat) -> str:
        """Write output string"""
        if output_format == OutputFormat.PROIEL_XML:
            return self._proiel_writer.write_string(corpus)
        elif output_format == OutputFormat.CONLLU:
            return self._conllu_writer.write_string(corpus)
        elif output_format == OutputFormat.JSON:
            return self._write_json_string(corpus)
        elif output_format == OutputFormat.CSV:
            return self._write_csv_string(corpus)
        elif output_format == OutputFormat.TSV:
            return self._write_tsv_string(corpus)
        elif output_format == OutputFormat.PLAIN_TEXT:
            return self._write_plain_text_string(corpus)
        elif output_format == OutputFormat.TEI_XML:
            return self._write_tei_xml_string(corpus)
        
        return ""
    
    def _read_json_file(self, file_path: Path) -> Corpus:
        """Read JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._json_to_corpus(data)
    
    def _read_json_string(self, json_string: str) -> Corpus:
        """Read JSON string"""
        data = json.loads(json_string)
        return self._json_to_corpus(data)
    
    def _json_to_corpus(self, data: Dict[str, Any]) -> Corpus:
        """Convert JSON data to Corpus"""
        corpus = Corpus(
            id=data.get("id", "corpus"),
            name=data.get("name", "Corpus"),
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
        
        if "language" in data:
            try:
                corpus.language = Language(data["language"])
            except ValueError:
                pass
        
        for doc_data in data.get("documents", []):
            document = self._json_to_document(doc_data)
            corpus.documents.append(document)
        
        return corpus
    
    def _json_to_document(self, data: Dict[str, Any]) -> Document:
        """Convert JSON data to Document"""
        document = Document(
            id=data.get("id", "doc"),
            title=data.get("title", "Document"),
            author=data.get("author"),
            metadata=data.get("metadata", {})
        )
        
        if "language" in data:
            try:
                document.language = Language(data["language"])
            except ValueError:
                pass
        
        for sent_data in data.get("sentences", []):
            sentence = self._json_to_sentence(sent_data)
            sentence.document_id = document.id
            document.sentences.append(sentence)
        
        return document
    
    def _json_to_sentence(self, data: Dict[str, Any]) -> Sentence:
        """Convert JSON data to Sentence"""
        sentence = Sentence(
            id=data.get("id", "sent"),
            text=data.get("text"),
            translation=data.get("translation"),
            metadata=data.get("metadata", {})
        )
        
        for tok_data in data.get("tokens", []):
            token = self._json_to_token(tok_data)
            sentence.tokens.append(token)
        
        return sentence
    
    def _json_to_token(self, data: Dict[str, Any]) -> Token:
        """Convert JSON data to Token"""
        morphology = None
        if "morphology" in data:
            morph_data = data["morphology"]
            morphology = MorphologicalFeatures()
            if "pos" in morph_data:
                try:
                    morphology.pos = PartOfSpeech(morph_data["pos"])
                except ValueError:
                    pass
        
        syntax = None
        if "syntax" in data:
            syn_data = data["syntax"]
            try:
                relation = DependencyRelation(syn_data.get("relation", "dep"))
            except ValueError:
                relation = DependencyRelation.DEP
            
            syntax = SyntacticRelation(
                head_id=syn_data.get("head_id", 0),
                relation=relation
            )
        
        return Token(
            id=data.get("id", 0),
            form=data.get("form", ""),
            lemma=data.get("lemma"),
            morphology=morphology,
            syntax=syntax,
            misc=data.get("misc", {})
        )
    
    def _write_json_file(self, corpus: Corpus, file_path: Path):
        """Write JSON file"""
        data = self._corpus_to_json(corpus)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2 if self.options.pretty_print else None, ensure_ascii=False)
    
    def _write_json_string(self, corpus: Corpus) -> str:
        """Write JSON string"""
        data = self._corpus_to_json(corpus)
        return json.dumps(data, indent=2 if self.options.pretty_print else None, ensure_ascii=False)
    
    def _corpus_to_json(self, corpus: Corpus) -> Dict[str, Any]:
        """Convert Corpus to JSON-serializable dict"""
        return {
            "id": corpus.id,
            "name": corpus.name,
            "description": corpus.description,
            "language": corpus.language.value if corpus.language else None,
            "version": corpus.version,
            "metadata": corpus.metadata,
            "documents": [self._document_to_json(d) for d in corpus.documents]
        }
    
    def _document_to_json(self, document: Document) -> Dict[str, Any]:
        """Convert Document to JSON-serializable dict"""
        return {
            "id": document.id,
            "title": document.title,
            "author": document.author,
            "language": document.language.value if document.language else None,
            "period": document.period.value if document.period else None,
            "genre": document.genre.value if document.genre else None,
            "metadata": document.metadata,
            "sentences": [self._sentence_to_json(s) for s in document.sentences]
        }
    
    def _sentence_to_json(self, sentence: Sentence) -> Dict[str, Any]:
        """Convert Sentence to JSON-serializable dict"""
        return {
            "id": sentence.id,
            "text": sentence.text,
            "translation": sentence.translation,
            "metadata": sentence.metadata,
            "tokens": [self._token_to_json(t) for t in sentence.tokens]
        }
    
    def _token_to_json(self, token: Token) -> Dict[str, Any]:
        """Convert Token to JSON-serializable dict"""
        result = {
            "id": token.id,
            "form": token.form,
            "lemma": token.lemma
        }
        
        if token.morphology:
            result["morphology"] = {
                "pos": token.morphology.pos.value if token.morphology.pos else None,
                "case": token.morphology.case.value if token.morphology.case else None,
                "number": token.morphology.number.value if token.morphology.number else None,
                "gender": token.morphology.gender.value if token.morphology.gender else None,
                "person": token.morphology.person.value if token.morphology.person else None,
                "tense": token.morphology.tense.value if token.morphology.tense else None,
                "mood": token.morphology.mood.value if token.morphology.mood else None,
                "voice": token.morphology.voice.value if token.morphology.voice else None,
            }
        
        if token.syntax:
            result["syntax"] = {
                "head_id": token.syntax.head_id,
                "relation": token.syntax.relation.value if token.syntax.relation else None
            }
        
        if token.misc:
            result["misc"] = token.misc
        
        return result
    
    def _read_plain_text_file(self, file_path: Path) -> Corpus:
        """Read plain text file"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._read_plain_text_string(content)
    
    def _read_plain_text_string(self, text: str) -> Corpus:
        """Read plain text string"""
        corpus = Corpus(
            id="plain_text",
            name="Plain Text Corpus"
        )
        
        document = Document(
            id="doc_1",
            title="Document 1"
        )
        
        paragraphs = text.split("\n\n")
        sentence_index = 0
        
        for para in paragraphs:
            lines = para.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    sentence = Sentence(
                        id=f"s{sentence_index + 1}",
                        text=line,
                        sentence_index=sentence_index,
                        document_id=document.id
                    )
                    
                    words = line.split()
                    for idx, word in enumerate(words, start=1):
                        token = Token(
                            id=idx,
                            form=word
                        )
                        sentence.tokens.append(token)
                    
                    document.sentences.append(sentence)
                    sentence_index += 1
        
        corpus.documents.append(document)
        return corpus
    
    def _write_plain_text_file(self, corpus: Corpus, file_path: Path):
        """Write plain text file"""
        text = self._write_plain_text_string(corpus)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    def _write_plain_text_string(self, corpus: Corpus) -> str:
        """Write plain text string"""
        lines = []
        for document in corpus.documents:
            for sentence in document.sentences:
                if sentence.text:
                    lines.append(sentence.text)
                else:
                    lines.append(" ".join(t.form for t in sentence.tokens))
            lines.append("")
        return "\n".join(lines)
    
    def _write_csv_file(self, corpus: Corpus, file_path: Path):
        """Write CSV file"""
        csv_string = self._write_csv_string(corpus)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_string)
    
    def _write_csv_string(self, corpus: Corpus) -> str:
        """Write CSV string"""
        return self._write_tabular_string(corpus, ",")
    
    def _write_tsv_file(self, corpus: Corpus, file_path: Path):
        """Write TSV file"""
        tsv_string = self._write_tsv_string(corpus)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tsv_string)
    
    def _write_tsv_string(self, corpus: Corpus) -> str:
        """Write TSV string"""
        return self._write_tabular_string(corpus, "\t")
    
    def _write_tabular_string(self, corpus: Corpus, delimiter: str) -> str:
        """Write tabular format string"""
        lines = []
        
        headers = ["doc_id", "sent_id", "token_id", "form", "lemma", "pos", "head", "deprel"]
        lines.append(delimiter.join(headers))
        
        for document in corpus.documents:
            for sentence in document.sentences:
                for token in sentence.tokens:
                    row = [
                        document.id,
                        sentence.id,
                        str(token.id),
                        token.form,
                        token.lemma or "_",
                        token.morphology.pos.value if token.morphology and token.morphology.pos else "_",
                        str(token.syntax.head_id) if token.syntax else "_",
                        token.syntax.relation.value if token.syntax and token.syntax.relation else "_"
                    ]
                    
                    escaped_row = []
                    for field in row:
                        if delimiter in field or '"' in field or '\n' in field:
                            field = '"' + field.replace('"', '""') + '"'
                        escaped_row.append(field)
                    
                    lines.append(delimiter.join(escaped_row))
        
        return "\n".join(lines)
    
    def _write_tei_xml_file(self, corpus: Corpus, file_path: Path):
        """Write TEI XML file"""
        xml_string = self._write_tei_xml_string(corpus)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
    
    def _write_tei_xml_string(self, corpus: Corpus) -> str:
        """Write TEI XML string (basic implementation)"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
            '  <teiHeader>',
            '    <fileDesc>',
            f'      <titleStmt><title>{corpus.name}</title></titleStmt>',
            '      <publicationStmt><p>Generated by HLP Platform</p></publicationStmt>',
            '      <sourceDesc><p>Converted from linguistic corpus</p></sourceDesc>',
            '    </fileDesc>',
            '  </teiHeader>',
            '  <text>',
            '    <body>'
        ]
        
        for document in corpus.documents:
            lines.append(f'      <div type="document" xml:id="{document.id}">')
            if document.title:
                lines.append(f'        <head>{self._escape_xml(document.title)}</head>')
            
            for sentence in document.sentences:
                lines.append(f'        <s xml:id="{sentence.id}">')
                for token in sentence.tokens:
                    pos = token.morphology.pos.value if token.morphology and token.morphology.pos else ""
                    lemma = token.lemma or ""
                    lines.append(f'          <w lemma="{self._escape_xml(lemma)}" pos="{pos}">{self._escape_xml(token.form)}</w>')
                lines.append('        </s>')
            
            lines.append('      </div>')
        
        lines.extend([
            '    </body>',
            '  </text>',
            '</TEI>'
        ])
        
        return "\n".join(lines)
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        if not text:
            return ""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def proiel_to_conllu(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    options: Optional[ConversionOptions] = None
) -> ConversionResult:
    """Convert PROIEL XML to CoNLL-U"""
    converter = FormatConverter(options)
    return converter.convert_file(
        input_path, 
        output_path,
        InputFormat.PROIEL_XML,
        OutputFormat.CONLLU
    )


def conllu_to_proiel(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    options: Optional[ConversionOptions] = None
) -> ConversionResult:
    """Convert CoNLL-U to PROIEL XML"""
    converter = FormatConverter(options)
    return converter.convert_file(
        input_path,
        output_path,
        InputFormat.CONLLU,
        OutputFormat.PROIEL_XML
    )


def proiel_to_dict(corpus: Corpus) -> Dict[str, Any]:
    """Convert PROIEL corpus to dictionary"""
    converter = FormatConverter()
    return converter._corpus_to_json(corpus)


def conllu_to_dict(corpus: Corpus) -> Dict[str, Any]:
    """Convert CoNLL-U corpus to dictionary"""
    converter = FormatConverter()
    return converter._corpus_to_json(corpus)


def convert_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    input_format: Optional[InputFormat] = None,
    output_format: Optional[OutputFormat] = None,
    options: Optional[ConversionOptions] = None
) -> ConversionResult:
    """Generic format conversion function"""
    converter = FormatConverter(options)
    return converter.convert_file(
        input_path,
        output_path,
        input_format or InputFormat.AUTO,
        output_format
    )


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    input_format: InputFormat = InputFormat.AUTO,
    output_format: OutputFormat = OutputFormat.CONLLU,
    options: Optional[ConversionOptions] = None,
    recursive: bool = False
) -> List[ConversionResult]:
    """Batch convert files in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = FormatConverter(options)
    results = []
    
    if recursive:
        files = list(input_dir.rglob("*"))
    else:
        files = list(input_dir.glob("*"))
    
    for input_file in files:
        if input_file.is_file():
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(_get_extension(output_format))
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            result = converter.convert_file(
                input_file,
                output_file,
                input_format,
                output_format
            )
            results.append(result)
    
    return results


def _get_extension(output_format: OutputFormat) -> str:
    """Get file extension for output format"""
    extensions = {
        OutputFormat.PROIEL_XML: ".xml",
        OutputFormat.CONLLU: ".conllu",
        OutputFormat.JSON: ".json",
        OutputFormat.CSV: ".csv",
        OutputFormat.TSV: ".tsv",
        OutputFormat.PLAIN_TEXT: ".txt",
        OutputFormat.TEI_XML: ".tei.xml"
    }
    return extensions.get(output_format, ".txt")
