"""
HLP Annotation Stanza Engine - Stanza-based Annotation Pipeline

This module provides annotation capabilities using Stanford Stanza,
with special support for Ancient Greek, Latin, and other historical
Indo-European languages.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from hlp_annotation.base_engine import (
    AnnotationEngine, AnnotationCapability, AnnotationResult,
    AnnotationConfig, EngineStatus
)
from hlp_core.models import (
    Token, Sentence, MorphologicalFeatures, SyntacticRelation,
    PartOfSpeech, DependencyRelation, Case, Number, Gender,
    Person, Tense, Mood, Voice, Degree
)

logger = logging.getLogger(__name__)


STANZA_UPOS_MAP = {
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

STANZA_DEPREL_MAP = {
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

STANZA_LANGUAGE_PACKAGES = {
    "grc": ["perseus", "proiel"],
    "la": ["perseus", "proiel", "ittb", "llct", "udante"],
    "cu": ["proiel"],
    "got": ["proiel"],
    "xcl": ["caval"],
    "ang": ["default"],
    "non": ["default"],
    "san": ["vedic"],
    "el": ["gdt"],
    "en": ["default"],
    "de": ["default"],
    "fr": ["default"],
    "it": ["default"],
    "es": ["default"],
    "pt": ["default"],
    "ru": ["default"],
}


@dataclass
class StanzaConfig(AnnotationConfig):
    """Configuration for Stanza engine"""
    package: str = "proiel"
    
    processors: str = "tokenize,pos,lemma,depparse"
    
    tokenize_pretokenized: bool = False
    tokenize_no_ssplit: bool = False
    
    pos_batch_size: int = 5000
    lemma_batch_size: int = 50
    depparse_batch_size: int = 5000
    
    download_method: str = "default"
    
    model_dir: Optional[str] = None
    
    use_gpu: bool = False
    
    logging_level: str = "WARN"
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get Stanza pipeline configuration"""
        config = {
            "lang": self.language,
            "processors": self.processors,
            "tokenize_pretokenized": self.tokenize_pretokenized,
            "tokenize_no_ssplit": self.tokenize_no_ssplit,
            "use_gpu": self.use_gpu,
            "logging_level": self.logging_level,
        }
        
        if self.package:
            config["package"] = self.package
        
        if self.model_dir:
            config["dir"] = self.model_dir
        
        return config


class StanzaEngine(AnnotationEngine):
    """Stanza-based annotation engine"""
    
    def __init__(self, config: Optional[StanzaConfig] = None):
        super().__init__(config or StanzaConfig())
        self._nlp = None
        self._stanza_version = None
    
    @property
    def name(self) -> str:
        return "StanzaEngine"
    
    @property
    def version(self) -> str:
        if self._stanza_version:
            return f"Stanza {self._stanza_version}"
        return "Stanza (not loaded)"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        caps = []
        
        processors = self.config.processors.split(",")
        
        if "tokenize" in processors:
            caps.append(AnnotationCapability.TOKENIZATION)
            caps.append(AnnotationCapability.SENTENCE_SPLITTING)
        
        if "pos" in processors:
            caps.append(AnnotationCapability.POS_TAGGING)
        
        if "lemma" in processors:
            caps.append(AnnotationCapability.LEMMATIZATION)
        
        if "mwt" in processors:
            caps.append(AnnotationCapability.TOKENIZATION)
        
        if "depparse" in processors:
            caps.append(AnnotationCapability.DEPENDENCY_PARSING)
        
        if "ner" in processors:
            caps.append(AnnotationCapability.NAMED_ENTITY_RECOGNITION)
        
        caps.append(AnnotationCapability.MORPHOLOGICAL_ANALYSIS)
        
        return caps
    
    @property
    def supported_languages(self) -> List[str]:
        return list(STANZA_LANGUAGE_PACKAGES.keys())
    
    def initialize(self) -> bool:
        """Initialize Stanza pipeline"""
        if self._initialized and self._nlp is not None:
            return True
        
        try:
            import stanza
            self._stanza_version = stanza.__version__
            
            self._status = EngineStatus.INITIALIZING
            
            config = self.config
            if isinstance(config, StanzaConfig):
                pipeline_config = config.get_pipeline_config()
            else:
                pipeline_config = {
                    "lang": config.language,
                    "processors": "tokenize,pos,lemma,depparse",
                    "use_gpu": config.use_gpu,
                }
            
            model_dir = pipeline_config.get("dir")
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            try:
                self._nlp = stanza.Pipeline(**pipeline_config)
            except Exception as download_error:
                logger.info(f"Downloading Stanza model for {config.language}")
                stanza.download(
                    config.language,
                    package=pipeline_config.get("package"),
                    model_dir=model_dir
                )
                self._nlp = stanza.Pipeline(**pipeline_config)
            
            self._status = EngineStatus.READY
            self._initialized = True
            logger.info(f"Stanza engine initialized for {config.language}")
            return True
            
        except ImportError:
            logger.error("Stanza is not installed. Install with: pip install stanza")
            self._status = EngineStatus.ERROR
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize Stanza: {e}")
            self._status = EngineStatus.ERROR
            return False
    
    def shutdown(self):
        """Shutdown Stanza pipeline"""
        self._nlp = None
        self._initialized = False
        self._status = EngineStatus.SHUTDOWN
        logger.info("Stanza engine shutdown")
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text with Stanza"""
        if self._nlp is None:
            return AnnotationResult(
                success=False,
                errors=["Stanza pipeline not initialized"]
            )
        
        try:
            doc = self._nlp(text)
            
            sentences = []
            all_tokens = []
            
            for sent_idx, stanza_sent in enumerate(doc.sentences):
                tokens = []
                
                for word in stanza_sent.words:
                    morphology = self._extract_morphology(word)
                    syntax = self._extract_syntax(word)
                    
                    token = Token(
                        id=word.id,
                        form=word.text,
                        lemma=word.lemma,
                        morphology=morphology,
                        syntax=syntax,
                        span_start=word.start_char if hasattr(word, 'start_char') else None,
                        span_end=word.end_char if hasattr(word, 'end_char') else None
                    )
                    tokens.append(token)
                    all_tokens.append(token)
                
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    tokens=tokens,
                    text=stanza_sent.text,
                    sentence_index=sent_idx
                )
                sentences.append(sentence)
            
            return AnnotationResult(
                success=True,
                sentences=sentences,
                tokens=all_tokens,
                tokens_processed=len(all_tokens),
                sentences_processed=len(sentences),
                raw_output=doc
            )
            
        except Exception as e:
            logger.exception(f"Stanza processing error: {e}")
            return AnnotationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _extract_morphology(self, word: Any) -> MorphologicalFeatures:
        """Extract morphological features from Stanza word"""
        morphology = MorphologicalFeatures()
        
        if hasattr(word, 'upos') and word.upos:
            morphology.pos = STANZA_UPOS_MAP.get(word.upos, PartOfSpeech.X)
        
        if hasattr(word, 'xpos') and word.xpos:
            morphology.proiel_morph = word.xpos
        
        if hasattr(word, 'feats') and word.feats:
            feats = self._parse_feats(word.feats)
            
            case_map = {
                "Nom": Case.NOMINATIVE, "Gen": Case.GENITIVE,
                "Dat": Case.DATIVE, "Acc": Case.ACCUSATIVE,
                "Voc": Case.VOCATIVE, "Abl": Case.ABLATIVE,
                "Loc": Case.LOCATIVE, "Ins": Case.INSTRUMENTAL
            }
            if "Case" in feats:
                morphology.case = case_map.get(feats["Case"])
            
            number_map = {"Sing": Number.SINGULAR, "Dual": Number.DUAL, "Plur": Number.PLURAL}
            if "Number" in feats:
                morphology.number = number_map.get(feats["Number"])
            
            gender_map = {"Masc": Gender.MASCULINE, "Fem": Gender.FEMININE, "Neut": Gender.NEUTER}
            if "Gender" in feats:
                morphology.gender = gender_map.get(feats["Gender"])
            
            person_map = {"1": Person.FIRST, "2": Person.SECOND, "3": Person.THIRD}
            if "Person" in feats:
                morphology.person = person_map.get(feats["Person"])
            
            tense_map = {
                "Pres": Tense.PRESENT, "Past": Tense.PAST, "Fut": Tense.FUTURE,
                "Imp": Tense.IMPERFECT, "Pqp": Tense.PLUPERFECT,
                "Aor": Tense.AORIST, "Perf": Tense.PERFECT
            }
            if "Tense" in feats:
                morphology.tense = tense_map.get(feats["Tense"])
            
            mood_map = {
                "Ind": Mood.INDICATIVE, "Sub": Mood.SUBJUNCTIVE,
                "Imp": Mood.IMPERATIVE, "Opt": Mood.OPTATIVE,
                "Inf": Mood.INFINITIVE, "Part": Mood.PARTICIPLE
            }
            if "Mood" in feats:
                morphology.mood = mood_map.get(feats["Mood"])
            
            if "VerbForm" in feats:
                verbform = feats["VerbForm"]
                if verbform == "Inf":
                    morphology.mood = Mood.INFINITIVE
                elif verbform == "Part":
                    morphology.mood = Mood.PARTICIPLE
                elif verbform == "Ger":
                    morphology.mood = Mood.GERUND
                elif verbform == "Gdv":
                    morphology.mood = Mood.GERUNDIVE
                elif verbform == "Sup":
                    morphology.mood = Mood.SUPINE
            
            voice_map = {"Act": Voice.ACTIVE, "Pass": Voice.PASSIVE, "Mid": Voice.MIDDLE}
            if "Voice" in feats:
                morphology.voice = voice_map.get(feats["Voice"])
            
            degree_map = {"Pos": Degree.POSITIVE, "Cmp": Degree.COMPARATIVE, "Sup": Degree.SUPERLATIVE}
            if "Degree" in feats:
                morphology.degree = degree_map.get(feats["Degree"])
        
        return morphology
    
    def _extract_syntax(self, word: Any) -> Optional[SyntacticRelation]:
        """Extract syntactic relation from Stanza word"""
        if not hasattr(word, 'head') or not hasattr(word, 'deprel'):
            return None
        
        head_id = word.head if word.head is not None else 0
        deprel = word.deprel or "dep"
        
        relation = STANZA_DEPREL_MAP.get(deprel)
        if relation is None:
            base_rel = deprel.split(":")[0]
            relation = STANZA_DEPREL_MAP.get(base_rel, DependencyRelation.DEP)
        
        return SyntacticRelation(
            head_id=head_id,
            relation=relation
        )
    
    def _parse_feats(self, feats_str: str) -> Dict[str, str]:
        """Parse feature string"""
        feats = {}
        if feats_str and feats_str != "_":
            for feat in feats_str.split("|"):
                if "=" in feat:
                    key, value = feat.split("=", 1)
                    feats[key] = value
        return feats
    
    def process_pretokenized(
        self,
        tokens: List[List[str]]
    ) -> AnnotationResult:
        """Process pre-tokenized text"""
        if self._nlp is None:
            if not self.initialize():
                return AnnotationResult(
                    success=False,
                    errors=["Failed to initialize Stanza"]
                )
        
        try:
            doc = self._nlp(tokens)
            
            sentences = []
            all_tokens = []
            
            for sent_idx, stanza_sent in enumerate(doc.sentences):
                sent_tokens = []
                
                for word in stanza_sent.words:
                    morphology = self._extract_morphology(word)
                    syntax = self._extract_syntax(word)
                    
                    token = Token(
                        id=word.id,
                        form=word.text,
                        lemma=word.lemma,
                        morphology=morphology,
                        syntax=syntax
                    )
                    sent_tokens.append(token)
                    all_tokens.append(token)
                
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    tokens=sent_tokens,
                    text=" ".join(t.form for t in sent_tokens),
                    sentence_index=sent_idx
                )
                sentences.append(sentence)
            
            return AnnotationResult(
                success=True,
                sentences=sentences,
                tokens=all_tokens,
                tokens_processed=len(all_tokens),
                sentences_processed=len(sentences)
            )
            
        except Exception as e:
            logger.exception(f"Stanza pretokenized processing error: {e}")
            return AnnotationResult(
                success=False,
                errors=[str(e)]
            )
    
    def get_available_packages(self, language: str) -> List[str]:
        """Get available packages for a language"""
        return STANZA_LANGUAGE_PACKAGES.get(language, ["default"])
    
    def download_model(self, language: str, package: Optional[str] = None) -> bool:
        """Download Stanza model"""
        try:
            import stanza
            
            model_dir = None
            if isinstance(self.config, StanzaConfig) and self.config.model_dir:
                model_dir = self.config.model_dir
            
            stanza.download(language, package=package, model_dir=model_dir)
            logger.info(f"Downloaded Stanza model for {language} ({package or 'default'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Stanza model: {e}")
            return False
    
    def switch_language(self, language: str, package: Optional[str] = None) -> bool:
        """Switch to a different language"""
        self.shutdown()
        
        if isinstance(self.config, StanzaConfig):
            self.config.language = language
            if package:
                self.config.package = package
        else:
            self.config.language = language
        
        return self.initialize()


def create_stanza_engine(
    language: str = "grc",
    package: str = "proiel",
    processors: str = "tokenize,pos,lemma,depparse",
    use_gpu: bool = False,
    model_dir: Optional[str] = None
) -> StanzaEngine:
    """Factory function to create Stanza engine"""
    config = StanzaConfig(
        language=language,
        package=package,
        processors=processors,
        use_gpu=use_gpu,
        model_dir=model_dir
    )
    return StanzaEngine(config)
