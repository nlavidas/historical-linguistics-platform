"""
HLP Annotation spaCy Engine - spaCy-based Annotation Pipeline

This module provides annotation capabilities using spaCy,
with support for custom models and pipelines.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
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
    Person, Tense, Mood, Voice, Degree,
    NamedEntity, NamedEntityType
)

logger = logging.getLogger(__name__)


SPACY_POS_MAP = {
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
    "SPACE": PartOfSpeech.X,
}

SPACY_DEPREL_MAP = {
    "acl": DependencyRelation.ACL,
    "acomp": DependencyRelation.XCOMP,
    "advcl": DependencyRelation.ADVCL,
    "advmod": DependencyRelation.ADVMOD,
    "agent": DependencyRelation.OBL,
    "amod": DependencyRelation.AMOD,
    "appos": DependencyRelation.APPOS,
    "attr": DependencyRelation.NSUBJ,
    "aux": DependencyRelation.AUX,
    "auxpass": DependencyRelation.AUX_PASS,
    "case": DependencyRelation.CASE,
    "cc": DependencyRelation.CC,
    "ccomp": DependencyRelation.CCOMP,
    "compound": DependencyRelation.COMPOUND,
    "conj": DependencyRelation.CONJ,
    "cop": DependencyRelation.COP,
    "csubj": DependencyRelation.CSUBJ,
    "csubjpass": DependencyRelation.CSUBJ_PASS,
    "dative": DependencyRelation.IOBJ,
    "dep": DependencyRelation.DEP,
    "det": DependencyRelation.DET,
    "dobj": DependencyRelation.OBJ,
    "expl": DependencyRelation.EXPL,
    "intj": DependencyRelation.DISCOURSE,
    "mark": DependencyRelation.MARK,
    "meta": DependencyRelation.DEP,
    "neg": DependencyRelation.ADVMOD,
    "nmod": DependencyRelation.NMOD,
    "npadvmod": DependencyRelation.OBL,
    "nsubj": DependencyRelation.NSUBJ,
    "nsubjpass": DependencyRelation.NSUBJ_PASS,
    "nummod": DependencyRelation.NUMMOD,
    "obj": DependencyRelation.OBJ,
    "obl": DependencyRelation.OBL,
    "oprd": DependencyRelation.XCOMP,
    "parataxis": DependencyRelation.PARATAXIS,
    "pcomp": DependencyRelation.CCOMP,
    "pobj": DependencyRelation.OBL,
    "poss": DependencyRelation.NMOD,
    "preconj": DependencyRelation.CC,
    "prep": DependencyRelation.CASE,
    "prt": DependencyRelation.COMPOUND,
    "punct": DependencyRelation.PUNCT,
    "quantmod": DependencyRelation.ADVMOD,
    "relcl": DependencyRelation.ACL_RELCL,
    "root": DependencyRelation.ROOT,
    "ROOT": DependencyRelation.ROOT,
    "vocative": DependencyRelation.VOCATIVE,
    "xcomp": DependencyRelation.XCOMP,
}

SPACY_NER_MAP = {
    "PERSON": NamedEntityType.PERSON,
    "PER": NamedEntityType.PERSON,
    "NORP": NamedEntityType.NORP,
    "FAC": NamedEntityType.FACILITY,
    "ORG": NamedEntityType.ORGANIZATION,
    "GPE": NamedEntityType.GPE,
    "LOC": NamedEntityType.LOCATION,
    "PRODUCT": NamedEntityType.PRODUCT,
    "EVENT": NamedEntityType.EVENT,
    "WORK_OF_ART": NamedEntityType.WORK_OF_ART,
    "LAW": NamedEntityType.LAW,
    "LANGUAGE": NamedEntityType.LANGUAGE,
    "DATE": NamedEntityType.DATE,
    "TIME": NamedEntityType.TIME,
    "PERCENT": NamedEntityType.PERCENT,
    "MONEY": NamedEntityType.MONEY,
    "QUANTITY": NamedEntityType.QUANTITY,
    "ORDINAL": NamedEntityType.ORDINAL,
    "CARDINAL": NamedEntityType.CARDINAL,
    "MISC": NamedEntityType.MISC,
}

SPACY_MODELS = {
    "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"],
    "de": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
    "fr": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
    "es": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
    "it": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    "pt": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    "nl": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    "el": ["el_core_news_sm", "el_core_news_md", "el_core_news_lg"],
    "ru": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
    "zh": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
    "ja": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    "xx": ["xx_ent_wiki_sm"],
}


@dataclass
class SpacyConfig(AnnotationConfig):
    """Configuration for spaCy engine"""
    model_name: Optional[str] = None
    
    disable_components: List[str] = field(default_factory=list)
    enable_components: List[str] = field(default_factory=list)
    
    use_transformer: bool = False
    
    custom_model_path: Optional[str] = None
    
    max_length: int = 1000000
    
    prefer_gpu: bool = False
    
    def get_model_name(self) -> str:
        """Get model name based on configuration"""
        if self.custom_model_path:
            return self.custom_model_path
        
        if self.model_name:
            return self.model_name
        
        lang_models = SPACY_MODELS.get(self.language, [])
        if lang_models:
            if self.use_transformer:
                trf_models = [m for m in lang_models if "trf" in m]
                if trf_models:
                    return trf_models[0]
            return lang_models[0]
        
        return "xx_ent_wiki_sm"


class SpacyEngine(AnnotationEngine):
    """spaCy-based annotation engine"""
    
    def __init__(self, config: Optional[SpacyConfig] = None):
        super().__init__(config or SpacyConfig())
        self._nlp = None
        self._spacy_version = None
    
    @property
    def name(self) -> str:
        return "SpacyEngine"
    
    @property
    def version(self) -> str:
        if self._spacy_version:
            return f"spaCy {self._spacy_version}"
        return "spaCy (not loaded)"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        caps = [
            AnnotationCapability.TOKENIZATION,
            AnnotationCapability.SENTENCE_SPLITTING,
        ]
        
        if self._nlp is not None:
            pipe_names = self._nlp.pipe_names
            
            if "tagger" in pipe_names:
                caps.append(AnnotationCapability.POS_TAGGING)
            
            if "lemmatizer" in pipe_names:
                caps.append(AnnotationCapability.LEMMATIZATION)
            
            if "morphologizer" in pipe_names:
                caps.append(AnnotationCapability.MORPHOLOGICAL_ANALYSIS)
            
            if "parser" in pipe_names:
                caps.append(AnnotationCapability.DEPENDENCY_PARSING)
            
            if "ner" in pipe_names:
                caps.append(AnnotationCapability.NAMED_ENTITY_RECOGNITION)
            
            if "transformer" in pipe_names or "tok2vec" in pipe_names:
                caps.append(AnnotationCapability.EMBEDDINGS)
        else:
            caps.extend([
                AnnotationCapability.POS_TAGGING,
                AnnotationCapability.LEMMATIZATION,
                AnnotationCapability.DEPENDENCY_PARSING,
                AnnotationCapability.NAMED_ENTITY_RECOGNITION,
            ])
        
        return caps
    
    @property
    def supported_languages(self) -> List[str]:
        return list(SPACY_MODELS.keys())
    
    def initialize(self) -> bool:
        """Initialize spaCy pipeline"""
        if self._initialized and self._nlp is not None:
            return True
        
        try:
            import spacy
            self._spacy_version = spacy.__version__
            
            self._status = EngineStatus.INITIALIZING
            
            config = self.config
            if isinstance(config, SpacyConfig):
                model_name = config.get_model_name()
                
                if config.prefer_gpu:
                    try:
                        spacy.prefer_gpu()
                    except Exception:
                        pass
            else:
                model_name = SPACY_MODELS.get(config.language, ["xx_ent_wiki_sm"])[0]
            
            try:
                if isinstance(config, SpacyConfig) and config.disable_components:
                    self._nlp = spacy.load(model_name, disable=config.disable_components)
                else:
                    self._nlp = spacy.load(model_name)
            except OSError:
                logger.info(f"Downloading spaCy model: {model_name}")
                spacy.cli.download(model_name)
                self._nlp = spacy.load(model_name)
            
            if isinstance(config, SpacyConfig):
                self._nlp.max_length = config.max_length
            
            self._status = EngineStatus.READY
            self._initialized = True
            logger.info(f"spaCy engine initialized with model: {model_name}")
            return True
            
        except ImportError:
            logger.error("spaCy is not installed. Install with: pip install spacy")
            self._status = EngineStatus.ERROR
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize spaCy: {e}")
            self._status = EngineStatus.ERROR
            return False
    
    def shutdown(self):
        """Shutdown spaCy pipeline"""
        self._nlp = None
        self._initialized = False
        self._status = EngineStatus.SHUTDOWN
        logger.info("spaCy engine shutdown")
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text with spaCy"""
        if self._nlp is None:
            return AnnotationResult(
                success=False,
                errors=["spaCy pipeline not initialized"]
            )
        
        try:
            doc = self._nlp(text)
            
            sentences = []
            all_tokens = []
            all_entities = []
            
            for sent_idx, spacy_sent in enumerate(doc.sents):
                tokens = []
                token_offset = len(all_tokens)
                
                for token in spacy_sent:
                    morphology = self._extract_morphology(token)
                    syntax = self._extract_syntax(token, token_offset)
                    
                    hlp_token = Token(
                        id=token.i - spacy_sent.start + 1,
                        form=token.text,
                        lemma=token.lemma_,
                        morphology=morphology,
                        syntax=syntax,
                        span_start=token.idx,
                        span_end=token.idx + len(token.text)
                    )
                    tokens.append(hlp_token)
                    all_tokens.append(hlp_token)
                
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    tokens=tokens,
                    text=spacy_sent.text,
                    sentence_index=sent_idx
                )
                sentences.append(sentence)
            
            if AnnotationCapability.NAMED_ENTITY_RECOGNITION in capabilities:
                for ent in doc.ents:
                    entity_type = SPACY_NER_MAP.get(ent.label_, NamedEntityType.MISC)
                    entity = NamedEntity(
                        entity_type=entity_type,
                        text=ent.text,
                        span_start=ent.start_char,
                        span_end=ent.end_char,
                        confidence=1.0
                    )
                    all_entities.append(entity)
            
            embeddings = None
            if AnnotationCapability.EMBEDDINGS in capabilities and doc.has_vector:
                embeddings = doc.vector
            
            return AnnotationResult(
                success=True,
                sentences=sentences,
                tokens=all_tokens,
                entities=all_entities,
                embeddings=embeddings,
                tokens_processed=len(all_tokens),
                sentences_processed=len(sentences),
                raw_output=doc
            )
            
        except Exception as e:
            logger.exception(f"spaCy processing error: {e}")
            return AnnotationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _extract_morphology(self, token: Any) -> MorphologicalFeatures:
        """Extract morphological features from spaCy token"""
        morphology = MorphologicalFeatures()
        
        if token.pos_:
            morphology.pos = SPACY_POS_MAP.get(token.pos_, PartOfSpeech.X)
        
        if token.tag_:
            morphology.proiel_morph = token.tag_
        
        morph = token.morph.to_dict() if hasattr(token.morph, 'to_dict') else {}
        
        case_map = {
            "Nom": Case.NOMINATIVE, "Gen": Case.GENITIVE,
            "Dat": Case.DATIVE, "Acc": Case.ACCUSATIVE,
            "Voc": Case.VOCATIVE, "Abl": Case.ABLATIVE,
            "Loc": Case.LOCATIVE, "Ins": Case.INSTRUMENTAL
        }
        if "Case" in morph:
            morphology.case = case_map.get(morph["Case"])
        
        number_map = {"Sing": Number.SINGULAR, "Dual": Number.DUAL, "Plur": Number.PLURAL}
        if "Number" in morph:
            morphology.number = number_map.get(morph["Number"])
        
        gender_map = {"Masc": Gender.MASCULINE, "Fem": Gender.FEMININE, "Neut": Gender.NEUTER}
        if "Gender" in morph:
            morphology.gender = gender_map.get(morph["Gender"])
        
        person_map = {"1": Person.FIRST, "2": Person.SECOND, "3": Person.THIRD}
        if "Person" in morph:
            morphology.person = person_map.get(morph["Person"])
        
        tense_map = {"Pres": Tense.PRESENT, "Past": Tense.PAST, "Fut": Tense.FUTURE}
        if "Tense" in morph:
            morphology.tense = tense_map.get(morph["Tense"])
        
        mood_map = {"Ind": Mood.INDICATIVE, "Sub": Mood.SUBJUNCTIVE, "Imp": Mood.IMPERATIVE}
        if "Mood" in morph:
            morphology.mood = mood_map.get(morph["Mood"])
        
        voice_map = {"Act": Voice.ACTIVE, "Pass": Voice.PASSIVE}
        if "Voice" in morph:
            morphology.voice = voice_map.get(morph["Voice"])
        
        return morphology
    
    def _extract_syntax(self, token: Any, offset: int = 0) -> Optional[SyntacticRelation]:
        """Extract syntactic relation from spaCy token"""
        if not hasattr(token, 'head') or not hasattr(token, 'dep_'):
            return None
        
        head_id = token.head.i - token.sent.start + 1 if token.head != token else 0
        deprel = token.dep_ or "dep"
        
        relation = SPACY_DEPREL_MAP.get(deprel.lower(), DependencyRelation.DEP)
        
        return SyntacticRelation(
            head_id=head_id,
            relation=relation
        )
    
    def process_batch(
        self,
        texts: List[str],
        capabilities: Optional[List[AnnotationCapability]] = None
    ) -> List[AnnotationResult]:
        """Process batch of texts efficiently using spaCy's pipe"""
        if self._nlp is None:
            if not self.initialize():
                return [AnnotationResult(success=False, errors=["Failed to initialize"])]
        
        if capabilities is None:
            capabilities = self.capabilities
        
        results = []
        
        try:
            batch_size = self.config.batch_size
            
            for doc in self._nlp.pipe(texts, batch_size=batch_size):
                sentences = []
                all_tokens = []
                all_entities = []
                
                for sent_idx, spacy_sent in enumerate(doc.sents):
                    tokens = []
                    
                    for token in spacy_sent:
                        morphology = self._extract_morphology(token)
                        syntax = self._extract_syntax(token)
                        
                        hlp_token = Token(
                            id=token.i - spacy_sent.start + 1,
                            form=token.text,
                            lemma=token.lemma_,
                            morphology=morphology,
                            syntax=syntax
                        )
                        tokens.append(hlp_token)
                        all_tokens.append(hlp_token)
                    
                    sentence = Sentence(
                        id=f"s{sent_idx + 1}",
                        tokens=tokens,
                        text=spacy_sent.text,
                        sentence_index=sent_idx
                    )
                    sentences.append(sentence)
                
                if AnnotationCapability.NAMED_ENTITY_RECOGNITION in capabilities:
                    for ent in doc.ents:
                        entity_type = SPACY_NER_MAP.get(ent.label_, NamedEntityType.MISC)
                        entity = NamedEntity(
                            entity_type=entity_type,
                            text=ent.text,
                            span_start=ent.start_char,
                            span_end=ent.end_char
                        )
                        all_entities.append(entity)
                
                results.append(AnnotationResult(
                    success=True,
                    sentences=sentences,
                    tokens=all_tokens,
                    entities=all_entities,
                    tokens_processed=len(all_tokens),
                    sentences_processed=len(sentences)
                ))
            
        except Exception as e:
            logger.exception(f"spaCy batch processing error: {e}")
            results.append(AnnotationResult(success=False, errors=[str(e)]))
        
        return results
    
    def add_custom_component(
        self,
        component_name: str,
        component_func: Any,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> bool:
        """Add custom component to pipeline"""
        if self._nlp is None:
            return False
        
        try:
            if before:
                self._nlp.add_pipe(component_name, before=before)
            elif after:
                self._nlp.add_pipe(component_name, after=after)
            else:
                self._nlp.add_pipe(component_name)
            return True
        except Exception as e:
            logger.error(f"Failed to add custom component: {e}")
            return False
    
    def get_available_models(self, language: str) -> List[str]:
        """Get available models for a language"""
        return SPACY_MODELS.get(language, [])
    
    def download_model(self, model_name: str) -> bool:
        """Download spaCy model"""
        try:
            import spacy
            spacy.cli.download(model_name)
            logger.info(f"Downloaded spaCy model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
            return False


def create_spacy_engine(
    language: str = "en",
    model_name: Optional[str] = None,
    use_transformer: bool = False
) -> SpacyEngine:
    """Factory function to create spaCy engine"""
    config = SpacyConfig(
        language=language,
        model_name=model_name,
        use_transformer=use_transformer
    )
    return SpacyEngine(config)
