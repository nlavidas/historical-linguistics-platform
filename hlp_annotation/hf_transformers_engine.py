"""
HLP Annotation HuggingFace Engine - HuggingFace Transformers Integration

This module provides annotation capabilities using HuggingFace Transformers,
with support for various NLP tasks including token classification,
sequence classification, and text generation.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum

from hlp_annotation.base_engine import (
    AnnotationEngine, AnnotationCapability, AnnotationResult,
    AnnotationConfig, EngineStatus
)
from hlp_core.models import (
    Token, Sentence, MorphologicalFeatures, SyntacticRelation,
    PartOfSpeech, DependencyRelation,
    NamedEntity, NamedEntityType, SemanticRole, SemanticRoleLabel
)

logger = logging.getLogger(__name__)


class HFTaskType(Enum):
    """HuggingFace task types"""
    TOKEN_CLASSIFICATION = "token-classification"
    SEQUENCE_CLASSIFICATION = "text-classification"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FILL_MASK = "fill-mask"
    NER = "ner"
    POS_TAGGING = "pos"
    DEPENDENCY_PARSING = "dep"
    FEATURE_EXTRACTION = "feature-extraction"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


HF_NER_LABEL_MAP = {
    "PER": NamedEntityType.PERSON,
    "PERSON": NamedEntityType.PERSON,
    "B-PER": NamedEntityType.PERSON,
    "I-PER": NamedEntityType.PERSON,
    "ORG": NamedEntityType.ORGANIZATION,
    "ORGANIZATION": NamedEntityType.ORGANIZATION,
    "B-ORG": NamedEntityType.ORGANIZATION,
    "I-ORG": NamedEntityType.ORGANIZATION,
    "LOC": NamedEntityType.LOCATION,
    "LOCATION": NamedEntityType.LOCATION,
    "B-LOC": NamedEntityType.LOCATION,
    "I-LOC": NamedEntityType.LOCATION,
    "GPE": NamedEntityType.GPE,
    "B-GPE": NamedEntityType.GPE,
    "I-GPE": NamedEntityType.GPE,
    "MISC": NamedEntityType.MISC,
    "B-MISC": NamedEntityType.MISC,
    "I-MISC": NamedEntityType.MISC,
    "DATE": NamedEntityType.DATE,
    "B-DATE": NamedEntityType.DATE,
    "I-DATE": NamedEntityType.DATE,
    "TIME": NamedEntityType.TIME,
    "B-TIME": NamedEntityType.TIME,
    "I-TIME": NamedEntityType.TIME,
    "EVENT": NamedEntityType.EVENT,
    "B-EVENT": NamedEntityType.EVENT,
    "I-EVENT": NamedEntityType.EVENT,
}

HF_POS_LABEL_MAP = {
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

RECOMMENDED_MODELS = {
    "ner": {
        "en": "dslim/bert-base-NER",
        "multilingual": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "grc": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "la": "Davlan/bert-base-multilingual-cased-ner-hrl",
    },
    "pos": {
        "en": "vblagoje/bert-english-uncased-finetuned-pos",
        "multilingual": "Davlan/bert-base-multilingual-cased-pos-english",
    },
    "embeddings": {
        "en": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "grc": "bowphs/GreekBERT",
        "la": "bowphs/LaBerta",
    },
    "translation": {
        "grc-en": "Helsinki-NLP/opus-mt-grk-en",
        "la-en": "Helsinki-NLP/opus-mt-la-en",
        "en-grc": "Helsinki-NLP/opus-mt-en-grk",
        "en-la": "Helsinki-NLP/opus-mt-en-la",
    },
    "generation": {
        "en": "gpt2",
        "multilingual": "bigscience/bloom-560m",
    }
}


@dataclass
class HuggingFaceConfig(AnnotationConfig):
    """Configuration for HuggingFace engine"""
    model_name: Optional[str] = None
    
    task_type: HFTaskType = HFTaskType.TOKEN_CLASSIFICATION
    
    tokenizer_name: Optional[str] = None
    
    model_revision: str = "main"
    
    use_auth_token: Optional[str] = None
    
    cache_dir: Optional[str] = None
    
    use_fast_tokenizer: bool = True
    
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    aggregation_strategy: str = "simple"
    
    generation_max_length: int = 100
    generation_num_beams: int = 4
    generation_temperature: float = 1.0
    generation_top_p: float = 0.9
    generation_do_sample: bool = False
    
    return_all_scores: bool = False
    
    local_files_only: bool = False
    
    trust_remote_code: bool = False
    
    def get_model_name(self) -> str:
        """Get model name based on configuration"""
        if self.model_name:
            return self.model_name
        
        task_models = RECOMMENDED_MODELS.get(self.task_type.value, {})
        if self.language in task_models:
            return task_models[self.language]
        if "multilingual" in task_models:
            return task_models["multilingual"]
        if "en" in task_models:
            return task_models["en"]
        
        return "bert-base-multilingual-cased"


class HuggingFaceEngine(AnnotationEngine):
    """HuggingFace Transformers-based annotation engine"""
    
    def __init__(self, config: Optional[HuggingFaceConfig] = None):
        super().__init__(config or HuggingFaceConfig())
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._transformers_version = None
    
    @property
    def name(self) -> str:
        return "HuggingFaceEngine"
    
    @property
    def version(self) -> str:
        if self._transformers_version:
            return f"Transformers {self._transformers_version}"
        return "Transformers (not loaded)"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        config = self.config
        if not isinstance(config, HuggingFaceConfig):
            return [AnnotationCapability.EMBEDDINGS]
        
        task_capabilities = {
            HFTaskType.TOKEN_CLASSIFICATION: [
                AnnotationCapability.POS_TAGGING,
                AnnotationCapability.NAMED_ENTITY_RECOGNITION,
            ],
            HFTaskType.NER: [
                AnnotationCapability.NAMED_ENTITY_RECOGNITION,
            ],
            HFTaskType.POS_TAGGING: [
                AnnotationCapability.POS_TAGGING,
            ],
            HFTaskType.SEQUENCE_CLASSIFICATION: [
                AnnotationCapability.TEXT_CLASSIFICATION,
                AnnotationCapability.SENTIMENT_ANALYSIS,
            ],
            HFTaskType.FEATURE_EXTRACTION: [
                AnnotationCapability.EMBEDDINGS,
            ],
            HFTaskType.TEXT_GENERATION: [
                AnnotationCapability.GENERATION,
            ],
            HFTaskType.TEXT2TEXT_GENERATION: [
                AnnotationCapability.GENERATION,
                AnnotationCapability.TRANSLATION,
            ],
            HFTaskType.TRANSLATION: [
                AnnotationCapability.TRANSLATION,
            ],
            HFTaskType.QUESTION_ANSWERING: [
                AnnotationCapability.TEXT_CLASSIFICATION,
            ],
        }
        
        return task_capabilities.get(config.task_type, [AnnotationCapability.EMBEDDINGS])
    
    @property
    def supported_languages(self) -> List[str]:
        return ["grc", "la", "en", "de", "fr", "es", "it", "pt", "ru", "el", "zh", "ja", "ar", "multilingual"]
    
    def initialize(self) -> bool:
        """Initialize HuggingFace pipeline"""
        if self._initialized and self._pipeline is not None:
            return True
        
        try:
            import transformers
            self._transformers_version = transformers.__version__
            
            from transformers import pipeline, AutoTokenizer, AutoModel
            
            self._status = EngineStatus.INITIALIZING
            
            config = self.config
            if isinstance(config, HuggingFaceConfig):
                model_name = config.get_model_name()
                task_type = config.task_type.value
                
                pipeline_kwargs = {
                    "task": task_type,
                    "model": model_name,
                    "device": 0 if config.use_gpu else -1,
                }
                
                if config.tokenizer_name:
                    pipeline_kwargs["tokenizer"] = config.tokenizer_name
                
                if config.cache_dir:
                    pipeline_kwargs["cache_dir"] = config.cache_dir
                
                if config.use_auth_token:
                    pipeline_kwargs["token"] = config.use_auth_token
                
                if config.trust_remote_code:
                    pipeline_kwargs["trust_remote_code"] = True
                
                if task_type in ["ner", "token-classification"]:
                    pipeline_kwargs["aggregation_strategy"] = config.aggregation_strategy
                
            else:
                model_name = "bert-base-multilingual-cased"
                task_type = "token-classification"
                pipeline_kwargs = {
                    "task": task_type,
                    "model": model_name,
                    "device": -1,
                }
            
            logger.info(f"Loading HuggingFace pipeline: {model_name} for {task_type}")
            
            self._pipeline = pipeline(**pipeline_kwargs)
            
            self._status = EngineStatus.READY
            self._initialized = True
            logger.info(f"HuggingFace engine initialized with model: {model_name}")
            return True
            
        except ImportError:
            logger.error("Transformers is not installed. Install with: pip install transformers")
            self._status = EngineStatus.ERROR
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize HuggingFace: {e}")
            self._status = EngineStatus.ERROR
            return False
    
    def shutdown(self):
        """Shutdown HuggingFace pipeline"""
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._status = EngineStatus.SHUTDOWN
        logger.info("HuggingFace engine shutdown")
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text with HuggingFace"""
        if self._pipeline is None:
            return AnnotationResult(
                success=False,
                errors=["HuggingFace pipeline not initialized"]
            )
        
        try:
            config = self.config
            task_type = config.task_type if isinstance(config, HuggingFaceConfig) else HFTaskType.TOKEN_CLASSIFICATION
            
            if task_type in [HFTaskType.TOKEN_CLASSIFICATION, HFTaskType.NER]:
                return self._process_token_classification(text, capabilities)
            elif task_type == HFTaskType.POS_TAGGING:
                return self._process_pos_tagging(text, capabilities)
            elif task_type == HFTaskType.SEQUENCE_CLASSIFICATION:
                return self._process_sequence_classification(text, capabilities)
            elif task_type == HFTaskType.FEATURE_EXTRACTION:
                return self._process_feature_extraction(text, capabilities)
            elif task_type in [HFTaskType.TEXT_GENERATION, HFTaskType.TEXT2TEXT_GENERATION]:
                return self._process_text_generation(text, capabilities)
            elif task_type == HFTaskType.TRANSLATION:
                return self._process_translation(text, capabilities)
            else:
                return self._process_token_classification(text, capabilities)
            
        except Exception as e:
            logger.exception(f"HuggingFace processing error: {e}")
            return AnnotationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _process_token_classification(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process token classification (NER, POS, etc.)"""
        results = self._pipeline(text)
        
        entities = []
        tokens = []
        
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    label = item.get("entity_group", item.get("entity", ""))
                    word = item.get("word", "")
                    score = item.get("score", 1.0)
                    start = item.get("start", 0)
                    end = item.get("end", len(word))
                    
                    if label.startswith("B-") or label.startswith("I-"):
                        label_clean = label[2:]
                    else:
                        label_clean = label
                    
                    entity_type = HF_NER_LABEL_MAP.get(label, HF_NER_LABEL_MAP.get(label_clean, NamedEntityType.MISC))
                    
                    entity = NamedEntity(
                        entity_type=entity_type,
                        text=word,
                        span_start=start,
                        span_end=end,
                        confidence=score,
                        label=label
                    )
                    entities.append(entity)
        
        words = text.split()
        for idx, word in enumerate(words, start=1):
            token = Token(
                id=idx,
                form=word
            )
            tokens.append(token)
        
        return AnnotationResult(
            success=True,
            tokens=tokens,
            entities=entities,
            tokens_processed=len(tokens),
            raw_output=results
        )
    
    def _process_pos_tagging(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process POS tagging"""
        results = self._pipeline(text)
        
        tokens = []
        
        if isinstance(results, list):
            for idx, item in enumerate(results, start=1):
                if isinstance(item, dict):
                    word = item.get("word", "")
                    label = item.get("entity", item.get("entity_group", "X"))
                    score = item.get("score", 1.0)
                    
                    pos = HF_POS_LABEL_MAP.get(label, PartOfSpeech.X)
                    
                    morphology = MorphologicalFeatures(pos=pos)
                    
                    token = Token(
                        id=idx,
                        form=word,
                        morphology=morphology
                    )
                    tokens.append(token)
        
        return AnnotationResult(
            success=True,
            tokens=tokens,
            tokens_processed=len(tokens),
            raw_output=results
        )
    
    def _process_sequence_classification(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process sequence classification"""
        results = self._pipeline(text)
        
        metadata = {}
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            metadata["label"] = result.get("label", "")
            metadata["score"] = result.get("score", 0.0)
        elif isinstance(results, dict):
            metadata["label"] = results.get("label", "")
            metadata["score"] = results.get("score", 0.0)
        
        return AnnotationResult(
            success=True,
            metadata=metadata,
            raw_output=results
        )
    
    def _process_feature_extraction(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process feature extraction (embeddings)"""
        results = self._pipeline(text)
        
        embeddings = None
        if isinstance(results, list):
            import numpy as np
            embeddings = np.array(results)
            if len(embeddings.shape) == 3:
                embeddings = embeddings.mean(axis=1)
        
        return AnnotationResult(
            success=True,
            embeddings=embeddings,
            raw_output=results
        )
    
    def _process_text_generation(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text generation"""
        config = self.config
        
        gen_kwargs = {}
        if isinstance(config, HuggingFaceConfig):
            gen_kwargs = {
                "max_length": config.generation_max_length,
                "num_beams": config.generation_num_beams,
                "temperature": config.generation_temperature,
                "top_p": config.generation_top_p,
                "do_sample": config.generation_do_sample,
            }
        
        results = self._pipeline(text, **gen_kwargs)
        
        generated_text = ""
        if isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get("generated_text", "")
        
        return AnnotationResult(
            success=True,
            metadata={"generated_text": generated_text},
            raw_output=results
        )
    
    def _process_translation(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process translation"""
        results = self._pipeline(text)
        
        translated_text = ""
        if isinstance(results, list) and len(results) > 0:
            translated_text = results[0].get("translation_text", "")
        
        return AnnotationResult(
            success=True,
            metadata={"translation": translated_text},
            raw_output=results
        )
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> Any:
        """Get embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self._pipeline is None:
            if not self.initialize():
                return None
        
        try:
            results = self._pipeline(texts)
            
            import numpy as np
            embeddings = np.array(results)
            
            if len(embeddings.shape) == 3:
                embeddings = embeddings.mean(axis=1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return None
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text"""
        if self._pipeline is None:
            if not self.initialize():
                return {}
        
        try:
            results = self._pipeline(text)
            
            if isinstance(results, list) and len(results) > 0:
                return results[0]
            elif isinstance(results, dict):
                return results
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to classify text: {e}")
            return {}
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text from prompt"""
        if self._pipeline is None:
            if not self.initialize():
                return []
        
        try:
            results = self._pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences
            )
            
            generated = []
            for result in results:
                if isinstance(result, dict):
                    generated.append(result.get("generated_text", ""))
            
            return generated
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return []
    
    def translate(self, text: str) -> str:
        """Translate text"""
        if self._pipeline is None:
            if not self.initialize():
                return ""
        
        try:
            results = self._pipeline(text)
            
            if isinstance(results, list) and len(results) > 0:
                return results[0].get("translation_text", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to translate: {e}")
            return ""
    
    @staticmethod
    def get_recommended_model(task: str, language: str = "en") -> Optional[str]:
        """Get recommended model for task and language"""
        task_models = RECOMMENDED_MODELS.get(task, {})
        
        if language in task_models:
            return task_models[language]
        if "multilingual" in task_models:
            return task_models["multilingual"]
        if "en" in task_models:
            return task_models["en"]
        
        return None


def create_hf_engine(
    model_name: Optional[str] = None,
    task_type: HFTaskType = HFTaskType.TOKEN_CLASSIFICATION,
    language: str = "en",
    use_gpu: bool = False
) -> HuggingFaceEngine:
    """Factory function to create HuggingFace engine"""
    config = HuggingFaceConfig(
        model_name=model_name,
        task_type=task_type,
        language=language,
        use_gpu=use_gpu
    )
    return HuggingFaceEngine(config)


def create_ner_engine(language: str = "en", use_gpu: bool = False) -> HuggingFaceEngine:
    """Create NER engine"""
    model = RECOMMENDED_MODELS["ner"].get(language, RECOMMENDED_MODELS["ner"]["multilingual"])
    return create_hf_engine(model, HFTaskType.NER, language, use_gpu)


def create_embedding_engine(language: str = "en", use_gpu: bool = False) -> HuggingFaceEngine:
    """Create embedding engine"""
    model = RECOMMENDED_MODELS["embeddings"].get(language, RECOMMENDED_MODELS["embeddings"]["multilingual"])
    return create_hf_engine(model, HFTaskType.FEATURE_EXTRACTION, language, use_gpu)


def create_translation_engine(source_lang: str, target_lang: str, use_gpu: bool = False) -> HuggingFaceEngine:
    """Create translation engine"""
    lang_pair = f"{source_lang}-{target_lang}"
    model = RECOMMENDED_MODELS["translation"].get(lang_pair)
    if model is None:
        model = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    return create_hf_engine(model, HFTaskType.TRANSLATION, source_lang, use_gpu)
