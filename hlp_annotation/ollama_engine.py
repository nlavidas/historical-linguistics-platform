"""
HLP Annotation Ollama Engine - Ollama Local LLM Integration

This module provides annotation capabilities using Ollama for running
local large language models, with support for various NLP tasks
including text generation, analysis, and annotation.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
import re
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


OLLAMA_MODELS = {
    "general": [
        "llama2",
        "llama2:13b",
        "llama2:70b",
        "mistral",
        "mixtral",
        "phi",
        "gemma",
        "gemma:7b",
        "qwen",
        "qwen:14b",
    ],
    "code": [
        "codellama",
        "codellama:13b",
        "deepseek-coder",
        "starcoder",
    ],
    "small": [
        "phi",
        "tinyllama",
        "orca-mini",
    ],
    "multilingual": [
        "llama2",
        "mistral",
        "qwen",
    ]
}

ANNOTATION_PROMPTS = {
    "pos_tagging": """Analyze the following text and provide POS tags for each word.
Return the result as a JSON array where each element has "word" and "pos" fields.
Use Universal Dependencies POS tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X.

Text: {text}

JSON output:""",

    "ner": """Identify named entities in the following text.
Return the result as a JSON array where each element has "text", "type", "start", and "end" fields.
Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, EVENT, MISC.

Text: {text}

JSON output:""",

    "lemmatization": """Provide lemmas for each word in the following text.
Return the result as a JSON array where each element has "word" and "lemma" fields.

Text: {text}

JSON output:""",

    "morphology": """Analyze the morphological features of each word in the following text.
Return the result as a JSON array where each element has "word", "pos", "case", "number", "gender", "person", "tense", "mood", "voice" fields.
Use null for features that don't apply.

Text: {text}

JSON output:""",

    "dependency": """Analyze the dependency structure of the following sentence.
Return the result as a JSON array where each element has "id", "word", "head", "deprel" fields.
Use Universal Dependencies relations.

Text: {text}

JSON output:""",

    "translation": """Translate the following {source_lang} text to {target_lang}.
Provide only the translation without any explanation.

Text: {text}

Translation:""",

    "analysis": """Analyze the following ancient text and provide:
1. A brief summary
2. Key linguistic features
3. Notable vocabulary
4. Grammatical constructions

Text: {text}

Analysis:""",

    "valency": """Analyze the valency patterns of verbs in the following text.
For each verb, identify:
1. The verb lemma
2. Its arguments (subject, object, oblique)
3. The case marking of each argument

Text: {text}

JSON output:""",
}


@dataclass
class OllamaConfig(AnnotationConfig):
    """Configuration for Ollama engine"""
    model_name: str = "llama2"
    
    host: str = "http://localhost:11434"
    
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    
    num_predict: int = 1024
    num_ctx: int = 4096
    
    repeat_penalty: float = 1.1
    
    system_prompt: Optional[str] = None
    
    format: Optional[str] = None
    
    keep_alive: str = "5m"
    
    request_timeout: float = 120.0
    
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    
    def get_generation_options(self) -> Dict[str, Any]:
        """Get generation options for Ollama"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty,
        }


class OllamaEngine(AnnotationEngine):
    """Ollama-based annotation engine"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        super().__init__(config or OllamaConfig())
        self._client = None
        self._available_models: List[str] = []
    
    @property
    def name(self) -> str:
        return "OllamaEngine"
    
    @property
    def version(self) -> str:
        config = self.config
        model = config.model_name if isinstance(config, OllamaConfig) else "llama2"
        return f"Ollama ({model})"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        return [
            AnnotationCapability.POS_TAGGING,
            AnnotationCapability.LEMMATIZATION,
            AnnotationCapability.NAMED_ENTITY_RECOGNITION,
            AnnotationCapability.MORPHOLOGICAL_ANALYSIS,
            AnnotationCapability.DEPENDENCY_PARSING,
            AnnotationCapability.TRANSLATION,
            AnnotationCapability.GENERATION,
            AnnotationCapability.TEXT_CLASSIFICATION,
        ]
    
    @property
    def supported_languages(self) -> List[str]:
        return ["grc", "la", "en", "de", "fr", "es", "it", "pt", "ru", "el", "zh", "ja", "ar", "multilingual"]
    
    def initialize(self) -> bool:
        """Initialize Ollama connection"""
        if self._initialized:
            return True
        
        try:
            import requests
            
            self._status = EngineStatus.INITIALIZING
            
            config = self.config
            host = config.host if isinstance(config, OllamaConfig) else "http://localhost:11434"
            
            try:
                response = requests.get(f"{host}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    logger.info(f"Ollama available models: {self._available_models}")
            except Exception as e:
                logger.warning(f"Could not fetch Ollama models: {e}")
            
            model_name = config.model_name if isinstance(config, OllamaConfig) else "llama2"
            
            test_response = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30
            )
            
            if test_response.status_code != 200:
                logger.warning(f"Model {model_name} may not be available, attempting to pull...")
                self._pull_model(model_name)
            
            self._status = EngineStatus.READY
            self._initialized = True
            logger.info(f"Ollama engine initialized with model: {model_name}")
            return True
            
        except ImportError:
            logger.error("requests is not installed. Install with: pip install requests")
            self._status = EngineStatus.ERROR
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize Ollama: {e}")
            self._status = EngineStatus.ERROR
            return False
    
    def shutdown(self):
        """Shutdown Ollama connection"""
        self._initialized = False
        self._status = EngineStatus.SHUTDOWN
        logger.info("Ollama engine shutdown")
    
    def _pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama"""
        try:
            import requests
            
            config = self.config
            host = config.host if isinstance(config, OllamaConfig) else "http://localhost:11434"
            
            response = requests.post(
                f"{host}/api/pull",
                json={"name": model_name},
                timeout=600
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def _generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using Ollama"""
        import requests
        
        config = self.config
        if isinstance(config, OllamaConfig):
            host = config.host
            model = config.model_name
            options = config.get_generation_options()
            timeout = config.request_timeout
            system_prompt = system or config.system_prompt
        else:
            host = "http://localhost:11434"
            model = "llama2"
            options = {}
            timeout = 120.0
            system_prompt = system
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        return data.get("response", "")
    
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with Ollama"""
        import requests
        
        config = self.config
        if isinstance(config, OllamaConfig):
            host = config.host
            model = config.model_name
            options = config.get_generation_options()
            timeout = config.request_timeout
        else:
            host = "http://localhost:11434"
            model = "llama2"
            options = {}
            timeout = 120.0
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        
        response = requests.post(
            f"{host}/api/chat",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text with Ollama"""
        try:
            tokens = []
            entities = []
            semantic_roles = []
            metadata = {}
            
            if AnnotationCapability.POS_TAGGING in capabilities:
                pos_result = self._annotate_pos(text)
                if pos_result:
                    tokens = pos_result
            
            if AnnotationCapability.NAMED_ENTITY_RECOGNITION in capabilities:
                ner_result = self._annotate_ner(text)
                if ner_result:
                    entities = ner_result
            
            if AnnotationCapability.LEMMATIZATION in capabilities and tokens:
                lemma_result = self._annotate_lemmas(text)
                if lemma_result:
                    for token, lemma_info in zip(tokens, lemma_result):
                        if isinstance(lemma_info, dict):
                            token.lemma = lemma_info.get("lemma")
            
            if not tokens:
                words = text.split()
                for idx, word in enumerate(words, start=1):
                    token = Token(id=idx, form=word)
                    tokens.append(token)
            
            return AnnotationResult(
                success=True,
                tokens=tokens,
                entities=entities,
                semantic_roles=semantic_roles,
                tokens_processed=len(tokens),
                metadata=metadata
            )
            
        except Exception as e:
            logger.exception(f"Ollama processing error: {e}")
            return AnnotationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _annotate_pos(self, text: str) -> List[Token]:
        """Annotate POS tags"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["pos_tagging"]
        if isinstance(config, OllamaConfig) and "pos_tagging" in config.custom_prompts:
            prompt_template = config.custom_prompts["pos_tagging"]
        
        prompt = prompt_template.format(text=text)
        response = self._generate(prompt)
        
        tokens = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for idx, item in enumerate(data, start=1):
                    if isinstance(item, dict):
                        word = item.get("word", "")
                        pos_str = item.get("pos", "X")
                        
                        pos_map = {
                            "ADJ": PartOfSpeech.ADJ, "ADP": PartOfSpeech.ADP,
                            "ADV": PartOfSpeech.ADV, "AUX": PartOfSpeech.AUX,
                            "CCONJ": PartOfSpeech.CCONJ, "DET": PartOfSpeech.DET,
                            "INTJ": PartOfSpeech.INTJ, "NOUN": PartOfSpeech.NOUN,
                            "NUM": PartOfSpeech.NUM, "PART": PartOfSpeech.PART,
                            "PRON": PartOfSpeech.PRON, "PROPN": PartOfSpeech.PROPN,
                            "PUNCT": PartOfSpeech.PUNCT, "SCONJ": PartOfSpeech.SCONJ,
                            "SYM": PartOfSpeech.SYM, "VERB": PartOfSpeech.VERB,
                            "X": PartOfSpeech.X,
                        }
                        pos = pos_map.get(pos_str.upper(), PartOfSpeech.X)
                        
                        morphology = MorphologicalFeatures(pos=pos)
                        token = Token(id=idx, form=word, morphology=morphology)
                        tokens.append(token)
        except json.JSONDecodeError:
            logger.warning("Failed to parse POS tagging response as JSON")
        
        return tokens
    
    def _annotate_ner(self, text: str) -> List[NamedEntity]:
        """Annotate named entities"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["ner"]
        if isinstance(config, OllamaConfig) and "ner" in config.custom_prompts:
            prompt_template = config.custom_prompts["ner"]
        
        prompt = prompt_template.format(text=text)
        response = self._generate(prompt)
        
        entities = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict):
                        entity_text = item.get("text", "")
                        entity_type_str = item.get("type", "MISC")
                        start = item.get("start", 0)
                        end = item.get("end", len(entity_text))
                        
                        type_map = {
                            "PERSON": NamedEntityType.PERSON,
                            "PER": NamedEntityType.PERSON,
                            "ORGANIZATION": NamedEntityType.ORGANIZATION,
                            "ORG": NamedEntityType.ORGANIZATION,
                            "LOCATION": NamedEntityType.LOCATION,
                            "LOC": NamedEntityType.LOCATION,
                            "GPE": NamedEntityType.GPE,
                            "DATE": NamedEntityType.DATE,
                            "TIME": NamedEntityType.TIME,
                            "EVENT": NamedEntityType.EVENT,
                            "MISC": NamedEntityType.MISC,
                        }
                        entity_type = type_map.get(entity_type_str.upper(), NamedEntityType.MISC)
                        
                        entity = NamedEntity(
                            entity_type=entity_type,
                            text=entity_text,
                            span_start=start,
                            span_end=end
                        )
                        entities.append(entity)
        except json.JSONDecodeError:
            logger.warning("Failed to parse NER response as JSON")
        
        return entities
    
    def _annotate_lemmas(self, text: str) -> List[Dict[str, str]]:
        """Annotate lemmas"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["lemmatization"]
        if isinstance(config, OllamaConfig) and "lemmatization" in config.custom_prompts:
            prompt_template = config.custom_prompts["lemmatization"]
        
        prompt = prompt_template.format(text=text)
        response = self._generate(prompt)
        
        lemmas = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict):
                        lemmas.append(item)
        except json.JSONDecodeError:
            logger.warning("Failed to parse lemmatization response as JSON")
        
        return lemmas
    
    def translate(
        self,
        text: str,
        source_lang: str = "grc",
        target_lang: str = "en"
    ) -> str:
        """Translate text"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["translation"]
        if isinstance(config, OllamaConfig) and "translation" in config.custom_prompts:
            prompt_template = config.custom_prompts["translation"]
        
        prompt = prompt_template.format(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        return self._generate(prompt)
    
    def analyze_text(self, text: str) -> str:
        """Analyze text"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["analysis"]
        if isinstance(config, OllamaConfig) and "analysis" in config.custom_prompts:
            prompt_template = config.custom_prompts["analysis"]
        
        prompt = prompt_template.format(text=text)
        return self._generate(prompt)
    
    def analyze_valency(self, text: str) -> List[Dict[str, Any]]:
        """Analyze valency patterns"""
        config = self.config
        prompt_template = ANNOTATION_PROMPTS["valency"]
        if isinstance(config, OllamaConfig) and "valency" in config.custom_prompts:
            prompt_template = config.custom_prompts["valency"]
        
        prompt = prompt_template.format(text=text)
        response = self._generate(prompt)
        
        patterns = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse valency response as JSON")
        
        return patterns
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text"""
        if max_tokens and isinstance(self.config, OllamaConfig):
            original = self.config.num_predict
            self.config.num_predict = max_tokens
            result = self._generate(prompt, system)
            self.config.num_predict = original
            return result
        
        return self._generate(prompt, system)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> str:
        """Chat with the model"""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        return self._chat(messages)
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            import requests
            
            config = self.config
            host = config.host if isinstance(config, OllamaConfig) else "http://localhost:11434"
            
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return self._available_models
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model"""
        return self._pull_model(model_name)
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if isinstance(self.config, OllamaConfig):
            self.config.model_name = model_name
            return True
        return False


def create_ollama_engine(
    model_name: str = "llama2",
    host: str = "http://localhost:11434",
    temperature: float = 0.1
) -> OllamaEngine:
    """Factory function to create Ollama engine"""
    config = OllamaConfig(
        model_name=model_name,
        host=host,
        temperature=temperature
    )
    return OllamaEngine(config)
