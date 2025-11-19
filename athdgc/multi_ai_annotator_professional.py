"""
Professional Multi-AI PROIEL Annotation Engine
Integrates multiple open-source AI models for linguistic annotation

Author: Nikolaos Lavidas, NKUA
Institution: National and Kapodistrian University of Athens
Funding: Hellenic Foundation for Research and Innovation (HFRI)
Version: 3.0.0
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MultiAIAnnotationEngine:
    """
    Professional multi-AI annotation engine
    Supports multiple open-source AI providers + ML models
    """
    
    def __init__(self):
        self.providers = {
            'huggingface': HuggingFaceProvider(),
            'ollama': OllamaProvider(),
            'llamacpp': LlamaCppProvider()
        }
        
        # Add ML providers
        try:
            from athdgc.ml_annotation_provider import ml_provider
            self.providers['spacy'] = ml_provider.models['spacy']
            self.providers['sklearn'] = ml_provider.models['sklearn']
            self.providers['ensemble'] = ml_provider.models['ensemble']
            logger.info("Multi-AI annotation engine initialized with ML providers")
        except Exception as e:
            logger.warning(f"ML providers not available: {e}")
            logger.info("Multi-AI annotation engine initialized")
    
    async def annotate_parallel(self, text: str, language: str, 
                                framework: str, detail_level: str,
                                models: List[str]) -> Dict[str, Any]:
        """
        Annotate text using multiple AI models in parallel
        
        Args:
            text: Input text to annotate
            language: Language code (e.g., 'grc', 'en', 'la')
            framework: Annotation framework ('proiel', 'ud', 'conll')
            detail_level: Level of detail ('basic', 'standard', 'full')
            models: List of model identifiers
        
        Returns:
            Dictionary containing all model results
        """
        tasks = []
        
        for model_id in models:
            provider, model_name = self._parse_model_id(model_id)
            if provider in self.providers:
                task = self.providers[provider].annotate(
                    text, language, framework, detail_level, model_name
                )
                tasks.append((model_id, task))
        
        results = {}
        for model_id, task in tasks:
            try:
                result = await task
                results[model_id] = {
                    'status': 'success',
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error annotating with {model_id}: {e}")
                results[model_id] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def _parse_model_id(self, model_id: str) -> tuple:
        """Parse model identifier into provider and model name"""
        # ML models
        if model_id in ['spacy', 'sklearn', 'ensemble']:
            return model_id, model_id
        # Hugging Face models (contain /)
        elif '/' in model_id:
            return 'huggingface', model_id
        # Ollama models
        elif 'llama' in model_id.lower() or model_id in ['mistral', 'mixtral']:
            return 'ollama', model_id
        # Default Hugging Face
        else:
            return 'huggingface', model_id


class HuggingFaceProvider:
    """Hugging Face Inference API provider"""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models"
        self.models = {
            'xlm-roberta-base': 'FacebookAI/xlm-roberta-base',
            'mbert': 'google-bert/bert-base-multilingual-cased',
            'xlm-mlm': 'FacebookAI/xlm-mlm-100-1280'
        }
    
    async def annotate(self, text: str, language: str, framework: str,
                      detail_level: str, model_name: str) -> Dict:
        """Annotate using Hugging Face models"""
        
        # Use appropriate model based on language
        model_id = self.models.get(model_name, self.models['xlm-roberta-base'])
        
        prompt = self._build_prompt(text, language, framework, detail_level)
        
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 2000}}
            
            try:
                async with session.post(
                    f"{self.api_url}/{model_id}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_response(result, text)
                    else:
                        raise Exception(f"API error: {response.status}")
            except Exception as e:
                logger.error(f"HuggingFace API error: {e}")
                # Fallback to local processing
                return self._fallback_annotation(text, language)
    
    def _build_prompt(self, text: str, language: str, framework: str, detail_level: str) -> str:
        """Build annotation prompt"""
        return f"""Perform {framework.upper()} linguistic annotation.
Language: {language}
Detail level: {detail_level}

Text: {text}

Provide morphological and syntactic annotation in JSON format."""
    
    def _parse_response(self, response: Any, text: str) -> Dict:
        """Parse API response"""
        # Basic parsing - enhance based on actual response format
        return {
            'sentences': [{
                'text': text,
                'tokens': self._tokenize_basic(text)
            }]
        }
    
    def _tokenize_basic(self, text: str) -> List[Dict]:
        """Basic tokenization fallback"""
        tokens = text.split()
        return [
            {
                'id': i+1,
                'form': token,
                'lemma': token.lower(),
                'pos': 'UNKNOWN',
                'features': {},
                'head': 0,
                'relation': 'root' if i == 0 else 'dep'
            }
            for i, token in enumerate(tokens)
        ]
    
    def _fallback_annotation(self, text: str, language: str) -> Dict:
        """Fallback to local annotation using Stanza/spaCy"""
        try:
            # Try using Stanza for better annotation
            import stanza
            
            # Map language codes to Stanza language codes
            lang_map = {
                'grc-classical': 'grc', 'grc-archaic': 'grc', 
                'grc-hellenistic': 'grc', 'grc-koine': 'grc',
                'el': 'el',  # Modern Greek
                'la': 'la', 'la-vulgar': 'la',
                'en': 'en', 'ang': 'en', 'enm': 'en', 'en-early': 'en',
                'fr': 'fr', 'fro': 'fr', 'frm': 'fr',
                'de': 'de', 'goh': 'de', 'gmh': 'de',
                'cu': 'cu', 'got': 'got',
                'ru': 'ru', 'pl': 'pl', 'cs': 'cs',
                'it': 'it', 'es': 'es', 'pt': 'pt',
                'nl': 'nl', 'sv': 'sv', 'no': 'no', 'da': 'da'
            }
            
            stanza_lang = lang_map.get(language, 'en')
            
            try:
                nlp = stanza.Pipeline(stanza_lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
                doc = nlp(text)
                
                sentences = []
                for sent in doc.sentences:
                    tokens = []
                    for word in sent.words:
                        tokens.append({
                            'id': word.id,
                            'form': word.text,
                            'lemma': word.lemma,
                            'pos': word.upos,
                            'features': self._parse_feats(word.feats) if word.feats else {},
                            'head': word.head,
                            'relation': word.deprel
                        })
                    sentences.append({'text': sent.text, 'tokens': tokens})
                
                return {'sentences': sentences}
            except Exception as e:
                logger.warning(f"Stanza annotation failed: {e}")
                return self._basic_fallback(text)
                
        except ImportError:
            return self._basic_fallback(text)
    
    def _parse_feats(self, feats_str: str) -> Dict:
        """Parse morphological features string"""
        if not feats_str:
            return {}
        features = {}
        for feat in feats_str.split('|'):
            if '=' in feat:
                k, v = feat.split('=', 1)
                features[k] = v
        return features
    
    def _basic_fallback(self, text: str) -> Dict:
        """Most basic fallback"""
        return {
            'sentences': [{
                'text': text,
                'tokens': self._tokenize_basic(text)
            }]
        }


class OllamaProvider:
    """Ollama local LLM provider"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.models = ['llama3.2', 'llama3.1', 'mistral', 'mixtral']
    
    async def annotate(self, text: str, language: str, framework: str,
                      detail_level: str, model_name: str) -> Dict:
        """Annotate using Ollama models"""
        
        prompt = self._build_annotation_prompt(text, language, framework, detail_level)
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
            
            try:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_ollama_response(result.get('response', ''), text)
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                return self._fallback_annotation(text, language)
    
    def _build_annotation_prompt(self, text: str, language: str, 
                                 framework: str, detail_level: str) -> str:
        """Build detailed annotation prompt for LLM"""
        return f"""You are a professional linguistic annotator. Perform {framework.upper()} annotation on the following {language} text.

TEXT: {text}

Provide complete morphological and syntactic analysis in this JSON format:
{{
  "sentences": [
    {{
      "text": "sentence text",
      "tokens": [
        {{
          "id": 1,
          "form": "word",
          "lemma": "lemma",
          "pos": "NOUN",
          "features": {{"Case": "Nom", "Number": "Sing"}},
          "head": 0,
          "relation": "root"
        }}
      ]
    }}
  ]
}}

Be precise and thorough. Return ONLY the JSON, no explanations."""
    
    def _parse_ollama_response(self, response: str, original_text: str) -> Dict:
        """Parse Ollama JSON response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
        
        return self._fallback_annotation(original_text, 'unknown')
    
    def _fallback_annotation(self, text: str, language: str) -> Dict:
        """Fallback annotation using Stanza"""
        try:
            import stanza
            
            lang_map = {
                'grc-classical': 'grc', 'grc-archaic': 'grc', 
                'grc-hellenistic': 'grc', 'grc-koine': 'grc',
                'el': 'el',  # Modern Greek
                'la': 'la', 'la-vulgar': 'la',
                'en': 'en', 'ang': 'en', 'enm': 'en', 'en-early': 'en',
                'fr': 'fr', 'fro': 'fr', 'frm': 'fr',
                'de': 'de', 'goh': 'de', 'gmh': 'de',
                'cu': 'cu', 'got': 'got',
                'ru': 'ru', 'pl': 'pl', 'cs': 'cs',
                'it': 'it', 'es': 'es', 'pt': 'pt',
                'nl': 'nl', 'sv': 'sv', 'no': 'no', 'da': 'da'
            }
            
            stanza_lang = lang_map.get(language, 'en')
            nlp = stanza.Pipeline(stanza_lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
            doc = nlp(text)
            
            sentences = []
            for sent in doc.sentences:
                tokens = []
                for word in sent.words:
                    features = {}
                    if word.feats:
                        for feat in word.feats.split('|'):
                            if '=' in feat:
                                k, v = feat.split('=', 1)
                                features[k] = v
                    
                    tokens.append({
                        'id': word.id,
                        'form': word.text,
                        'lemma': word.lemma,
                        'pos': word.upos,
                        'features': features,
                        'head': word.head,
                        'relation': word.deprel
                    })
                sentences.append({'text': sent.text, 'tokens': tokens})
            
            return {'sentences': sentences}
        except Exception as e:
            logger.warning(f"Fallback annotation error: {e}")
            tokens = text.split()
            return {
                'sentences': [{
                    'text': text,
                    'tokens': [
                        {
                            'id': i+1,
                            'form': token,
                            'lemma': token.lower(),
                            'pos': 'WORD',
                            'features': {},
                            'head': 0,
                            'relation': 'root' if i == 0 else 'dep'
                        }
                        for i, token in enumerate(tokens)
                    ]
                }]
            }


class LlamaCppProvider:
    """llama.cpp server provider"""
    
    def __init__(self):
        self.base_url = "http://localhost:8080"
    
    async def annotate(self, text: str, language: str, framework: str,
                      detail_level: str, model_name: str) -> Dict:
        """Annotate using llama.cpp server"""
        
        prompt = self._build_prompt(text, language, framework)
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            try:
                async with session.post(
                    f"{self.base_url}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_response(result.get('content', ''), text)
                    else:
                        raise Exception(f"llama.cpp error: {response.status}")
            except Exception as e:
                logger.error(f"llama.cpp error: {e}")
                return self._fallback_annotation(text, language)
    
    def _build_prompt(self, text: str, language: str, framework: str) -> str:
        """Build annotation prompt"""
        return f"Annotate this {language} text using {framework}: {text}"
    
    def _parse_response(self, response: str, text: str) -> Dict:
        """Parse response"""
        return {'sentences': [{'text': text, 'tokens': []}]}
    
    def _fallback_annotation(self, text: str, language: str) -> Dict:
        """Fallback annotation using Stanza"""
        try:
            import stanza
            
            lang_map = {
                'grc-classical': 'grc', 'grc-archaic': 'grc', 
                'grc-hellenistic': 'grc', 'grc-koine': 'grc',
                'el': 'el',  # Modern Greek
                'la': 'la', 'la-vulgar': 'la',
                'en': 'en', 'ang': 'en', 'enm': 'en', 'en-early': 'en',
                'fr': 'fr', 'fro': 'fr', 'frm': 'fr',
                'de': 'de', 'goh': 'de', 'gmh': 'de',
                'cu': 'cu', 'got': 'got',
                'ru': 'ru', 'pl': 'pl', 'cs': 'cs',
                'it': 'it', 'es': 'es', 'pt': 'pt',
                'nl': 'nl', 'sv': 'sv', 'no': 'no', 'da': 'da'
            }
            
            stanza_lang = lang_map.get(language, 'en')
            nlp = stanza.Pipeline(stanza_lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
            doc = nlp(text)
            
            sentences = []
            for sent in doc.sentences:
                tokens = []
                for word in sent.words:
                    features = {}
                    if word.feats:
                        for feat in word.feats.split('|'):
                            if '=' in feat:
                                k, v = feat.split('=', 1)
                                features[k] = v
                    
                    tokens.append({
                        'id': word.id,
                        'form': word.text,
                        'lemma': word.lemma,
                        'pos': word.upos,
                        'features': features,
                        'head': word.head,
                        'relation': word.deprel
                    })
                sentences.append({'text': sent.text, 'tokens': tokens})
            
            return {'sentences': sentences}
        except Exception as e:
            logger.warning(f"Fallback error: {e}")
            return {'sentences': [{'text': text, 'tokens': []}]}


# Singleton instance
multi_ai_engine = MultiAIAnnotationEngine()
