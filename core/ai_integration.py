#!/usr/bin/env python3
"""
AI Integration Hub for Greek Diachronic Linguistics Platform
"Windsurf for Greek Linguistics"

Integrates community-driven open-source AI services online:
- No local downloads required
- Direct API access through the platform
- ERC-quality research standards

Supported Services:
1. Hugging Face Inference API
2. OpenAI API (optional)
3. Anthropic Claude API (optional)
4. Google Gemini API (optional)
5. Ollama (self-hosted)
6. LM Studio API
7. Together AI
8. Replicate
9. Groq
10. Perplexity
"""

import os
import json
import logging
import hashlib
import sqlite3
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

AI_CONFIG = {
    "default_provider": "huggingface",
    "timeout": 60,
    "max_retries": 3,
    "rate_limit_delay": 1.0,
    "cache_responses": True,
    "log_all_requests": True
}

# ============================================================================
# ENUMS
# ============================================================================

class AIProvider(Enum):
    """Supported AI providers"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TOGETHER = "together"
    REPLICATE = "replicate"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class TaskType(Enum):
    """AI task types"""
    TEXT_GENERATION = "text_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    NER = "ner"
    POS_TAGGING = "pos_tagging"
    LEMMATIZATION = "lemmatization"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    QA = "question_answering"
    SIMILARITY = "similarity"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AIRequest:
    """AI request"""
    id: str
    task: TaskType
    provider: AIProvider
    model: str
    input_text: str
    parameters: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"


@dataclass
class AIResponse:
    """AI response"""
    request_id: str
    output: Any
    model: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0
    cached: bool = False
    error: str = ""
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# BASE PROVIDER
# ============================================================================

class BaseAIProvider(ABC):
    """Base class for AI providers"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get(f"{self.provider_name.upper()}_API_KEY", "")
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(calls_per_minute=60)
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, model: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass
    
    def _make_request(self, endpoint: str, payload: Dict, 
                      method: str = "POST") -> Dict:
        """Make API request with rate limiting and retries"""
        self.rate_limiter.wait()
        
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        
        for attempt in range(AI_CONFIG["max_retries"]):
            try:
                if method == "POST":
                    response = self.session.post(
                        url, json=payload, headers=headers,
                        timeout=AI_CONFIG["timeout"]
                    )
                else:
                    response = self.session.get(
                        url, headers=headers,
                        timeout=AI_CONFIG["timeout"]
                    )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return {"error": response.text}
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < AI_CONFIG["max_retries"] - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def _get_headers(self) -> Dict:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


# ============================================================================
# HUGGING FACE PROVIDER
# ============================================================================

class HuggingFaceProvider(BaseAIProvider):
    """Hugging Face Inference API provider"""
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    @property
    def base_url(self) -> str:
        return "https://api-inference.huggingface.co/models"
    
    # Greek-specific models
    GREEK_MODELS = {
        "translation": "Helsinki-NLP/opus-mt-el-en",
        "ner": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "classification": "nlpaueb/bert-base-greek-uncased-v1",
        "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "generation": "bigscience/bloom-560m",
        "summarization": "facebook/mbart-large-50-many-to-many-mmt",
        "pos": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "ancient_greek": "pranaydeeps/Ancient-Greek-BERT"
    }
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        if not self.api_key:
            self.api_key = os.environ.get("HF_TOKEN", "")
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text"""
        model = model or self.GREEK_MODELS["generation"]
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True
            }
        }
        
        response = self._make_request(model, payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        if isinstance(response, list) and len(response) > 0:
            return response[0].get("generated_text", "")
        
        return str(response)
    
    def translate(self, text: str, source_lang: str = "el", 
                  target_lang: str = "en") -> str:
        """Translate text"""
        model = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        payload = {"inputs": text}
        response = self._make_request(model, payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        if isinstance(response, list) and len(response) > 0:
            return response[0].get("translation_text", "")
        
        return str(response)
    
    def classify(self, text: str, labels: List[str] = None) -> Dict:
        """Classify text"""
        model = "facebook/bart-large-mnli"
        
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": labels or ["positive", "negative", "neutral"]
            }
        }
        
        response = self._make_request(model, payload)
        return response
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings"""
        model = self.GREEK_MODELS["embedding"]
        
        payload = {"inputs": texts}
        response = self._make_request(model, payload)
        
        if "error" in response:
            return []
        
        return response if isinstance(response, list) else []
    
    def named_entity_recognition(self, text: str) -> List[Dict]:
        """Perform NER"""
        model = self.GREEK_MODELS["ner"]
        
        payload = {"inputs": text}
        response = self._make_request(model, payload)
        
        if "error" in response:
            return []
        
        return response if isinstance(response, list) else []
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text"""
        model = "facebook/bart-large-cnn"
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": 30
            }
        }
        
        response = self._make_request(model, payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        if isinstance(response, list) and len(response) > 0:
            return response[0].get("summary_text", "")
        
        return str(response)
    
    def get_available_models(self) -> List[str]:
        """Get available models"""
        return list(self.GREEK_MODELS.values())


# ============================================================================
# TOGETHER AI PROVIDER
# ============================================================================

class TogetherAIProvider(BaseAIProvider):
    """Together AI provider - open source models"""
    
    @property
    def provider_name(self) -> str:
        return "together"
    
    @property
    def base_url(self) -> str:
        return "https://api.together.xyz/v1"
    
    MODELS = {
        "llama3": "meta-llama/Llama-3-70b-chat-hf",
        "llama3_8b": "meta-llama/Llama-3-8b-chat-hf",
        "mistral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen": "Qwen/Qwen2-72B-Instruct",
        "codellama": "codellama/CodeLlama-34b-Instruct-hf"
    }
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text"""
        model = model or self.MODELS["llama3_8b"]
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        response = self._make_request("chat/completions", payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(response)
    
    def get_available_models(self) -> List[str]:
        return list(self.MODELS.values())


# ============================================================================
# GROQ PROVIDER
# ============================================================================

class GroqProvider(BaseAIProvider):
    """Groq provider - fast inference"""
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    @property
    def base_url(self) -> str:
        return "https://api.groq.com/openai/v1"
    
    MODELS = {
        "llama3_70b": "llama3-70b-8192",
        "llama3_8b": "llama3-8b-8192",
        "mixtral": "mixtral-8x7b-32768",
        "gemma": "gemma-7b-it"
    }
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text with Groq"""
        model = model or self.MODELS["llama3_8b"]
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = self._make_request("chat/completions", payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(response)
    
    def get_available_models(self) -> List[str]:
        return list(self.MODELS.values())


# ============================================================================
# REPLICATE PROVIDER
# ============================================================================

class ReplicateProvider(BaseAIProvider):
    """Replicate provider - run open source models"""
    
    @property
    def provider_name(self) -> str:
        return "replicate"
    
    @property
    def base_url(self) -> str:
        return "https://api.replicate.com/v1"
    
    MODELS = {
        "llama3": "meta/meta-llama-3-70b-instruct",
        "llama3_8b": "meta/meta-llama-3-8b-instruct",
        "mistral": "mistralai/mixtral-8x7b-instruct-v0.1"
    }
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text"""
        model = model or self.MODELS["llama3_8b"]
        
        payload = {
            "version": model,
            "input": {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }
        
        response = self._make_request("predictions", payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        # Replicate returns async - need to poll for result
        prediction_id = response.get("id")
        if prediction_id:
            return self._wait_for_prediction(prediction_id)
        
        return str(response)
    
    def _wait_for_prediction(self, prediction_id: str, 
                              max_wait: int = 60) -> str:
        """Wait for prediction to complete"""
        start = time.time()
        
        while time.time() - start < max_wait:
            response = self._make_request(
                f"predictions/{prediction_id}", {}, method="GET"
            )
            
            status = response.get("status")
            if status == "succeeded":
                output = response.get("output", "")
                if isinstance(output, list):
                    return "".join(output)
                return str(output)
            elif status == "failed":
                return f"Error: {response.get('error', 'Unknown error')}"
            
            time.sleep(1)
        
        return "Error: Prediction timeout"
    
    def get_available_models(self) -> List[str]:
        return list(self.MODELS.values())


# ============================================================================
# PERPLEXITY PROVIDER
# ============================================================================

class PerplexityProvider(BaseAIProvider):
    """Perplexity AI provider - with search"""
    
    @property
    def provider_name(self) -> str:
        return "perplexity"
    
    @property
    def base_url(self) -> str:
        return "https://api.perplexity.ai"
    
    MODELS = {
        "sonar_small": "llama-3.1-sonar-small-128k-online",
        "sonar_large": "llama-3.1-sonar-large-128k-online",
        "sonar_huge": "llama-3.1-sonar-huge-128k-online"
    }
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text with web search"""
        model = model or self.MODELS["sonar_small"]
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = self._make_request("chat/completions", payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(response)
    
    def search_and_answer(self, query: str) -> Dict:
        """Search web and answer"""
        prompt = f"Search for and answer: {query}"
        answer = self.generate(prompt)
        return {"query": query, "answer": answer}
    
    def get_available_models(self) -> List[str]:
        return list(self.MODELS.values())


# ============================================================================
# OLLAMA PROVIDER (Self-hosted)
# ============================================================================

class OllamaProvider(BaseAIProvider):
    """Ollama provider - self-hosted models"""
    
    def __init__(self, api_key: str = None, host: str = "http://localhost:11434"):
        self.host = host
        super().__init__(api_key)
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def base_url(self) -> str:
        return self.host
    
    def generate(self, prompt: str, model: str = "llama3", **kwargs) -> str:
        """Generate text"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 512)
            }
        }
        
        response = self._make_request("api/generate", payload)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        return response.get("response", str(response))
    
    def get_available_models(self) -> List[str]:
        """Get locally available models"""
        response = self._make_request("api/tags", {}, method="GET")
        
        if "error" in response:
            return []
        
        models = response.get("models", [])
        return [m.get("name", "") for m in models]
    
    def _get_headers(self) -> Dict:
        return {"Content-Type": "application/json"}


# ============================================================================
# AI HUB - UNIFIED INTERFACE
# ============================================================================

class AIHub:
    """Unified AI interface for the platform"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self.providers: Dict[str, BaseAIProvider] = {}
        self.cache = ResponseCache(db_path)
        
        self._init_providers()
        self._init_db()
    
    def _init_providers(self):
        """Initialize available providers"""
        # Always available (free tier)
        self.providers["huggingface"] = HuggingFaceProvider()
        
        # Optional providers (require API keys)
        if os.environ.get("TOGETHER_API_KEY"):
            self.providers["together"] = TogetherAIProvider()
        
        if os.environ.get("GROQ_API_KEY"):
            self.providers["groq"] = GroqProvider()
        
        if os.environ.get("REPLICATE_API_TOKEN"):
            self.providers["replicate"] = ReplicateProvider()
        
        if os.environ.get("PERPLEXITY_API_KEY"):
            self.providers["perplexity"] = PerplexityProvider()
        
        # Self-hosted
        self.providers["ollama"] = OllamaProvider()
    
    def _init_db(self):
        """Initialize AI tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_requests (
                id TEXT PRIMARY KEY,
                task TEXT,
                provider TEXT,
                model TEXT,
                input_text TEXT,
                output_text TEXT,
                tokens_used INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                cached INTEGER DEFAULT 0,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT,
                model TEXT,
                tokens INTEGER,
                cost_estimate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_provider(self, name: str) -> Optional[BaseAIProvider]:
        """Get provider by name"""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self.providers.keys())
    
    def list_models(self, provider: str = None) -> Dict[str, List[str]]:
        """List available models"""
        if provider:
            p = self.providers.get(provider)
            if p:
                return {provider: p.get_available_models()}
            return {}
        
        return {
            name: p.get_available_models() 
            for name, p in self.providers.items()
        }
    
    # ========================================================================
    # GREEK LINGUISTICS TASKS
    # ========================================================================
    
    def translate_greek(self, text: str, target_lang: str = "en",
                        source_period: str = None) -> Dict:
        """Translate Greek text"""
        # Check cache
        cache_key = f"translate:{text}:{target_lang}"
        cached = self.cache.get(cache_key)
        if cached:
            return {"translation": cached, "cached": True}
        
        # Use Hugging Face for translation
        hf = self.providers.get("huggingface")
        if hf:
            translation = hf.translate(text, "el", target_lang)
            self.cache.set(cache_key, translation)
            self._log_request("translate", "huggingface", text, translation)
            return {"translation": translation, "cached": False}
        
        return {"error": "No translation provider available"}
    
    def analyze_greek_text(self, text: str) -> Dict:
        """Comprehensive Greek text analysis"""
        results = {
            "text": text,
            "analysis": {}
        }
        
        hf = self.providers.get("huggingface")
        if not hf:
            return {"error": "Hugging Face provider not available"}
        
        # NER
        try:
            entities = hf.named_entity_recognition(text)
            results["analysis"]["entities"] = entities
        except Exception as e:
            results["analysis"]["entities_error"] = str(e)
        
        # Embeddings
        try:
            embeddings = hf.get_embeddings([text])
            results["analysis"]["embedding_dim"] = len(embeddings[0]) if embeddings else 0
        except Exception as e:
            results["analysis"]["embedding_error"] = str(e)
        
        # Translation
        try:
            translation = hf.translate(text, "el", "en")
            results["analysis"]["translation_en"] = translation
        except Exception as e:
            results["analysis"]["translation_error"] = str(e)
        
        return results
    
    def classify_period(self, text: str) -> Dict:
        """Classify Greek text by historical period"""
        periods = [
            "Archaic Greek (800-500 BCE)",
            "Classical Greek (500-323 BCE)",
            "Hellenistic Greek (323-31 BCE)",
            "Roman Period Greek (31 BCE-300 CE)",
            "Byzantine Greek (300-1453 CE)",
            "Early Modern Greek (1453-1830 CE)"
        ]
        
        hf = self.providers.get("huggingface")
        if hf:
            result = hf.classify(text, periods)
            return result
        
        return {"error": "Classification not available"}
    
    def classify_genre(self, text: str) -> Dict:
        """Classify Greek text by genre"""
        genres = [
            "Epic poetry",
            "Lyric poetry",
            "Tragedy",
            "Comedy",
            "Historical prose",
            "Philosophical prose",
            "Oratory",
            "Religious text",
            "Scientific text",
            "Legal document"
        ]
        
        hf = self.providers.get("huggingface")
        if hf:
            result = hf.classify(text, genres)
            return result
        
        return {"error": "Classification not available"}
    
    def generate_linguistic_analysis(self, text: str, 
                                      analysis_type: str = "morphological") -> str:
        """Generate linguistic analysis using LLM"""
        prompts = {
            "morphological": f"""Analyze the morphology of this Greek text. 
For each word, identify: part of speech, case, number, gender, tense, mood, voice.
Text: {text}

Provide detailed morphological analysis:""",
            
            "syntactic": f"""Analyze the syntax of this Greek text.
Identify: sentence structure, clause types, word order patterns, dependencies.
Text: {text}

Provide detailed syntactic analysis:""",
            
            "semantic": f"""Analyze the semantics of this Greek text.
Identify: semantic roles, argument structure, thematic relations.
Text: {text}

Provide detailed semantic analysis:""",
            
            "diachronic": f"""Analyze this Greek text from a diachronic perspective.
Identify: period-specific features, archaisms, innovations, language change indicators.
Text: {text}

Provide detailed diachronic analysis:""",
            
            "stylistic": f"""Analyze the style of this Greek text.
Identify: register, genre markers, rhetorical devices, authorial features.
Text: {text}

Provide detailed stylistic analysis:"""
        }
        
        prompt = prompts.get(analysis_type, prompts["morphological"])
        
        # Try providers in order of preference
        for provider_name in ["groq", "together", "huggingface"]:
            provider = self.providers.get(provider_name)
            if provider:
                try:
                    result = provider.generate(prompt, max_tokens=1024)
                    if not result.startswith("Error"):
                        self._log_request(f"analysis_{analysis_type}", 
                                         provider_name, text, result)
                        return result
                except Exception as e:
                    logger.error(f"Provider {provider_name} failed: {e}")
                    continue
        
        return "Error: No AI provider available for analysis"
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """Compare two Greek texts"""
        hf = self.providers.get("huggingface")
        if not hf:
            return {"error": "Embedding provider not available"}
        
        # Get embeddings
        embeddings = hf.get_embeddings([text1, text2])
        
        if len(embeddings) == 2:
            # Calculate cosine similarity
            import math
            
            def cosine_sim(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b) if norm_a and norm_b else 0
            
            similarity = cosine_sim(embeddings[0], embeddings[1])
            
            return {
                "text1": text1[:100] + "...",
                "text2": text2[:100] + "...",
                "similarity": similarity,
                "interpretation": self._interpret_similarity(similarity)
            }
        
        return {"error": "Could not compute embeddings"}
    
    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score"""
        if score > 0.9:
            return "Very similar - likely same author/period/genre"
        elif score > 0.7:
            return "Similar - possibly related texts"
        elif score > 0.5:
            return "Moderately similar"
        elif score > 0.3:
            return "Somewhat different"
        else:
            return "Very different - likely different periods/genres"
    
    def answer_question(self, question: str, context: str = None) -> str:
        """Answer question about Greek linguistics"""
        if context:
            prompt = f"""Context: {context}

Question: {question}

Answer based on the context and your knowledge of Greek linguistics:"""
        else:
            prompt = f"""You are an expert in Greek diachronic linguistics.

Question: {question}

Provide a detailed, scholarly answer:"""
        
        # Try Perplexity first (has search)
        if "perplexity" in self.providers:
            result = self.providers["perplexity"].generate(prompt)
            if not result.startswith("Error"):
                return result
        
        # Fall back to other providers
        for provider_name in ["groq", "together", "huggingface"]:
            provider = self.providers.get(provider_name)
            if provider:
                try:
                    result = provider.generate(prompt, max_tokens=1024)
                    if not result.startswith("Error"):
                        return result
                except Exception:
                    continue
        
        return "Error: No AI provider available"
    
    def _log_request(self, task: str, provider: str, 
                     input_text: str, output_text: str):
        """Log AI request"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            request_id = hashlib.md5(
                f"{task}:{input_text}:{datetime.now()}".encode()
            ).hexdigest()[:16]
            
            cursor.execute("""
                INSERT INTO ai_requests
                (id, task, provider, input_text, output_text)
                VALUES (?, ?, ?, ?, ?)
            """, (request_id, task, provider, input_text[:1000], output_text[:2000]))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    def get_usage_stats(self) -> Dict:
        """Get AI usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("""
            SELECT provider, COUNT(*) as count
            FROM ai_requests
            GROUP BY provider
        """)
        stats["by_provider"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT task, COUNT(*) as count
            FROM ai_requests
            GROUP BY task
        """)
        stats["by_task"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) FROM ai_requests")
        stats["total_requests"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """Cache for AI responses"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_table()
    
    def _init_table(self):
        """Initialize cache table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT value FROM ai_cache 
            WHERE key = ? AND (expires_at IS NULL OR expires_at > datetime('now'))
        """, (key,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None
    
    def set(self, key: str, value: str, ttl_hours: int = 24):
        """Set cached value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO ai_cache (key, value, expires_at)
            VALUES (?, ?, datetime('now', '+' || ? || ' hours'))
        """, (key, value, ttl_hours))
        
        conn.commit()
        conn.close()
    
    def clear(self):
        """Clear cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ai_cache")
        conn.commit()
        conn.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Integration Hub")
    parser.add_argument('command', choices=['providers', 'models', 'translate', 
                                            'analyze', 'classify', 'question'],
                       help="Command to run")
    parser.add_argument('--text', '-t', help="Input text")
    parser.add_argument('--provider', '-p', help="AI provider")
    parser.add_argument('--type', help="Analysis type")
    
    args = parser.parse_args()
    
    hub = AIHub()
    
    if args.command == 'providers':
        providers = hub.list_providers()
        print("Available providers:")
        for p in providers:
            print(f"  - {p}")
    
    elif args.command == 'models':
        models = hub.list_models(args.provider)
        print(json.dumps(models, indent=2))
    
    elif args.command == 'translate':
        if args.text:
            result = hub.translate_greek(args.text)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Please provide --text")
    
    elif args.command == 'analyze':
        if args.text:
            result = hub.generate_linguistic_analysis(
                args.text, args.type or "morphological"
            )
            print(result)
        else:
            print("Please provide --text")
    
    elif args.command == 'classify':
        if args.text:
            result = hub.classify_period(args.text)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Please provide --text")
    
    elif args.command == 'question':
        if args.text:
            result = hub.answer_question(args.text)
            print(result)
        else:
            print("Please provide --text (question)")


if __name__ == "__main__":
    main()
