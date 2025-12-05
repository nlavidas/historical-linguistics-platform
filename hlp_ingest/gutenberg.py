"""
HLP Ingest Gutenberg - Project Gutenberg Text Fetching

This module provides utilities for fetching texts from Project Gutenberg,
including classical texts in various languages.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum

from hlp_core.models import Language, Period, Document, Corpus

logger = logging.getLogger(__name__)


GUTENBERG_BASE_URL = "https://www.gutenberg.org"
GUTENBERG_MIRROR_URL = "https://gutenberg.pglaf.org"
GUTENBERG_API_URL = "https://gutendex.com"


class GutenbergFormat(Enum):
    """Available formats"""
    TEXT = "text/plain"
    HTML = "text/html"
    EPUB = "application/epub+zip"
    KINDLE = "application/x-mobipocket-ebook"


LANGUAGE_CODES = {
    Language.ANCIENT_GREEK: ["grc", "el"],
    Language.LATIN: ["la"],
    Language.OLD_ENGLISH: ["ang"],
    Language.OLD_NORSE: ["non"],
    Language.GOTHIC: ["got"],
    Language.ENGLISH: ["en"],
    Language.GERMAN: ["de"],
    Language.FRENCH: ["fr"],
    Language.ITALIAN: ["it"],
    Language.SPANISH: ["es"],
}


@dataclass
class GutenbergConfig:
    """Configuration for Gutenberg client"""
    base_url: str = GUTENBERG_API_URL
    
    mirror_url: str = GUTENBERG_MIRROR_URL
    
    cache_dir: Optional[str] = None
    
    rate_limit: float = 1.0
    
    timeout: float = 30.0
    
    max_retries: int = 3
    
    preferred_format: GutenbergFormat = GutenbergFormat.TEXT
    
    user_agent: str = "HLP-Platform/1.0 (Historical Linguistics Platform)"


@dataclass
class GutenbergText:
    """Represents a text from Project Gutenberg"""
    gutenberg_id: int
    title: str
    author: str
    
    language: Language
    
    text_content: str = ""
    
    subjects: List[str] = field(default_factory=list)
    
    bookshelves: List[str] = field(default_factory=list)
    
    download_count: int = 0
    
    formats: Dict[str, str] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_document(self) -> Document:
        """Convert to HLP Document"""
        return Document(
            id=f"gutenberg_{self.gutenberg_id}",
            title=self.title,
            author=self.author,
            language=self.language,
            text=self.text_content,
            metadata={
                "source": "gutenberg",
                "gutenberg_id": self.gutenberg_id,
                "subjects": self.subjects,
                "bookshelves": self.bookshelves,
                "download_count": self.download_count,
                **self.metadata
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gutenberg_id": self.gutenberg_id,
            "title": self.title,
            "author": self.author,
            "language": self.language.value,
            "text_length": len(self.text_content),
            "subjects": self.subjects,
            "bookshelves": self.bookshelves,
            "download_count": self.download_count,
            "formats": list(self.formats.keys())
        }


@dataclass
class GutenbergWork:
    """Represents a work in Gutenberg catalog"""
    gutenberg_id: int
    title: str
    authors: List[str]
    
    languages: List[str]
    
    subjects: List[str] = field(default_factory=list)
    
    bookshelves: List[str] = field(default_factory=list)
    
    download_count: int = 0
    
    formats: Dict[str, str] = field(default_factory=dict)


class GutenbergClient:
    """Client for Project Gutenberg"""
    
    def __init__(self, config: Optional[GutenbergConfig] = None):
        self.config = config or GutenbergConfig()
        self._last_request_time = 0.0
        self._session = None
    
    def _get_session(self):
        """Get or create HTTP session"""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self.config.user_agent
            })
        return self._session
    
    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Make HTTP request with retries"""
        session = self._get_session()
        
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                response = session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {url}")
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _make_json_request(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make JSON request"""
        content = self._make_request(url, params)
        if content:
            import json
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None
    
    def fetch_text(
        self,
        gutenberg_id: int,
        format: Optional[GutenbergFormat] = None
    ) -> Optional[GutenbergText]:
        """Fetch a text by Gutenberg ID"""
        metadata = self._fetch_metadata(gutenberg_id)
        if not metadata:
            return None
        
        format = format or self.config.preferred_format
        text_content = self._fetch_text_content(gutenberg_id, metadata, format)
        
        if not text_content:
            return None
        
        language = self._detect_language(metadata.get("languages", []))
        
        authors = metadata.get("authors", [])
        author = authors[0].get("name", "Unknown") if authors else "Unknown"
        
        return GutenbergText(
            gutenberg_id=gutenberg_id,
            title=metadata.get("title", "Unknown"),
            author=author,
            language=language,
            text_content=text_content,
            subjects=metadata.get("subjects", []),
            bookshelves=metadata.get("bookshelves", []),
            download_count=metadata.get("download_count", 0),
            formats=metadata.get("formats", {}),
            metadata=metadata
        )
    
    def _fetch_metadata(self, gutenberg_id: int) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a book"""
        url = f"{self.config.base_url}/books/{gutenberg_id}"
        return self._make_json_request(url)
    
    def _fetch_text_content(
        self,
        gutenberg_id: int,
        metadata: Dict[str, Any],
        format: GutenbergFormat
    ) -> Optional[str]:
        """Fetch text content"""
        formats = metadata.get("formats", {})
        
        text_url = None
        
        for fmt_type, url in formats.items():
            if format.value in fmt_type:
                text_url = url
                break
        
        if not text_url:
            for fmt_type, url in formats.items():
                if "text/plain" in fmt_type:
                    text_url = url
                    break
        
        if not text_url:
            text_url = f"{self.config.mirror_url}/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
        
        content = self._make_request(text_url)
        
        if content:
            content = self._clean_gutenberg_text(content)
        
        return content
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Gutenberg text by removing headers/footers"""
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
        ]
        
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]
        
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                newline_pos = text.find("\n", pos)
                if newline_pos != -1:
                    start_pos = newline_pos + 1
                break
        
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                end_pos = pos
                break
        
        return text[start_pos:end_pos].strip()
    
    def _detect_language(self, languages: List[str]) -> Language:
        """Detect language from language codes"""
        for lang in languages:
            lang_lower = lang.lower()
            for hlp_lang, codes in LANGUAGE_CODES.items():
                if lang_lower in codes:
                    return hlp_lang
        
        return Language.ENGLISH
    
    def search(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[Language] = None,
        topic: Optional[str] = None,
        max_results: int = 50
    ) -> List[GutenbergWork]:
        """Search Gutenberg catalog"""
        params = {}
        
        if query:
            params["search"] = query
        if author:
            params["author"] = author
        if topic:
            params["topic"] = topic
        
        if language:
            codes = LANGUAGE_CODES.get(language, [])
            if codes:
                params["languages"] = codes[0]
        
        results = []
        url = f"{self.config.base_url}/books"
        
        while len(results) < max_results:
            data = self._make_json_request(url, params)
            if not data:
                break
            
            for book in data.get("results", []):
                authors = [a.get("name", "Unknown") for a in book.get("authors", [])]
                
                results.append(GutenbergWork(
                    gutenberg_id=book.get("id"),
                    title=book.get("title", "Unknown"),
                    authors=authors,
                    languages=book.get("languages", []),
                    subjects=book.get("subjects", []),
                    bookshelves=book.get("bookshelves", []),
                    download_count=book.get("download_count", 0),
                    formats=book.get("formats", {})
                ))
                
                if len(results) >= max_results:
                    break
            
            next_url = data.get("next")
            if not next_url:
                break
            url = next_url
            params = {}
        
        return results
    
    def list_languages(self) -> List[str]:
        """List available languages"""
        return list(LANGUAGE_CODES.keys())
    
    def fetch_by_language(
        self,
        language: Language,
        max_texts: int = 100
    ) -> List[GutenbergText]:
        """Fetch texts by language"""
        works = self.search(language=language, max_results=max_texts)
        
        texts = []
        for work in works:
            text = self.fetch_text(work.gutenberg_id)
            if text:
                texts.append(text)
        
        return texts
    
    def fetch_corpus(
        self,
        gutenberg_ids: List[int],
        corpus_name: str = "gutenberg_corpus"
    ) -> Corpus:
        """Fetch multiple texts as a corpus"""
        documents = []
        
        for gid in gutenberg_ids:
            text = self.fetch_text(gid)
            if text:
                documents.append(text.to_document())
        
        return Corpus(
            id=corpus_name,
            name=corpus_name,
            documents=documents,
            metadata={"source": "gutenberg", "gutenberg_ids": gutenberg_ids}
        )


def fetch_gutenberg_text(
    gutenberg_id: int,
    config: Optional[GutenbergConfig] = None
) -> Optional[GutenbergText]:
    """Fetch a text from Project Gutenberg"""
    client = GutenbergClient(config)
    return client.fetch_text(gutenberg_id)


def search_gutenberg(
    query: Optional[str] = None,
    author: Optional[str] = None,
    language: Optional[Language] = None,
    config: Optional[GutenbergConfig] = None
) -> List[GutenbergWork]:
    """Search Project Gutenberg catalog"""
    client = GutenbergClient(config)
    return client.search(query=query, author=author, language=language)


def list_gutenberg_languages() -> List[str]:
    """List available languages in Gutenberg"""
    return list(LANGUAGE_CODES.keys())
