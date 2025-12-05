"""
HLP Ingest Perseus - Perseus Digital Library Harvesting

This module provides utilities for fetching texts from the Perseus
Digital Library, including Greek and Latin classical texts.

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
import xml.etree.ElementTree as ET

from hlp_core.models import Language, Period, Document, Corpus

logger = logging.getLogger(__name__)


PERSEUS_BASE_URL = "https://www.perseus.tufts.edu"
PERSEUS_CATALOG_URL = f"{PERSEUS_BASE_URL}/hopper/collection"
PERSEUS_TEXT_URL = f"{PERSEUS_BASE_URL}/hopper/text"
PERSEUS_XML_URL = f"{PERSEUS_BASE_URL}/hopper/xmlchunk"

SCAIFE_BASE_URL = "https://scaife.perseus.org"
SCAIFE_API_URL = "https://scaife-cts.perseus.org/api/cts"


class PerseusCollection(Enum):
    """Perseus collections"""
    GREEK = "Perseus:collection:Greco-Roman"
    LATIN = "Perseus:collection:Greco-Roman"
    GREEK_TEXTS = "Perseus:collection:Greek"
    LATIN_TEXTS = "Perseus:collection:Latin"
    RENAISSANCE = "Perseus:collection:Renaissance"
    GERMANIC = "Perseus:collection:Germanic"


@dataclass
class PerseusConfig:
    """Configuration for Perseus client"""
    base_url: str = PERSEUS_BASE_URL
    
    use_scaife: bool = True
    
    cache_dir: Optional[str] = None
    
    rate_limit: float = 1.0
    
    timeout: float = 30.0
    
    max_retries: int = 3
    
    user_agent: str = "HLP-Platform/1.0 (Historical Linguistics Platform)"


@dataclass
class PerseusText:
    """Represents a text from Perseus"""
    urn: str
    title: str
    author: str
    
    language: Language
    
    text_content: str = ""
    
    xml_content: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    sections: List[Dict[str, Any]] = field(default_factory=list)
    
    period: Optional[Period] = None
    
    date: Optional[str] = None
    
    def to_document(self) -> Document:
        """Convert to HLP Document"""
        return Document(
            id=self.urn,
            title=self.title,
            author=self.author,
            language=self.language,
            text=self.text_content,
            metadata={
                "source": "perseus",
                "urn": self.urn,
                "date": self.date,
                **self.metadata
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "urn": self.urn,
            "title": self.title,
            "author": self.author,
            "language": self.language.value,
            "text_length": len(self.text_content),
            "has_xml": self.xml_content is not None,
            "section_count": len(self.sections),
            "period": self.period.value if self.period else None,
            "date": self.date,
            "metadata": self.metadata
        }


@dataclass
class PerseusWork:
    """Represents a work in Perseus catalog"""
    urn: str
    title: str
    author: str
    language: str
    
    description: Optional[str] = None
    
    editions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerseusClient:
    """Client for Perseus Digital Library"""
    
    def __init__(self, config: Optional[PerseusConfig] = None):
        self.config = config or PerseusConfig()
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
    
    def fetch_text(
        self,
        urn: str,
        format: str = "text"
    ) -> Optional[PerseusText]:
        """Fetch a text by URN"""
        if self.config.use_scaife:
            return self._fetch_from_scaife(urn, format)
        else:
            return self._fetch_from_perseus(urn, format)
    
    def _fetch_from_scaife(
        self,
        urn: str,
        format: str = "text"
    ) -> Optional[PerseusText]:
        """Fetch text from Scaife Viewer API"""
        passage_url = f"{SCAIFE_API_URL}?request=GetPassage&urn={urn}"
        
        content = self._make_request(passage_url)
        if not content:
            return None
        
        try:
            text_content = self._parse_cts_response(content)
            
            metadata = self._fetch_metadata(urn)
            
            language = self._detect_language(urn)
            
            return PerseusText(
                urn=urn,
                title=metadata.get("title", "Unknown"),
                author=metadata.get("author", "Unknown"),
                language=language,
                text_content=text_content,
                xml_content=content if format == "xml" else None,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing Scaife response: {e}")
            return None
    
    def _fetch_from_perseus(
        self,
        urn: str,
        format: str = "text"
    ) -> Optional[PerseusText]:
        """Fetch text from Perseus Hopper"""
        doc_param = urn.replace("urn:cts:", "Perseus:text:")
        
        if format == "xml":
            url = f"{PERSEUS_XML_URL}?doc={doc_param}"
        else:
            url = f"{PERSEUS_TEXT_URL}?doc={doc_param}"
        
        content = self._make_request(url)
        if not content:
            return None
        
        try:
            if format == "xml":
                text_content = self._extract_text_from_xml(content)
            else:
                text_content = self._extract_text_from_html(content)
            
            metadata = self._extract_metadata_from_html(content)
            language = self._detect_language(urn)
            
            return PerseusText(
                urn=urn,
                title=metadata.get("title", "Unknown"),
                author=metadata.get("author", "Unknown"),
                language=language,
                text_content=text_content,
                xml_content=content if format == "xml" else None,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing Perseus response: {e}")
            return None
    
    def _parse_cts_response(self, xml_content: str) -> str:
        """Parse CTS API response"""
        try:
            root = ET.fromstring(xml_content)
            
            namespaces = {
                'cts': 'http://chs.harvard.edu/xmlns/cts',
                'tei': 'http://www.tei-c.org/ns/1.0'
            }
            
            passage = root.find('.//cts:passage', namespaces)
            if passage is not None:
                return self._get_element_text(passage)
            
            return self._get_element_text(root)
            
        except ET.ParseError:
            text = re.sub(r'<[^>]+>', '', xml_content)
            return text.strip()
    
    def _extract_text_from_xml(self, xml_content: str) -> str:
        """Extract text from TEI XML"""
        try:
            root = ET.fromstring(xml_content)
            
            namespaces = {
                'tei': 'http://www.tei-c.org/ns/1.0'
            }
            
            body = root.find('.//tei:body', namespaces)
            if body is None:
                body = root.find('.//body')
            
            if body is not None:
                return self._get_element_text(body)
            
            return self._get_element_text(root)
            
        except ET.ParseError:
            text = re.sub(r'<[^>]+>', '', xml_content)
            return text.strip()
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML page"""
        text_match = re.search(
            r'<div[^>]*class="[^"]*text_container[^"]*"[^>]*>(.*?)</div>',
            html_content,
            re.DOTALL | re.IGNORECASE
        )
        
        if text_match:
            text = text_match.group(1)
        else:
            text = html_content
        
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_metadata_from_html(self, html_content: str) -> Dict[str, Any]:
        """Extract metadata from HTML page"""
        metadata = {}
        
        title_match = re.search(r'<title>([^<]+)</title>', html_content)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        author_match = re.search(
            r'<meta[^>]*name="DC\.Creator"[^>]*content="([^"]+)"',
            html_content
        )
        if author_match:
            metadata["author"] = author_match.group(1).strip()
        
        return metadata
    
    def _fetch_metadata(self, urn: str) -> Dict[str, Any]:
        """Fetch metadata for a URN"""
        metadata_url = f"{SCAIFE_API_URL}?request=GetCapabilities&urn={urn}"
        
        content = self._make_request(metadata_url)
        if not content:
            return {}
        
        try:
            root = ET.fromstring(content)
            
            metadata = {}
            
            title_elem = root.find('.//{http://chs.harvard.edu/xmlns/cts}title')
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text
            
            return metadata
            
        except Exception:
            return {}
    
    def _detect_language(self, urn: str) -> Language:
        """Detect language from URN"""
        urn_lower = urn.lower()
        
        if "greeklit" in urn_lower or "grc" in urn_lower:
            return Language.ANCIENT_GREEK
        elif "latinlit" in urn_lower or "lat" in urn_lower:
            return Language.LATIN
        elif "germaniclit" in urn_lower:
            return Language.OLD_ENGLISH
        else:
            return Language.ANCIENT_GREEK
    
    def _get_element_text(self, element: ET.Element) -> str:
        """Get all text from an element"""
        texts = []
        
        if element.text:
            texts.append(element.text)
        
        for child in element:
            texts.append(self._get_element_text(child))
            if child.tail:
                texts.append(child.tail)
        
        return ' '.join(texts)
    
    def search(
        self,
        query: str,
        language: Optional[Language] = None,
        max_results: int = 50
    ) -> List[PerseusWork]:
        """Search Perseus catalog"""
        results = []
        
        search_url = f"{SCAIFE_API_URL}?request=GetCapabilities"
        
        content = self._make_request(search_url)
        if not content:
            return results
        
        try:
            root = ET.fromstring(content)
            
            for work in root.findall('.//{http://chs.harvard.edu/xmlns/cts}work'):
                urn = work.get('urn', '')
                
                if query.lower() not in urn.lower():
                    title_elem = work.find('.//{http://chs.harvard.edu/xmlns/cts}title')
                    title = title_elem.text if title_elem is not None else ""
                    if query.lower() not in title.lower():
                        continue
                
                if language:
                    work_lang = self._detect_language(urn)
                    if work_lang != language:
                        continue
                
                results.append(PerseusWork(
                    urn=urn,
                    title=title_elem.text if title_elem is not None else "Unknown",
                    author="Unknown",
                    language=self._detect_language(urn).value
                ))
                
                if len(results) >= max_results:
                    break
                    
        except Exception as e:
            logger.error(f"Error searching Perseus: {e}")
        
        return results
    
    def list_works(
        self,
        collection: Optional[PerseusCollection] = None,
        language: Optional[Language] = None
    ) -> List[PerseusWork]:
        """List available works"""
        works = []
        
        caps_url = f"{SCAIFE_API_URL}?request=GetCapabilities"
        
        content = self._make_request(caps_url)
        if not content:
            return works
        
        try:
            root = ET.fromstring(content)
            
            for work in root.findall('.//{http://chs.harvard.edu/xmlns/cts}work'):
                urn = work.get('urn', '')
                
                if language:
                    work_lang = self._detect_language(urn)
                    if work_lang != language:
                        continue
                
                title_elem = work.find('.//{http://chs.harvard.edu/xmlns/cts}title')
                
                works.append(PerseusWork(
                    urn=urn,
                    title=title_elem.text if title_elem is not None else "Unknown",
                    author="Unknown",
                    language=self._detect_language(urn).value
                ))
                    
        except Exception as e:
            logger.error(f"Error listing works: {e}")
        
        return works
    
    def fetch_corpus(
        self,
        urns: List[str],
        corpus_name: str = "perseus_corpus"
    ) -> Corpus:
        """Fetch multiple texts as a corpus"""
        documents = []
        
        for urn in urns:
            text = self.fetch_text(urn)
            if text:
                documents.append(text.to_document())
        
        return Corpus(
            id=corpus_name,
            name=corpus_name,
            documents=documents,
            metadata={"source": "perseus", "urns": urns}
        )


def fetch_perseus_text(
    urn: str,
    config: Optional[PerseusConfig] = None
) -> Optional[PerseusText]:
    """Fetch a text from Perseus"""
    client = PerseusClient(config)
    return client.fetch_text(urn)


def search_perseus(
    query: str,
    language: Optional[Language] = None,
    config: Optional[PerseusConfig] = None
) -> List[PerseusWork]:
    """Search Perseus catalog"""
    client = PerseusClient(config)
    return client.search(query, language)


def list_perseus_works(
    language: Optional[Language] = None,
    config: Optional[PerseusConfig] = None
) -> List[PerseusWork]:
    """List available works in Perseus"""
    client = PerseusClient(config)
    return client.list_works(language=language)
