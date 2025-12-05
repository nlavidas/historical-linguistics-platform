"""
HLP Ingest First1KGreek - First1KGreek Corpus Routines

This module provides utilities for fetching texts from the First1KGreek
corpus, a collection of ancient Greek texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import xml.etree.ElementTree as ET

from hlp_core.models import Language, Period, Document, Corpus

logger = logging.getLogger(__name__)


FIRST1K_GITHUB_URL = "https://github.com/OpenGreekAndLatin/First1KGreek"
FIRST1K_RAW_URL = "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master"
FIRST1K_API_URL = "https://api.github.com/repos/OpenGreekAndLatin/First1KGreek"


@dataclass
class First1KGreekConfig:
    """Configuration for First1KGreek client"""
    github_token: Optional[str] = None
    
    cache_dir: Optional[str] = None
    
    rate_limit: float = 0.5
    
    timeout: float = 30.0
    
    max_retries: int = 3
    
    user_agent: str = "HLP-Platform/1.0 (Historical Linguistics Platform)"


@dataclass
class First1KText:
    """Represents a text from First1KGreek"""
    urn: str
    title: str
    author: str
    
    text_content: str = ""
    
    xml_content: Optional[str] = None
    
    file_path: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    period: Optional[Period] = None
    
    def to_document(self) -> Document:
        """Convert to HLP Document"""
        return Document(
            id=self.urn,
            title=self.title,
            author=self.author,
            language=Language.ANCIENT_GREEK,
            text=self.text_content,
            metadata={
                "source": "first1kgreek",
                "urn": self.urn,
                "file_path": self.file_path,
                **self.metadata
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "urn": self.urn,
            "title": self.title,
            "author": self.author,
            "text_length": len(self.text_content),
            "has_xml": self.xml_content is not None,
            "file_path": self.file_path,
            "period": self.period.value if self.period else None,
            "metadata": self.metadata
        }


@dataclass
class First1KWork:
    """Represents a work in First1KGreek"""
    urn: str
    title: str
    author: str
    
    file_path: str
    
    description: Optional[str] = None
    
    editions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class First1KGreekClient:
    """Client for First1KGreek corpus"""
    
    def __init__(self, config: Optional[First1KGreekConfig] = None):
        self.config = config or First1KGreekConfig()
        self._last_request_time = 0.0
        self._session = None
        self._catalog_cache: Optional[List[First1KWork]] = None
    
    def _get_session(self):
        """Get or create HTTP session"""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {"User-Agent": self.config.user_agent}
            if self.config.github_token:
                headers["Authorization"] = f"token {self.config.github_token}"
            self._session.headers.update(headers)
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
                elif response.status_code == 403:
                    logger.warning("Rate limited by GitHub API")
                    time.sleep(60)
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {url}")
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def fetch_text(self, file_path: str) -> Optional[First1KText]:
        """Fetch a text by file path"""
        if self.config.cache_dir:
            cached = self._load_from_cache(file_path)
            if cached:
                return cached
        
        url = f"{FIRST1K_RAW_URL}/{file_path}"
        content = self._make_request(url)
        
        if not content:
            return None
        
        try:
            text_content = self._extract_text_from_xml(content)
            metadata = self._extract_metadata_from_xml(content)
            
            urn = self._extract_urn(file_path, content)
            author = metadata.get("author", self._extract_author_from_path(file_path))
            title = metadata.get("title", self._extract_title_from_path(file_path))
            
            text = First1KText(
                urn=urn,
                title=title,
                author=author,
                text_content=text_content,
                xml_content=content,
                file_path=file_path,
                metadata=metadata
            )
            
            if self.config.cache_dir:
                self._save_to_cache(file_path, text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error parsing First1KGreek text: {e}")
            return None
    
    def _extract_text_from_xml(self, xml_content: str) -> str:
        """Extract text from TEI XML"""
        try:
            xml_content = re.sub(r'xmlns\s*=\s*"[^"]*"', '', xml_content)
            
            root = ET.fromstring(xml_content)
            
            body = root.find('.//body')
            if body is None:
                body = root.find('.//{http://www.tei-c.org/ns/1.0}body')
            
            if body is not None:
                return self._get_element_text(body)
            
            text_elem = root.find('.//text')
            if text_elem is not None:
                return self._get_element_text(text_elem)
            
            return self._get_element_text(root)
            
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            text = re.sub(r'<[^>]+>', ' ', xml_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def _extract_metadata_from_xml(self, xml_content: str) -> Dict[str, Any]:
        """Extract metadata from TEI XML"""
        metadata = {}
        
        try:
            xml_content = re.sub(r'xmlns\s*=\s*"[^"]*"', '', xml_content)
            root = ET.fromstring(xml_content)
            
            title_elem = root.find('.//titleStmt/title')
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()
            
            author_elem = root.find('.//titleStmt/author')
            if author_elem is not None and author_elem.text:
                metadata["author"] = author_elem.text.strip()
            
            date_elem = root.find('.//publicationStmt/date')
            if date_elem is not None and date_elem.text:
                metadata["publication_date"] = date_elem.text.strip()
            
            source_elem = root.find('.//sourceDesc')
            if source_elem is not None:
                metadata["source_description"] = self._get_element_text(source_elem)
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_urn(self, file_path: str, xml_content: str) -> str:
        """Extract URN from file path or content"""
        urn_match = re.search(r'urn:cts:greekLit:[^\s<"]+', xml_content)
        if urn_match:
            return urn_match.group(0)
        
        parts = file_path.replace("data/", "").replace(".xml", "").split("/")
        if len(parts) >= 2:
            return f"urn:cts:greekLit:{parts[0]}.{parts[-1]}"
        
        return f"urn:cts:greekLit:{file_path}"
    
    def _extract_author_from_path(self, file_path: str) -> str:
        """Extract author from file path"""
        parts = file_path.split("/")
        for part in parts:
            if part.startswith("tlg"):
                return part
        return "Unknown"
    
    def _extract_title_from_path(self, file_path: str) -> str:
        """Extract title from file path"""
        filename = os.path.basename(file_path)
        title = filename.replace(".xml", "").replace("_", " ")
        return title
    
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
    
    def _load_from_cache(self, file_path: str) -> Optional[First1KText]:
        """Load text from cache"""
        if not self.config.cache_dir:
            return None
        
        cache_path = Path(self.config.cache_dir) / file_path.replace("/", "_")
        if cache_path.exists():
            try:
                import json
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return First1KText(**data)
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, file_path: str, text: First1KText):
        """Save text to cache"""
        if not self.config.cache_dir:
            return
        
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = cache_dir / file_path.replace("/", "_")
        
        try:
            import json
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(text.to_dict(), f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to cache text: {e}")
    
    def list_works(self, author_filter: Optional[str] = None) -> List[First1KWork]:
        """List available works"""
        if self._catalog_cache is not None:
            works = self._catalog_cache
        else:
            works = self._fetch_catalog()
            self._catalog_cache = works
        
        if author_filter:
            works = [w for w in works if author_filter.lower() in w.author.lower()]
        
        return works
    
    def _fetch_catalog(self) -> List[First1KWork]:
        """Fetch catalog from GitHub"""
        works = []
        
        url = f"{FIRST1K_API_URL}/git/trees/master?recursive=1"
        content = self._make_request(url)
        
        if not content:
            return works
        
        try:
            import json
            data = json.loads(content)
            
            for item in data.get("tree", []):
                path = item.get("path", "")
                
                if path.startswith("data/") and path.endswith(".xml"):
                    if "__cts__" in path or "metadata" in path.lower():
                        continue
                    
                    urn = self._extract_urn(path, "")
                    author = self._extract_author_from_path(path)
                    title = self._extract_title_from_path(path)
                    
                    works.append(First1KWork(
                        urn=urn,
                        title=title,
                        author=author,
                        file_path=path
                    ))
                    
        except Exception as e:
            logger.error(f"Error fetching catalog: {e}")
        
        return works
    
    def search(
        self,
        query: str,
        max_results: int = 50
    ) -> List[First1KWork]:
        """Search works"""
        works = self.list_works()
        
        results = []
        query_lower = query.lower()
        
        for work in works:
            if (query_lower in work.title.lower() or
                query_lower in work.author.lower() or
                query_lower in work.urn.lower()):
                results.append(work)
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def fetch_corpus(
        self,
        file_paths: List[str],
        corpus_name: str = "first1k_corpus"
    ) -> Corpus:
        """Fetch multiple texts as a corpus"""
        documents = []
        
        for path in file_paths:
            text = self.fetch_text(path)
            if text:
                documents.append(text.to_document())
        
        return Corpus(
            id=corpus_name,
            name=corpus_name,
            documents=documents,
            language=Language.ANCIENT_GREEK,
            metadata={"source": "first1kgreek", "file_paths": file_paths}
        )
    
    def download_corpus(
        self,
        output_dir: Union[str, Path],
        max_texts: Optional[int] = None
    ) -> int:
        """Download entire corpus to local directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        works = self.list_works()
        
        if max_texts:
            works = works[:max_texts]
        
        downloaded = 0
        
        for work in works:
            try:
                text = self.fetch_text(work.file_path)
                if text:
                    safe_filename = re.sub(r'[^\w\-.]', '_', work.file_path)
                    output_path = output_dir / safe_filename
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text.text_content)
                    
                    downloaded += 1
                    logger.info(f"Downloaded: {work.file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to download {work.file_path}: {e}")
        
        return downloaded


def fetch_first1k_text(
    file_path: str,
    config: Optional[First1KGreekConfig] = None
) -> Optional[First1KText]:
    """Fetch a text from First1KGreek"""
    client = First1KGreekClient(config)
    return client.fetch_text(file_path)


def list_first1k_works(
    author_filter: Optional[str] = None,
    config: Optional[First1KGreekConfig] = None
) -> List[First1KWork]:
    """List available works in First1KGreek"""
    client = First1KGreekClient(config)
    return client.list_works(author_filter)


def download_first1k_corpus(
    output_dir: Union[str, Path],
    max_texts: Optional[int] = None,
    config: Optional[First1KGreekConfig] = None
) -> int:
    """Download First1KGreek corpus"""
    client = First1KGreekClient(config)
    return client.download_corpus(output_dir, max_texts)
