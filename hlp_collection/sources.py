"""
Source Collectors - Text collection from various digital libraries

This module provides collectors for open access texts from:
- Perseus Digital Library
- First1KGreek
- Internet Archive
- Byzantine text archives
- Project Gutenberg

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import time
import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote

logger = logging.getLogger(__name__)


@dataclass
class CollectedText:
    source: str
    source_id: str
    title: str
    author: str
    language: str
    content: str
    url: str
    period: str = ""
    genre: str = ""
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'source_id': self.source_id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'url': self.url,
            'period': self.period,
            'genre': self.genre,
            'word_count': self.word_count,
            'metadata': self.metadata,
            'collected_at': self.collected_at.isoformat(),
        }


class BaseCollector(ABC):
    
    def __init__(self, rate_limit: float = 2.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.headers = {
            'User-Agent': 'NKUA-Historical-Linguistics-Platform/1.0 (Academic Research)'
        }
        self.stats = {
            'texts_collected': 0,
            'requests_made': 0,
            'errors': 0,
        }
    
    @abstractmethod
    def collect(self, **kwargs) -> Generator[CollectedText, None, None]:
        pass
    
    @abstractmethod
    def get_available_texts(self) -> List[Dict[str, Any]]:
        pass
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        self._rate_limit_wait()
        self.stats['requests_made'] += 1
        
        try:
            response = requests.get(url, headers=self.headers, timeout=60, **kwargs)
            if response.status_code == 200:
                return response
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            self.stats['errors'] += 1
            return None


class PerseusCollector(BaseCollector):
    
    BASE_URL = "https://www.perseus.tufts.edu"
    CATALOG_URL = "https://www.perseus.tufts.edu/hopper/collection"
    
    GREEK_COLLECTIONS = [
        "Perseus:collection:Greco-Roman",
        "Perseus:collection:Greek",
    ]
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.collected_urns = set()
    
    def collect(
        self,
        language: str = "greek",
        max_texts: int = 100,
        **kwargs
    ) -> Generator[CollectedText, None, None]:
        
        texts = self.get_available_texts(language=language)
        
        for i, text_info in enumerate(texts[:max_texts]):
            if text_info.get('urn') in self.collected_urns:
                continue
            
            try:
                text = self._fetch_text(text_info)
                if text and text.content:
                    self.collected_urns.add(text_info.get('urn'))
                    self.stats['texts_collected'] += 1
                    yield text
            except Exception as e:
                logger.error(f"Error collecting {text_info.get('title', 'unknown')}: {e}")
                self.stats['errors'] += 1
    
    def get_available_texts(self, language: str = "greek") -> List[Dict[str, Any]]:
        texts = []
        
        greek_texts = [
            {
                'urn': 'urn:cts:greekLit:tlg0012.tlg001',
                'title': 'Iliad',
                'author': 'Homer',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'epic',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0012.tlg002',
                'title': 'Odyssey',
                'author': 'Homer',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'epic',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0059.tlg030',
                'title': 'Republic',
                'author': 'Plato',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'philosophy',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0086.tlg034',
                'title': 'Nicomachean Ethics',
                'author': 'Aristotle',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'philosophy',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0016.tlg001',
                'title': 'Histories',
                'author': 'Herodotus',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'history',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0003.tlg001',
                'title': 'History of the Peloponnesian War',
                'author': 'Thucydides',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'history',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0085.tlg001',
                'title': 'Agamemnon',
                'author': 'Aeschylus',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'drama',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0011.tlg003',
                'title': 'Oedipus Tyrannus',
                'author': 'Sophocles',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'drama',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0006.tlg003',
                'title': 'Medea',
                'author': 'Euripides',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'drama',
            },
            {
                'urn': 'urn:cts:greekLit:tlg0527.tlg001',
                'title': 'New Testament',
                'author': 'Various',
                'language': 'grc',
                'period': 'Ancient',
                'genre': 'biblical',
            },
        ]
        
        if language.lower() in ['greek', 'grc', 'ancient_greek']:
            texts.extend(greek_texts)
        
        return texts
    
    def _fetch_text(self, text_info: Dict[str, Any]) -> Optional[CollectedText]:
        urn = text_info.get('urn', '')
        
        url = f"{self.BASE_URL}/hopper/text?doc={quote(urn)}"
        
        response = self._make_request(url)
        if not response:
            return None
        
        content = self._extract_text_from_html(response.text)
        
        if not content:
            return None
        
        return CollectedText(
            source='perseus',
            source_id=urn,
            title=text_info.get('title', ''),
            author=text_info.get('author', ''),
            language=text_info.get('language', 'grc'),
            content=content,
            url=url,
            period=text_info.get('period', ''),
            genre=text_info.get('genre', ''),
            metadata={'urn': urn},
        )
    
    def _extract_text_from_html(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            text_div = soup.find('div', {'class': 'text_container'})
            if text_div:
                for script in text_div.find_all(['script', 'style']):
                    script.decompose()
                return text_div.get_text(separator=' ', strip=True)
            
            main_content = soup.find('div', {'id': 'main_col'})
            if main_content:
                return main_content.get_text(separator=' ', strip=True)
            
            return ""
            
        except ImportError:
            logger.warning("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""


class First1KGreekCollector(BaseCollector):
    
    BASE_URL = "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data"
    CATALOG_URL = "https://api.github.com/repos/OpenGreekAndLatin/First1KGreek/contents/data"
    
    def __init__(self, rate_limit: float = 1.0):
        super().__init__(rate_limit)
        self.collected_files = set()
    
    def collect(
        self,
        max_texts: int = 100,
        **kwargs
    ) -> Generator[CollectedText, None, None]:
        
        texts = self.get_available_texts()
        
        for i, text_info in enumerate(texts[:max_texts]):
            file_path = text_info.get('path', '')
            if file_path in self.collected_files:
                continue
            
            try:
                text = self._fetch_text(text_info)
                if text and text.content:
                    self.collected_files.add(file_path)
                    self.stats['texts_collected'] += 1
                    yield text
            except Exception as e:
                logger.error(f"Error collecting {text_info.get('name', 'unknown')}: {e}")
                self.stats['errors'] += 1
    
    def get_available_texts(self) -> List[Dict[str, Any]]:
        texts = []
        
        response = self._make_request(self.CATALOG_URL)
        if not response:
            return self._get_fallback_texts()
        
        try:
            items = response.json()
            
            for item in items:
                if item.get('type') == 'dir':
                    author_dir = item.get('name', '')
                    
                    dir_url = item.get('url', '')
                    dir_response = self._make_request(dir_url)
                    
                    if dir_response:
                        dir_items = dir_response.json()
                        for file_item in dir_items:
                            if file_item.get('name', '').endswith('.xml'):
                                texts.append({
                                    'name': file_item.get('name', ''),
                                    'path': file_item.get('path', ''),
                                    'download_url': file_item.get('download_url', ''),
                                    'author': author_dir,
                                    'language': 'grc',
                                    'period': 'Ancient',
                                })
            
        except Exception as e:
            logger.error(f"Error parsing First1KGreek catalog: {e}")
            return self._get_fallback_texts()
        
        return texts
    
    def _get_fallback_texts(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'tlg0012.tlg001.1st1K-grc1.xml',
                'path': 'tlg0012/tlg001/tlg0012.tlg001.1st1K-grc1.xml',
                'download_url': f'{self.BASE_URL}/tlg0012/tlg001/tlg0012.tlg001.1st1K-grc1.xml',
                'author': 'Homer',
                'title': 'Iliad',
                'language': 'grc',
                'period': 'Ancient',
            },
        ]
    
    def _fetch_text(self, text_info: Dict[str, Any]) -> Optional[CollectedText]:
        url = text_info.get('download_url', '')
        if not url:
            return None
        
        response = self._make_request(url)
        if not response:
            return None
        
        content = self._extract_text_from_xml(response.text)
        
        if not content:
            return None
        
        title = text_info.get('title', text_info.get('name', '').replace('.xml', ''))
        
        return CollectedText(
            source='first1kgreek',
            source_id=text_info.get('path', ''),
            title=title,
            author=text_info.get('author', ''),
            language=text_info.get('language', 'grc'),
            content=content,
            url=url,
            period=text_info.get('period', 'Ancient'),
            genre=text_info.get('genre', ''),
        )
    
    def _extract_text_from_xml(self, xml_content: str) -> str:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(xml_content, 'xml')
            
            body = soup.find('body')
            if body:
                for note in body.find_all(['note', 'bibl', 'ref']):
                    note.decompose()
                return body.get_text(separator=' ', strip=True)
            
            text_elem = soup.find('text')
            if text_elem:
                return text_elem.get_text(separator=' ', strip=True)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text from XML: {e}")
            return ""


class InternetArchiveCollector(BaseCollector):
    
    BASE_URL = "https://archive.org"
    SEARCH_URL = "https://archive.org/advancedsearch.php"
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.collected_ids = set()
    
    def collect(
        self,
        query: str = "greek texts",
        language: str = "grc",
        max_texts: int = 50,
        **kwargs
    ) -> Generator[CollectedText, None, None]:
        
        texts = self.search_texts(query=query, max_results=max_texts)
        
        for text_info in texts:
            item_id = text_info.get('identifier', '')
            if item_id in self.collected_ids:
                continue
            
            try:
                text = self._fetch_text(text_info, language)
                if text and text.content:
                    self.collected_ids.add(item_id)
                    self.stats['texts_collected'] += 1
                    yield text
            except Exception as e:
                logger.error(f"Error collecting {text_info.get('title', 'unknown')}: {e}")
                self.stats['errors'] += 1
    
    def search_texts(
        self,
        query: str = "greek texts",
        mediatype: str = "texts",
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        
        params = {
            'q': f'{query} AND mediatype:{mediatype}',
            'fl[]': ['identifier', 'title', 'creator', 'date', 'language', 'subject'],
            'sort[]': 'downloads desc',
            'rows': max_results,
            'page': 1,
            'output': 'json',
        }
        
        response = self._make_request(self.SEARCH_URL, params=params)
        if not response:
            return []
        
        try:
            data = response.json()
            return data.get('response', {}).get('docs', [])
        except Exception as e:
            logger.error(f"Error parsing Internet Archive search results: {e}")
            return []
    
    def get_available_texts(self) -> List[Dict[str, Any]]:
        greek_queries = [
            "ancient greek texts",
            "byzantine greek manuscripts",
            "medieval greek",
            "greek patristics",
        ]
        
        all_texts = []
        for query in greek_queries:
            texts = self.search_texts(query=query, max_results=25)
            all_texts.extend(texts)
        
        return all_texts
    
    def _fetch_text(self, text_info: Dict[str, Any], language: str = "grc") -> Optional[CollectedText]:
        item_id = text_info.get('identifier', '')
        if not item_id:
            return None
        
        metadata_url = f"{self.BASE_URL}/metadata/{item_id}"
        response = self._make_request(metadata_url)
        
        if not response:
            return None
        
        try:
            metadata = response.json()
            files = metadata.get('files', [])
            
            txt_file = None
            for f in files:
                name = f.get('name', '')
                if name.endswith('.txt') and 'djvu' not in name.lower():
                    txt_file = f
                    break
            
            if not txt_file:
                for f in files:
                    if f.get('name', '').endswith('_djvu.txt'):
                        txt_file = f
                        break
            
            if not txt_file:
                return None
            
            text_url = f"{self.BASE_URL}/download/{item_id}/{txt_file['name']}"
            text_response = self._make_request(text_url)
            
            if not text_response:
                return None
            
            content = text_response.text
            
            return CollectedText(
                source='internet_archive',
                source_id=item_id,
                title=text_info.get('title', ''),
                author=text_info.get('creator', ''),
                language=language,
                content=content,
                url=f"{self.BASE_URL}/details/{item_id}",
                metadata={
                    'date': text_info.get('date', ''),
                    'subject': text_info.get('subject', []),
                },
            )
            
        except Exception as e:
            logger.error(f"Error fetching Internet Archive item {item_id}: {e}")
            return None


class ByzantineTextCollector(BaseCollector):
    
    SOURCES = {
        'dumbarton_oaks': {
            'name': 'Dumbarton Oaks Medieval Library',
            'base_url': 'https://www.doaks.org/resources/publications/doml',
        },
        'tlg': {
            'name': 'Thesaurus Linguae Graecae (Open Access)',
            'base_url': 'http://stephanus.tlg.uci.edu/',
        },
        'pinakes': {
            'name': 'Pinakes Database',
            'base_url': 'https://pinakes.irht.cnrs.fr/',
        },
    }
    
    BYZANTINE_TEXTS = [
        {
            'title': 'Chronicle',
            'author': 'John Malalas',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '6th',
            'genre': 'chronicle',
        },
        {
            'title': 'Chronographia',
            'author': 'Theophanes the Confessor',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '9th',
            'genre': 'chronicle',
        },
        {
            'title': 'History',
            'author': 'Michael Psellos',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '11th',
            'genre': 'history',
        },
        {
            'title': 'Alexiad',
            'author': 'Anna Komnene',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '12th',
            'genre': 'history',
        },
        {
            'title': 'Digenis Akritas',
            'author': 'Anonymous',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '12th',
            'genre': 'epic',
        },
        {
            'title': 'Chronicle of Morea',
            'author': 'Anonymous',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '14th',
            'genre': 'chronicle',
        },
        {
            'title': 'Homilies',
            'author': 'John Chrysostom',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '4th-5th',
            'genre': 'religious',
        },
        {
            'title': 'Ladder of Divine Ascent',
            'author': 'John Climacus',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '7th',
            'genre': 'religious',
        },
        {
            'title': 'Myriobiblon (Bibliotheca)',
            'author': 'Photius',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '9th',
            'genre': 'encyclopedia',
        },
        {
            'title': 'Suda',
            'author': 'Anonymous',
            'language': 'grc',
            'period': 'Byzantine',
            'century': '10th',
            'genre': 'encyclopedia',
        },
    ]
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.collected_texts = set()
    
    def collect(
        self,
        max_texts: int = 50,
        **kwargs
    ) -> Generator[CollectedText, None, None]:
        
        texts = self.get_available_texts()
        
        for text_info in texts[:max_texts]:
            text_key = f"{text_info.get('author', '')}_{text_info.get('title', '')}"
            if text_key in self.collected_texts:
                continue
            
            try:
                text = self._search_and_fetch(text_info)
                if text and text.content:
                    self.collected_texts.add(text_key)
                    self.stats['texts_collected'] += 1
                    yield text
            except Exception as e:
                logger.error(f"Error collecting {text_info.get('title', 'unknown')}: {e}")
                self.stats['errors'] += 1
    
    def get_available_texts(self) -> List[Dict[str, Any]]:
        return self.BYZANTINE_TEXTS.copy()
    
    def _search_and_fetch(self, text_info: Dict[str, Any]) -> Optional[CollectedText]:
        author = text_info.get('author', '')
        title = text_info.get('title', '')
        
        ia_collector = InternetArchiveCollector(rate_limit=self.rate_limit)
        query = f"{author} {title} greek"
        
        results = ia_collector.search_texts(query=query, max_results=5)
        
        for result in results:
            text = ia_collector._fetch_text(result, language='grc')
            if text and text.content and len(text.content) > 1000:
                text.period = text_info.get('period', 'Byzantine')
                text.genre = text_info.get('genre', '')
                text.metadata['century'] = text_info.get('century', '')
                text.metadata['original_search'] = text_info
                return text
        
        return CollectedText(
            source='byzantine_catalog',
            source_id=f"{author}_{title}".replace(' ', '_'),
            title=title,
            author=author,
            language='grc',
            content=f"[Catalog entry - text not yet digitized]\n\nTitle: {title}\nAuthor: {author}\nPeriod: {text_info.get('period', 'Byzantine')}\nCentury: {text_info.get('century', '')}\nGenre: {text_info.get('genre', '')}",
            url='',
            period=text_info.get('period', 'Byzantine'),
            genre=text_info.get('genre', ''),
            metadata={
                'catalog_only': True,
                'century': text_info.get('century', ''),
            },
        )


class GutenbergCollector(BaseCollector):
    
    BASE_URL = "https://www.gutenberg.org"
    CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.zip"
    
    GREEK_TEXTS = [
        {'id': 8300, 'title': 'New Testament (Greek)', 'author': 'Various', 'language': 'grc'},
        {'id': 6130, 'title': 'Iliad (Greek)', 'author': 'Homer', 'language': 'grc'},
        {'id': 1999, 'title': 'Odyssey (Greek)', 'author': 'Homer', 'language': 'grc'},
        {'id': 35899, 'title': 'Republic (Greek)', 'author': 'Plato', 'language': 'grc'},
        {'id': 7700, 'title': 'Poetics (Greek)', 'author': 'Aristotle', 'language': 'grc'},
        {'id': 5827, 'title': 'Histories (Greek)', 'author': 'Herodotus', 'language': 'grc'},
        {'id': 7142, 'title': 'Peloponnesian War (Greek)', 'author': 'Thucydides', 'language': 'grc'},
        {'id': 35688, 'title': 'Medea (Greek)', 'author': 'Euripides', 'language': 'grc'},
        {'id': 2707, 'title': 'Meditations (Greek)', 'author': 'Marcus Aurelius', 'language': 'grc'},
        {'id': 45109, 'title': 'Septuagint Genesis (Greek)', 'author': 'Various', 'language': 'grc'},
    ]
    
    LATIN_TEXTS = [
        {'id': 232, 'title': 'Aeneid', 'author': 'Virgil', 'language': 'lat'},
        {'id': 2772, 'title': 'Metamorphoses', 'author': 'Ovid', 'language': 'lat'},
        {'id': 5711, 'title': 'De Bello Gallico', 'author': 'Julius Caesar', 'language': 'lat'},
        {'id': 10048, 'title': 'Confessions', 'author': 'Augustine', 'language': 'lat'},
    ]
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.collected_ids = set()
    
    def collect(
        self,
        language: str = "grc",
        max_texts: int = 50,
        **kwargs
    ) -> Generator[CollectedText, None, None]:
        
        if language.lower() in ['grc', 'greek', 'ancient_greek']:
            texts = self.GREEK_TEXTS
        elif language.lower() in ['lat', 'latin']:
            texts = self.LATIN_TEXTS
        else:
            texts = self.GREEK_TEXTS + self.LATIN_TEXTS
        
        for text_info in texts[:max_texts]:
            book_id = text_info.get('id')
            if book_id in self.collected_ids:
                continue
            
            try:
                text = self._fetch_text(text_info)
                if text and text.content:
                    self.collected_ids.add(book_id)
                    self.stats['texts_collected'] += 1
                    yield text
            except Exception as e:
                logger.error(f"Error collecting Gutenberg #{book_id}: {e}")
                self.stats['errors'] += 1
    
    def get_available_texts(self) -> List[Dict[str, Any]]:
        return self.GREEK_TEXTS + self.LATIN_TEXTS
    
    def _fetch_text(self, text_info: Dict[str, Any]) -> Optional[CollectedText]:
        book_id = text_info.get('id')
        
        urls = [
            f"{self.BASE_URL}/cache/epub/{book_id}/pg{book_id}.txt",
            f"{self.BASE_URL}/files/{book_id}/{book_id}-0.txt",
            f"{self.BASE_URL}/files/{book_id}/{book_id}.txt",
        ]
        
        content = None
        used_url = None
        
        for url in urls:
            response = self._make_request(url)
            if response:
                content = response.text
                used_url = url
                break
        
        if not content:
            return None
        
        content = self._clean_gutenberg_text(content)
        
        return CollectedText(
            source='gutenberg',
            source_id=str(book_id),
            title=text_info.get('title', ''),
            author=text_info.get('author', ''),
            language=text_info.get('language', 'en'),
            content=content,
            url=used_url or f"{self.BASE_URL}/ebooks/{book_id}",
            period=text_info.get('period', 'Ancient'),
            genre=text_info.get('genre', ''),
        )
    
    def _clean_gutenberg_text(self, text: str) -> str:
        start_markers = [
            '*** START OF THIS PROJECT GUTENBERG',
            '*** START OF THE PROJECT GUTENBERG',
            '*END*THE SMALL PRINT',
        ]
        
        end_markers = [
            '*** END OF THIS PROJECT GUTENBERG',
            '*** END OF THE PROJECT GUTENBERG',
            'End of Project Gutenberg',
            'End of the Project Gutenberg',
        ]
        
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        
        for marker in end_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        
        return text.strip()
