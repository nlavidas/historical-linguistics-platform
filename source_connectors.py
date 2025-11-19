"""
Source Connectors for External Corpus Sources
Production-grade connectors for Perseus, GitHub, Archive.org, etc.
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerseusConnector:
    """Connector for Perseus Digital Library"""
    
    BASE_URL = "https://www.perseus.tufts.edu"
    
    async def get_catalog(self) -> List[Dict]:
        """Fetch Perseus catalog of texts"""
        async with aiohttp.ClientSession() as session:
            # Perseus catalog URLs
            catalog_urls = [
                f"{self.BASE_URL}/hopper/collection?collection=Perseus:collection:Greco-Roman",
                f"{self.BASE_URL}/hopper/collection?collection=Perseus:collection:cwar",
            ]
            
            texts = []
            for url in catalog_urls:
                try:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            # Parse catalog (simplified)
                            texts.append({
                                'source': 'perseus',
                                'url': url,
                                'language': 'grc',
                                'collection': 'greco-roman'
                            })
                except Exception as e:
                    logger.error(f"Error fetching Perseus catalog: {e}")
            
            return texts
    
    def get_sample_texts(self) -> List[str]:
        """Get sample Perseus text URLs"""
        return [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133",  # Homer Iliad
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0134",  # Homer Odyssey
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0135",  # Hesiod
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0136",  # Herodotus
        ]


class GitHubConnector:
    """Connector for GitHub repositories"""
    
    async def get_canonical_greek(self) -> List[Dict]:
        """Fetch First1K Greek texts from GitHub"""
        base = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data"
        
        # Major authors and works
        texts = []
        authors = {
            'tlg0012': 'Homer',  # tlg001 = Iliad, tlg002 = Odyssey
            'tlg0016': 'Herodotus',
            'tlg0003': 'Thucydides',
            'tlg0059': 'Plato',
            'tlg0086': 'Aristotle',
        }
        
        for author_id, author_name in authors.items():
            # Typical structure: data/tlg0012/tlg001/file.xml
            for work_num in range(1, 5):  # Check first few works
                work_id = f"tlg{work_num:03d}"
                url = f"{base}/{author_id}/{work_id}/{author_id}.{work_id}.perseus-grc2.xml"
                texts.append({
                    'source': 'github',
                    'url': url,
                    'author': author_name,
                    'language': 'grc',
                    'format': 'xml'
                })
        
        return texts


class GutenbergConnector:
    """Connector for Project Gutenberg"""
    
    BASE_URL = "https://www.gutenberg.org"
    
    def get_sample_texts(self) -> List[str]:
        """Get sample Gutenberg text URLs"""
        return [
            "https://www.gutenberg.org/cache/epub/1/pg1.txt",  # Jefferson: Declaration of Independence
            "https://www.gutenberg.org/cache/epub/2/pg2.txt",  # US Bill of Rights
            "https://www.gutenberg.org/cache/epub/11/pg11.txt",  # Alice in Wonderland
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        ]


class ArchiveOrgConnector:
    """Connector for Archive.org"""
    
    BASE_URL = "https://archive.org"
    
    async def search_texts(self, query: str = "ancient greek", limit: int = 10) -> List[Dict]:
        """Search Archive.org for texts"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.BASE_URL}/advancedsearch.php"
            params = {
                'q': query,
                'fl[]': ['identifier', 'title'],
                'rows': limit,
                'page': 1,
                'output': 'json',
                'mediatype': 'texts'
            }
            
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('response', {}).get('docs', []):
                            identifier = item.get('identifier')
                            if identifier:
                                results.append({
                                    'source': 'archive_org',
                                    'url': f"{self.BASE_URL}/download/{identifier}",
                                    'title': item.get('title', 'Unknown'),
                                    'identifier': identifier
                                })
                        return results
            except Exception as e:
                logger.error(f"Error searching Archive.org: {e}")
        
        return []


class SourceManager:
    """Manages all external source connectors"""
    
    def __init__(self):
        self.perseus = PerseusConnector()
        self.github = GitHubConnector()
        self.gutenberg = GutenbergConnector()
        self.archive = ArchiveOrgConnector()
    
    async def discover_sources(self) -> Dict[str, List]:
        """Discover available texts from all sources"""
        logger.info("Discovering texts from external sources...")
        
        sources = {
            'perseus': self.perseus.get_sample_texts(),
            'github': await self.github.get_canonical_greek(),
            'gutenberg': self.gutenberg.get_sample_texts(),
            'archive_org': await self.archive.search_texts()
        }
        
        total = sum(len(v) for v in sources.values())
        logger.info(f"Discovered {total} texts across {len(sources)} sources")
        
        return sources
    
    def get_quick_start_collection(self) -> List[Dict]:
        """Get a curated collection for quick start"""
        return [
            {
                'url': 'https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml',
                'source': 'github',
                'title': 'Homer - Iliad',
                'language': 'grc',
                'priority': 10
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
                'source': 'gutenberg',
                'title': 'Pride and Prejudice',
                'language': 'en',
                'priority': 8
            },
            {
                'url': 'https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0016/tlg001/tlg0016.tlg001.perseus-grc2.xml',
                'source': 'github',
                'title': 'Herodotus - Histories',
                'language': 'grc',
                'priority': 9
            }
        ]


async def test_connectors():
    """Test all connectors"""
    print("Testing Source Connectors")
    print("=" * 70)
    
    manager = SourceManager()
    
    print("\nDiscovering sources...")
    sources = await manager.discover_sources()
    
    for source_name, texts in sources.items():
        print(f"\n{source_name.upper()}:")
        print(f"  Found {len(texts)} texts")
        if texts:
            print(f"  Sample: {texts[0]}")
    
    print("\nQuick Start Collection:")
    collection = manager.get_quick_start_collection()
    for item in collection:
        print(f"  - {item['title']} ({item['language']}) from {item['source']}")


if __name__ == "__main__":
    asyncio.run(test_connectors())
