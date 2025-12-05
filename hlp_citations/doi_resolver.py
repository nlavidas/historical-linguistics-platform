"""
DOI Resolver - DOI lookup and validation for citations

This module provides DOI resolution and validation for
academic citations in the Historical Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import re
import json
import logging
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

from hlp_citations.citation_manager import Citation, Author

logger = logging.getLogger(__name__)


class DOIValidator:
    DOI_PATTERN = re.compile(r'^10\.\d{4,}/[^\s]+$')
    DOI_URL_PATTERN = re.compile(r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s]+)')
    
    @staticmethod
    def is_valid_format(doi: str) -> bool:
        if DOIValidator.DOI_PATTERN.match(doi):
            return True
        match = DOIValidator.DOI_URL_PATTERN.match(doi)
        return match is not None
    
    @staticmethod
    def extract_doi(text: str) -> Optional[str]:
        match = DOIValidator.DOI_URL_PATTERN.search(text)
        if match:
            return match.group(1)
        match = DOIValidator.DOI_PATTERN.search(text)
        if match:
            return match.group(0)
        return None
    
    @staticmethod
    def normalize_doi(doi: str) -> str:
        match = DOIValidator.DOI_URL_PATTERN.match(doi)
        if match:
            return match.group(1)
        return doi.strip()
    
    @staticmethod
    def to_url(doi: str) -> str:
        normalized = DOIValidator.normalize_doi(doi)
        return f"https://doi.org/{normalized}"


@dataclass
class DOIMetadata:
    doi: str
    title: str
    authors: List[Dict[str, str]]
    container_title: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    page: Optional[str] = None
    published_date: Optional[str] = None
    publisher: Optional[str] = None
    type: Optional[str] = None
    issn: Optional[List[str]] = None
    isbn: Optional[List[str]] = None
    abstract: Optional[str] = None
    subject: Optional[List[str]] = None
    url: Optional[str] = None
    license: Optional[str] = None
    raw_data: Optional[Dict] = None


class DOIResolver:
    CROSSREF_API = "https://api.crossref.org/works/"
    DATACITE_API = "https://api.datacite.org/dois/"
    
    def __init__(self, email: Optional[str] = None):
        self.email = email or "hlp@nkua.gr"
        self.cache: Dict[str, DOIMetadata] = {}
        self.headers = {
            "User-Agent": f"HLP-Platform/1.0 (mailto:{self.email})",
            "Accept": "application/json"
        }
    
    def resolve(self, doi: str) -> Optional[DOIMetadata]:
        normalized_doi = DOIValidator.normalize_doi(doi)
        
        if normalized_doi in self.cache:
            return self.cache[normalized_doi]
        
        metadata = self._resolve_crossref(normalized_doi)
        
        if not metadata:
            metadata = self._resolve_datacite(normalized_doi)
        
        if metadata:
            self.cache[normalized_doi] = metadata
        
        return metadata
    
    def _resolve_crossref(self, doi: str) -> Optional[DOIMetadata]:
        try:
            url = f"{self.CROSSREF_API}{doi}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            message = data.get('message', {})
            
            authors = []
            for author in message.get('author', []):
                authors.append({
                    'family': author.get('family', ''),
                    'given': author.get('given', ''),
                    'orcid': author.get('ORCID', ''),
                    'affiliation': ', '.join([a.get('name', '') for a in author.get('affiliation', [])])
                })
            
            published = message.get('published-print', message.get('published-online', {}))
            date_parts = published.get('date-parts', [[]])[0]
            if date_parts:
                if len(date_parts) >= 3:
                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 1:
                    published_date = str(date_parts[0])
                else:
                    published_date = None
            else:
                published_date = None
            
            title_list = message.get('title', [])
            title = title_list[0] if title_list else ''
            
            container_list = message.get('container-title', [])
            container_title = container_list[0] if container_list else None
            
            return DOIMetadata(
                doi=doi,
                title=title,
                authors=authors,
                container_title=container_title,
                volume=message.get('volume'),
                issue=message.get('issue'),
                page=message.get('page'),
                published_date=published_date,
                publisher=message.get('publisher'),
                type=message.get('type'),
                issn=message.get('ISSN'),
                isbn=message.get('ISBN'),
                abstract=message.get('abstract'),
                subject=message.get('subject'),
                url=message.get('URL'),
                license=message.get('license', [{}])[0].get('URL') if message.get('license') else None,
                raw_data=message
            )
            
        except Exception as e:
            logger.error(f"Error resolving DOI {doi} via CrossRef: {e}")
            return None
    
    def _resolve_datacite(self, doi: str) -> Optional[DOIMetadata]:
        try:
            url = f"{self.DATACITE_API}{doi}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            attributes = data.get('data', {}).get('attributes', {})
            
            authors = []
            for creator in attributes.get('creators', []):
                name_parts = creator.get('name', '').split(', ')
                if len(name_parts) >= 2:
                    authors.append({
                        'family': name_parts[0],
                        'given': name_parts[1],
                        'orcid': creator.get('nameIdentifiers', [{}])[0].get('nameIdentifier', '') if creator.get('nameIdentifiers') else '',
                        'affiliation': ', '.join([a.get('name', '') for a in creator.get('affiliation', [])])
                    })
                else:
                    authors.append({
                        'family': creator.get('name', ''),
                        'given': '',
                        'orcid': '',
                        'affiliation': ''
                    })
            
            titles = attributes.get('titles', [])
            title = titles[0].get('title', '') if titles else ''
            
            return DOIMetadata(
                doi=doi,
                title=title,
                authors=authors,
                container_title=attributes.get('container', {}).get('title'),
                published_date=str(attributes.get('publicationYear', '')),
                publisher=attributes.get('publisher'),
                type=attributes.get('types', {}).get('resourceTypeGeneral'),
                url=attributes.get('url'),
                raw_data=attributes
            )
            
        except Exception as e:
            logger.error(f"Error resolving DOI {doi} via DataCite: {e}")
            return None
    
    def to_citation(self, doi: str) -> Optional[Citation]:
        metadata = self.resolve(doi)
        if not metadata:
            return None
        
        authors = []
        for author_data in metadata.authors:
            authors.append(Author(
                family=author_data.get('family', ''),
                given=author_data.get('given', ''),
                orcid=author_data.get('orcid'),
                affiliation=author_data.get('affiliation')
            ))
        
        year = None
        if metadata.published_date:
            try:
                year = int(metadata.published_date[:4])
            except (ValueError, IndexError):
                pass
        
        citation_type = self._map_type(metadata.type)
        
        return Citation(
            citation_type=citation_type,
            title=metadata.title,
            authors=authors,
            year=year,
            journal=metadata.container_title,
            volume=metadata.volume,
            issue=metadata.issue,
            pages=metadata.page,
            publisher=metadata.publisher,
            doi=metadata.doi,
            url=metadata.url,
            abstract=metadata.abstract,
            keywords=metadata.subject or [],
            verified=True,
            verification_date=datetime.now().isoformat(),
            verification_source="doi_resolver"
        )
    
    def _map_type(self, doi_type: Optional[str]) -> str:
        if not doi_type:
            return 'article'
        
        type_map = {
            'journal-article': 'article',
            'book': 'book',
            'book-chapter': 'chapter',
            'proceedings-article': 'inproceedings',
            'dissertation': 'phdthesis',
            'dataset': 'dataset',
            'software': 'software',
            'report': 'techreport',
            'monograph': 'book',
            'edited-book': 'book',
            'reference-book': 'book',
            'posted-content': 'unpublished',
        }
        
        return type_map.get(doi_type.lower(), 'article')
    
    def verify_doi_exists(self, doi: str) -> bool:
        normalized_doi = DOIValidator.normalize_doi(doi)
        try:
            url = f"https://doi.org/{normalized_doi}"
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def batch_resolve(self, dois: List[str]) -> Dict[str, Optional[DOIMetadata]]:
        results = {}
        for doi in dois:
            results[doi] = self.resolve(doi)
        return results
    
    def search_crossref(
        self,
        query: str,
        rows: int = 10,
        filter_type: Optional[str] = None
    ) -> List[DOIMetadata]:
        try:
            params = {
                'query': query,
                'rows': rows,
            }
            
            if filter_type:
                params['filter'] = f'type:{filter_type}'
            
            url = "https://api.crossref.org/works"
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            items = data.get('message', {}).get('items', [])
            
            results = []
            for item in items:
                doi = item.get('DOI')
                if doi:
                    metadata = self._parse_crossref_item(item)
                    if metadata:
                        results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {e}")
            return []
    
    def _parse_crossref_item(self, item: Dict) -> Optional[DOIMetadata]:
        try:
            authors = []
            for author in item.get('author', []):
                authors.append({
                    'family': author.get('family', ''),
                    'given': author.get('given', ''),
                    'orcid': author.get('ORCID', ''),
                })
            
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ''
            
            container_list = item.get('container-title', [])
            container_title = container_list[0] if container_list else None
            
            published = item.get('published-print', item.get('published-online', {}))
            date_parts = published.get('date-parts', [[]])[0]
            published_date = str(date_parts[0]) if date_parts else None
            
            return DOIMetadata(
                doi=item.get('DOI', ''),
                title=title,
                authors=authors,
                container_title=container_title,
                volume=item.get('volume'),
                issue=item.get('issue'),
                page=item.get('page'),
                published_date=published_date,
                publisher=item.get('publisher'),
                type=item.get('type'),
            )
        except Exception:
            return None


class CitationVerifier:
    
    def __init__(self):
        self.doi_resolver = DOIResolver()
    
    def verify_citation(self, citation: Citation) -> Dict[str, Any]:
        result = {
            'verified': False,
            'doi_valid': False,
            'doi_resolves': False,
            'metadata_matches': False,
            'issues': [],
            'suggestions': []
        }
        
        if citation.doi:
            if DOIValidator.is_valid_format(citation.doi):
                result['doi_valid'] = True
                
                resolved = self.doi_resolver.resolve(citation.doi)
                if resolved:
                    result['doi_resolves'] = True
                    
                    matches = self._compare_metadata(citation, resolved)
                    result['metadata_matches'] = matches['overall']
                    result['issues'] = matches['issues']
                    result['suggestions'] = matches['suggestions']
                    
                    if matches['overall']:
                        result['verified'] = True
                else:
                    result['issues'].append("DOI does not resolve")
            else:
                result['issues'].append("DOI format is invalid")
        else:
            result['issues'].append("No DOI provided")
            result['suggestions'].append("Consider adding a DOI for verification")
        
        return result
    
    def _compare_metadata(self, citation: Citation, resolved: DOIMetadata) -> Dict[str, Any]:
        issues = []
        suggestions = []
        matches = True
        
        if citation.title and resolved.title:
            title_similarity = self._string_similarity(
                citation.title.lower(),
                resolved.title.lower()
            )
            if title_similarity < 0.8:
                issues.append(f"Title mismatch: '{citation.title}' vs '{resolved.title}'")
                suggestions.append(f"Consider updating title to: {resolved.title}")
                matches = False
        
        if citation.authors and resolved.authors:
            citation_families = {a.family.lower() for a in citation.authors}
            resolved_families = {a.get('family', '').lower() for a in resolved.authors}
            
            if not citation_families.intersection(resolved_families):
                issues.append("No matching author family names found")
                matches = False
        
        if citation.year and resolved.published_date:
            try:
                resolved_year = int(resolved.published_date[:4])
                if citation.year != resolved_year:
                    issues.append(f"Year mismatch: {citation.year} vs {resolved_year}")
                    suggestions.append(f"Consider updating year to: {resolved_year}")
            except (ValueError, IndexError):
                pass
        
        return {
            'overall': matches and len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def batch_verify(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        return [self.verify_citation(c) for c in citations]
