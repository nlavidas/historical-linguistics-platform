"""
Text Search Engine - Full-text search for historical linguistics texts

This module provides comprehensive search capabilities including:
- Full-text search with ranking
- Filters by language, period, genre, author
- Fuzzy matching for historical spelling variants
- Lemma-based search for morphologically rich languages
- N-gram search for partial matches

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
from collections import Counter
import math

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    LEMMA = "lemma"
    REGEX = "regex"
    NGRAM = "ngram"


@dataclass
class SearchFilter:
    languages: List[str] = field(default_factory=list)
    periods: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None
    is_biblical: Optional[bool] = None
    is_classical: Optional[bool] = None
    is_translation: Optional[bool] = None
    has_treebank: Optional[bool] = None
    
    def to_sql_conditions(self) -> Tuple[str, List[Any]]:
        conditions = []
        params = []
        
        if self.languages:
            placeholders = ','.join(['?' for _ in self.languages])
            conditions.append(f"language IN ({placeholders})")
            params.extend(self.languages)
        
        if self.periods:
            placeholders = ','.join(['?' for _ in self.periods])
            conditions.append(f"period IN ({placeholders})")
            params.extend(self.periods)
        
        if self.genres:
            placeholders = ','.join(['?' for _ in self.genres])
            conditions.append(f"genre IN ({placeholders})")
            params.extend(self.genres)
        
        if self.authors:
            author_conditions = []
            for author in self.authors:
                author_conditions.append("author LIKE ?")
                params.append(f"%{author}%")
            conditions.append(f"({' OR '.join(author_conditions)})")
        
        if self.sources:
            placeholders = ','.join(['?' for _ in self.sources])
            conditions.append(f"source IN ({placeholders})")
            params.extend(self.sources)
        
        if self.min_word_count is not None:
            conditions.append("word_count >= ?")
            params.append(self.min_word_count)
        
        if self.max_word_count is not None:
            conditions.append("word_count <= ?")
            params.append(self.max_word_count)
        
        if self.is_biblical is not None:
            conditions.append("is_biblical = ?")
            params.append(1 if self.is_biblical else 0)
        
        if self.is_classical is not None:
            conditions.append("is_classical = ?")
            params.append(1 if self.is_classical else 0)
        
        if self.is_translation is not None:
            conditions.append("is_retranslation = ?")
            params.append(1 if self.is_translation else 0)
        
        if self.has_treebank is not None:
            conditions.append("has_treebank = ?")
            params.append(1 if self.has_treebank else 0)
        
        return ' AND '.join(conditions) if conditions else '1=1', params


@dataclass
class SearchQuery:
    query: str
    mode: SearchMode = SearchMode.EXACT
    filters: Optional[SearchFilter] = None
    limit: int = 50
    offset: int = 0
    highlight: bool = True
    context_chars: int = 100
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = SearchFilter()


@dataclass
class SearchMatch:
    position: int
    length: int
    context_before: str
    matched_text: str
    context_after: str
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'length': self.length,
            'context_before': self.context_before,
            'matched_text': self.matched_text,
            'context_after': self.context_after,
            'line_number': self.line_number,
        }


@dataclass
class SearchResult:
    text_id: int
    title: str
    author: str
    language: str
    period: str
    genre: str
    source: str
    word_count: int
    score: float
    matches: List[SearchMatch] = field(default_factory=list)
    match_count: int = 0
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text_id': self.text_id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'period': self.period,
            'genre': self.genre,
            'source': self.source,
            'word_count': self.word_count,
            'score': self.score,
            'matches': [m.to_dict() for m in self.matches],
            'match_count': self.match_count,
            'snippet': self.snippet,
            'metadata': self.metadata,
        }


class TextSearchEngine:
    
    def __init__(self, db_path: str = "data/corpus_platform.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_fts_table()
        
        self.greek_lemmatizer = None
        self.latin_lemmatizer = None
        self._init_lemmatizers()
    
    def _init_lemmatizers(self):
        try:
            from hlp_collection.arcas_tools import GreekLemmatizer, LatinLemmatizer
            self.greek_lemmatizer = GreekLemmatizer()
            self.latin_lemmatizer = LatinLemmatizer()
            logger.info("Lemmatizers initialized for search")
        except ImportError:
            logger.warning("Lemmatizers not available - lemma search disabled")
    
    def _ensure_fts_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
                        title,
                        author,
                        content,
                        content='corpus_items',
                        content_rowid='id'
                    )
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS corpus_fts_insert AFTER INSERT ON corpus_items BEGIN
                        INSERT INTO corpus_fts(rowid, title, author, content)
                        VALUES (new.id, new.title, new.author, new.content);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS corpus_fts_delete AFTER DELETE ON corpus_items BEGIN
                        INSERT INTO corpus_fts(corpus_fts, rowid, title, author, content)
                        VALUES ('delete', old.id, old.title, old.author, old.content);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS corpus_fts_update AFTER UPDATE ON corpus_items BEGIN
                        INSERT INTO corpus_fts(corpus_fts, rowid, title, author, content)
                        VALUES ('delete', old.id, old.title, old.author, old.content);
                        INSERT INTO corpus_fts(rowid, title, author, content)
                        VALUES (new.id, new.title, new.author, new.content);
                    END
                """)
                
                conn.commit()
                logger.info("FTS5 table and triggers created")
        except Exception as e:
            logger.warning(f"Could not create FTS table (may already exist or SQLite version issue): {e}")
    
    def rebuild_fts_index(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM corpus_fts")
                conn.execute("""
                    INSERT INTO corpus_fts(rowid, title, author, content)
                    SELECT id, title, author, content FROM corpus_items
                """)
                conn.commit()
                logger.info("FTS index rebuilt")
        except Exception as e:
            logger.error(f"Error rebuilding FTS index: {e}")
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        if query.mode == SearchMode.EXACT:
            return self._search_exact(query)
        elif query.mode == SearchMode.FUZZY:
            return self._search_fuzzy(query)
        elif query.mode == SearchMode.LEMMA:
            return self._search_lemma(query)
        elif query.mode == SearchMode.REGEX:
            return self._search_regex(query)
        elif query.mode == SearchMode.NGRAM:
            return self._search_ngram(query)
        else:
            return self._search_exact(query)
    
    def _search_exact(self, query: SearchQuery) -> List[SearchResult]:
        results = []
        
        try:
            filter_sql, filter_params = query.filters.to_sql_conditions()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                try:
                    fts_query = f"""
                        SELECT c.*, bm25(corpus_fts) as score
                        FROM corpus_fts f
                        JOIN corpus_items c ON f.rowid = c.id
                        WHERE corpus_fts MATCH ? AND {filter_sql}
                        ORDER BY score
                        LIMIT ? OFFSET ?
                    """
                    cursor = conn.execute(
                        fts_query,
                        [query.query] + filter_params + [query.limit, query.offset]
                    )
                except:
                    like_query = f"""
                        SELECT *, 1.0 as score
                        FROM corpus_items
                        WHERE (content LIKE ? OR title LIKE ? OR author LIKE ?) AND {filter_sql}
                        ORDER BY word_count DESC
                        LIMIT ? OFFSET ?
                    """
                    search_pattern = f"%{query.query}%"
                    cursor = conn.execute(
                        like_query,
                        [search_pattern, search_pattern, search_pattern] + filter_params + [query.limit, query.offset]
                    )
                
                for row in cursor:
                    result = self._row_to_result(row, query)
                    if result.match_count > 0 or not query.highlight:
                        results.append(result)
        
        except Exception as e:
            logger.error(f"Search error: {e}")
        
        return results
    
    def _search_fuzzy(self, query: SearchQuery) -> List[SearchResult]:
        results = []
        
        variants = self._generate_fuzzy_variants(query.query)
        
        try:
            filter_sql, filter_params = query.filters.to_sql_conditions()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                like_conditions = ' OR '.join(['content LIKE ?' for _ in variants])
                sql = f"""
                    SELECT *, 1.0 as score
                    FROM corpus_items
                    WHERE ({like_conditions}) AND {filter_sql}
                    ORDER BY word_count DESC
                    LIMIT ? OFFSET ?
                """
                
                variant_patterns = [f"%{v}%" for v in variants]
                cursor = conn.execute(
                    sql,
                    variant_patterns + filter_params + [query.limit, query.offset]
                )
                
                for row in cursor:
                    result = self._row_to_result(row, query, variants)
                    if result.match_count > 0:
                        results.append(result)
        
        except Exception as e:
            logger.error(f"Fuzzy search error: {e}")
        
        return results
    
    def _generate_fuzzy_variants(self, query: str) -> List[str]:
        variants = [query]
        
        greek_variants = {
            'ει': ['ι', 'η'],
            'οι': ['υ', 'ι'],
            'αι': ['ε'],
            'ου': ['υ'],
            'ω': ['ο'],
            'η': ['ι', 'ε'],
            'υ': ['ι', 'οι'],
        }
        
        for original, replacements in greek_variants.items():
            if original in query:
                for replacement in replacements:
                    variants.append(query.replace(original, replacement))
        
        return list(set(variants))[:10]
    
    def _search_lemma(self, query: SearchQuery) -> List[SearchResult]:
        if not self.greek_lemmatizer and not self.latin_lemmatizer:
            logger.warning("Lemmatizers not available, falling back to exact search")
            return self._search_exact(query)
        
        lemma = query.query
        
        if self.greek_lemmatizer:
            lemma = self.greek_lemmatizer.lemmatize(query.query)
        
        lemma_query = SearchQuery(
            query=lemma,
            mode=SearchMode.FUZZY,
            filters=query.filters,
            limit=query.limit,
            offset=query.offset,
            highlight=query.highlight,
            context_chars=query.context_chars,
        )
        
        return self._search_fuzzy(lemma_query)
    
    def _search_regex(self, query: SearchQuery) -> List[SearchResult]:
        results = []
        
        try:
            pattern = re.compile(query.query, re.IGNORECASE | re.UNICODE)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return results
        
        try:
            filter_sql, filter_params = query.filters.to_sql_conditions()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = f"""
                    SELECT *, 1.0 as score
                    FROM corpus_items
                    WHERE {filter_sql}
                    ORDER BY word_count DESC
                """
                
                cursor = conn.execute(sql, filter_params)
                
                count = 0
                for row in cursor:
                    if count >= query.offset + query.limit:
                        break
                    
                    content = row['content'] or ''
                    if pattern.search(content):
                        if count >= query.offset:
                            result = self._row_to_result_regex(row, query, pattern)
                            results.append(result)
                        count += 1
        
        except Exception as e:
            logger.error(f"Regex search error: {e}")
        
        return results
    
    def _search_ngram(self, query: SearchQuery) -> List[SearchResult]:
        ngrams = self._generate_ngrams(query.query, n=3)
        
        ngram_query = SearchQuery(
            query=' OR '.join(ngrams[:5]),
            mode=SearchMode.EXACT,
            filters=query.filters,
            limit=query.limit,
            offset=query.offset,
            highlight=query.highlight,
            context_chars=query.context_chars,
        )
        
        return self._search_exact(ngram_query)
    
    def _generate_ngrams(self, text: str, n: int = 3) -> List[str]:
        text = text.lower()
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams
    
    def _row_to_result(
        self,
        row: sqlite3.Row,
        query: SearchQuery,
        search_terms: Optional[List[str]] = None
    ) -> SearchResult:
        content = row['content'] or ''
        search_terms = search_terms or [query.query]
        
        matches = []
        for term in search_terms:
            matches.extend(self._find_matches(content, term, query.context_chars))
        
        snippet = ""
        if matches:
            first_match = matches[0]
            snippet = f"...{first_match.context_before}<mark>{first_match.matched_text}</mark>{first_match.context_after}..."
        elif content:
            snippet = content[:200] + "..." if len(content) > 200 else content
        
        return SearchResult(
            text_id=row['id'],
            title=row['title'] or '',
            author=row['author'] or '',
            language=row['language'] or '',
            period=row['period'] or '',
            genre=row['genre'] or '',
            source=row.get('source', 'corpus'),
            word_count=row['word_count'] or 0,
            score=row['score'] if 'score' in row.keys() else 1.0,
            matches=matches[:10],
            match_count=len(matches),
            snippet=snippet,
            metadata={
                'diachronic_stage': row.get('diachronic_stage', ''),
                'is_biblical': bool(row.get('is_biblical', False)),
                'is_classical': bool(row.get('is_classical', False)),
                'is_retranslation': bool(row.get('is_retranslation', False)),
            },
        )
    
    def _row_to_result_regex(
        self,
        row: sqlite3.Row,
        query: SearchQuery,
        pattern: re.Pattern
    ) -> SearchResult:
        content = row['content'] or ''
        
        matches = []
        for match in pattern.finditer(content):
            start = match.start()
            end = match.end()
            
            context_start = max(0, start - query.context_chars)
            context_end = min(len(content), end + query.context_chars)
            
            matches.append(SearchMatch(
                position=start,
                length=end - start,
                context_before=content[context_start:start],
                matched_text=match.group(),
                context_after=content[end:context_end],
            ))
        
        snippet = ""
        if matches:
            first_match = matches[0]
            snippet = f"...{first_match.context_before}<mark>{first_match.matched_text}</mark>{first_match.context_after}..."
        
        return SearchResult(
            text_id=row['id'],
            title=row['title'] or '',
            author=row['author'] or '',
            language=row['language'] or '',
            period=row['period'] or '',
            genre=row['genre'] or '',
            source=row.get('source', 'corpus'),
            word_count=row['word_count'] or 0,
            score=1.0,
            matches=matches[:10],
            match_count=len(matches),
            snippet=snippet,
        )
    
    def _find_matches(self, content: str, term: str, context_chars: int) -> List[SearchMatch]:
        matches = []
        
        content_lower = content.lower()
        term_lower = term.lower()
        
        start = 0
        while True:
            pos = content_lower.find(term_lower, start)
            if pos == -1:
                break
            
            context_start = max(0, pos - context_chars)
            context_end = min(len(content), pos + len(term) + context_chars)
            
            lines_before = content[:pos].count('\n')
            
            matches.append(SearchMatch(
                position=pos,
                length=len(term),
                context_before=content[context_start:pos],
                matched_text=content[pos:pos+len(term)],
                context_after=content[pos+len(term):context_end],
                line_number=lines_before + 1,
            ))
            
            start = pos + 1
            
            if len(matches) >= 100:
                break
        
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_texts': 0,
            'total_words': 0,
            'by_language': {},
            'by_period': {},
            'by_genre': {},
            'by_source': {},
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats['total_texts'] = conn.execute(
                    "SELECT COUNT(*) FROM corpus_items"
                ).fetchone()[0]
                
                stats['total_words'] = conn.execute(
                    "SELECT SUM(word_count) FROM corpus_items"
                ).fetchone()[0] or 0
                
                for row in conn.execute(
                    "SELECT language, COUNT(*) as count FROM corpus_items GROUP BY language"
                ):
                    if row[0]:
                        stats['by_language'][row[0]] = row[1]
                
                for row in conn.execute(
                    "SELECT period, COUNT(*) as count FROM corpus_items GROUP BY period"
                ):
                    if row[0]:
                        stats['by_period'][row[0]] = row[1]
                
                for row in conn.execute(
                    "SELECT genre, COUNT(*) as count FROM corpus_items GROUP BY genre"
                ):
                    if row[0]:
                        stats['by_genre'][row[0]] = row[1]
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
        
        return stats
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        filters = {
            'languages': [],
            'periods': [],
            'genres': [],
            'authors': [],
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for row in conn.execute(
                    "SELECT DISTINCT language FROM corpus_items WHERE language IS NOT NULL ORDER BY language"
                ):
                    filters['languages'].append(row[0])
                
                for row in conn.execute(
                    "SELECT DISTINCT period FROM corpus_items WHERE period IS NOT NULL ORDER BY period"
                ):
                    filters['periods'].append(row[0])
                
                for row in conn.execute(
                    "SELECT DISTINCT genre FROM corpus_items WHERE genre IS NOT NULL ORDER BY genre"
                ):
                    filters['genres'].append(row[0])
                
                for row in conn.execute(
                    "SELECT DISTINCT author FROM corpus_items WHERE author IS NOT NULL ORDER BY author"
                ):
                    filters['authors'].append(row[0])
        
        except Exception as e:
            logger.error(f"Error getting available filters: {e}")
        
        return filters
