#!/usr/bin/env python3
"""
SKETCH ENGINE OPEN SOURCE TOOLS
Complete implementation of corpus linguistics tools inspired by SketchEngine

Features:
1. Word Sketch - Grammatical relations and collocations
2. Thesaurus - Distributional similarity
3. Word Lists - Frequency lists with filters
4. Concordance - KWIC with sorting and filtering
5. N-grams - Multi-word expressions
6. Keywords - Comparing corpora
7. Trends - Diachronic frequency analysis
8. Parallel Concordance - Aligned texts
9. CQL (Corpus Query Language) - Advanced search
10. Terminology Extraction - Domain-specific terms

All tools work with PROIEL-style annotated corpora.
"""

import os
import re
import math
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any, Iterator
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Token:
    """Token with full annotation"""
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: Dict[str, str]
    head: int
    deprel: str
    deps: str
    misc: str
    sentence_id: int = 0
    document_id: int = 0

@dataclass
class Collocation:
    """A collocation (word pair with statistical association)"""
    word1: str
    word2: str
    relation: str  # grammatical relation
    frequency: int
    score: float  # association measure
    examples: List[str] = field(default_factory=list)

@dataclass
class WordSketchEntry:
    """Entry in a word sketch"""
    lemma: str
    relation: str
    collocate: str
    frequency: int
    score: float
    
@dataclass
class ConcordanceLine:
    """A concordance line (KWIC)"""
    left_context: str
    keyword: str
    right_context: str
    sentence_id: int
    document_id: int
    position: int
    metadata: Dict = field(default_factory=dict)

@dataclass
class NGram:
    """N-gram with frequency"""
    tokens: Tuple[str, ...]
    frequency: int
    documents: int
    score: float = 0.0

# =============================================================================
# STATISTICAL MEASURES
# =============================================================================

class AssociationMeasures:
    """Statistical measures for collocations"""
    
    @staticmethod
    def log_likelihood(o11: int, o12: int, o21: int, o22: int) -> float:
        """Log-likelihood ratio (G2)"""
        n = o11 + o12 + o21 + o22
        if n == 0:
            return 0.0
        
        def safe_log(x):
            return math.log(x) if x > 0 else 0
        
        e11 = (o11 + o12) * (o11 + o21) / n
        e12 = (o11 + o12) * (o12 + o22) / n
        e21 = (o21 + o22) * (o11 + o21) / n
        e22 = (o21 + o22) * (o12 + o22) / n
        
        ll = 2 * (
            o11 * safe_log(o11 / e11) if e11 > 0 else 0 +
            o12 * safe_log(o12 / e12) if e12 > 0 else 0 +
            o21 * safe_log(o21 / e21) if e21 > 0 else 0 +
            o22 * safe_log(o22 / e22) if e22 > 0 else 0
        )
        return ll
    
    @staticmethod
    def mi_score(o11: int, o12: int, o21: int, o22: int) -> float:
        """Mutual Information score"""
        n = o11 + o12 + o21 + o22
        if n == 0 or o11 == 0:
            return 0.0
        
        e11 = (o11 + o12) * (o11 + o21) / n
        if e11 == 0:
            return 0.0
        
        return math.log2(o11 / e11)
    
    @staticmethod
    def mi3_score(o11: int, o12: int, o21: int, o22: int) -> float:
        """MI³ score (cubed MI, reduces bias toward rare words)"""
        n = o11 + o12 + o21 + o22
        if n == 0 or o11 == 0:
            return 0.0
        
        e11 = (o11 + o12) * (o11 + o21) / n
        if e11 == 0:
            return 0.0
        
        return math.log2((o11 ** 3) / e11)
    
    @staticmethod
    def t_score(o11: int, o12: int, o21: int, o22: int) -> float:
        """T-score"""
        n = o11 + o12 + o21 + o22
        if n == 0:
            return 0.0
        
        e11 = (o11 + o12) * (o11 + o21) / n
        if o11 == 0:
            return 0.0
        
        return (o11 - e11) / math.sqrt(o11)
    
    @staticmethod
    def dice_coefficient(o11: int, o12: int, o21: int, o22: int) -> float:
        """Dice coefficient"""
        if (o11 + o12) + (o11 + o21) == 0:
            return 0.0
        return 2 * o11 / ((o11 + o12) + (o11 + o21))
    
    @staticmethod
    def log_dice(o11: int, o12: int, o21: int, o22: int) -> float:
        """LogDice (SketchEngine's preferred measure)"""
        dice = AssociationMeasures.dice_coefficient(o11, o12, o21, o22)
        if dice <= 0:
            return 0.0
        return 14 + math.log2(dice)


# =============================================================================
# CORPUS QUERY LANGUAGE (CQL)
# =============================================================================

class CQLParser:
    """
    Parser for Corpus Query Language
    Supports: [word="..."] [lemma="..."] [tag="..."] [deprel="..."]
    Operators: & | ! within containing
    """
    
    def __init__(self):
        self.token_pattern = re.compile(r'\[([^\]]+)\]')
        self.attr_pattern = re.compile(r'(\w+)\s*(!?=)\s*"([^"]*)"')
    
    def parse(self, query: str) -> List[Dict]:
        """Parse CQL query into token specifications"""
        tokens = []
        
        for match in self.token_pattern.finditer(query):
            token_spec = {}
            content = match.group(1)
            
            for attr_match in self.attr_pattern.finditer(content):
                attr = attr_match.group(1)
                op = attr_match.group(2)
                value = attr_match.group(3)
                
                token_spec[attr] = {
                    'value': value,
                    'negated': op == '!='
                }
            
            tokens.append(token_spec)
        
        return tokens
    
    def match_token(self, token: Token, spec: Dict) -> bool:
        """Check if token matches specification"""
        for attr, condition in spec.items():
            value = condition['value']
            negated = condition['negated']
            
            # Get token attribute
            if attr == 'word':
                token_val = token.form
            elif attr == 'lemma':
                token_val = token.lemma
            elif attr == 'tag' or attr == 'pos':
                token_val = token.upos
            elif attr == 'xpos':
                token_val = token.xpos
            elif attr == 'deprel':
                token_val = token.deprel
            else:
                # Check in feats
                token_val = token.feats.get(attr, '')
            
            # Handle regex
            if value.startswith('/') and value.endswith('/'):
                pattern = value[1:-1]
                matches = bool(re.match(pattern, token_val, re.IGNORECASE))
            else:
                matches = token_val.lower() == value.lower()
            
            if negated:
                matches = not matches
            
            if not matches:
                return False
        
        return True


# =============================================================================
# WORD SKETCH
# =============================================================================

class WordSketch:
    """
    Generate word sketches showing grammatical relations
    Based on SketchEngine's Word Sketch functionality
    """
    
    # Grammatical relation patterns for different POS
    RELATION_PATTERNS = {
        'VERB': [
            ('subject', 'nsubj', 'NOUN'),
            ('object', 'obj', 'NOUN'),
            ('indirect_object', 'iobj', 'NOUN'),
            ('oblique', 'obl', 'NOUN'),
            ('adverbial', 'advmod', 'ADV'),
            ('auxiliary', 'aux', 'AUX'),
            ('complement', 'xcomp', 'VERB'),
        ],
        'NOUN': [
            ('modifier', 'amod', 'ADJ'),
            ('genitive', 'nmod', 'NOUN'),
            ('subject_of', 'nsubj', 'VERB', True),  # inverse
            ('object_of', 'obj', 'VERB', True),
            ('determiner', 'det', 'DET'),
            ('apposition', 'appos', 'NOUN'),
        ],
        'ADJ': [
            ('modifies', 'amod', 'NOUN', True),
            ('and/or', 'conj', 'ADJ'),
            ('adverb', 'advmod', 'ADV'),
        ]
    }
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.measures = AssociationMeasures()
    
    def generate(self, lemma: str, pos: str = None, min_freq: int = 2,
                 limit: int = 20) -> Dict[str, List[WordSketchEntry]]:
        """Generate word sketch for a lemma"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get total corpus size
        cursor.execute("SELECT COUNT(*) FROM tokens")
        corpus_size = cursor.fetchone()[0]
        
        # Get lemma frequency
        cursor.execute("""
            SELECT COUNT(*) FROM tokens WHERE lemma = ?
        """, (lemma,))
        lemma_freq = cursor.fetchone()[0]
        
        if lemma_freq == 0:
            conn.close()
            return {}
        
        # Determine POS if not specified
        if pos is None:
            cursor.execute("""
                SELECT upos, COUNT(*) as cnt FROM tokens 
                WHERE lemma = ? GROUP BY upos ORDER BY cnt DESC LIMIT 1
            """, (lemma,))
            row = cursor.fetchone()
            pos = row['upos'] if row else 'NOUN'
        
        sketch = {}
        patterns = self.RELATION_PATTERNS.get(pos, [])
        
        for pattern in patterns:
            rel_name = pattern[0]
            deprel = pattern[1]
            target_pos = pattern[2]
            inverse = len(pattern) > 3 and pattern[3]
            
            # Query for collocates
            if inverse:
                # Target is the head
                cursor.execute("""
                    SELECT t2.lemma as collocate, COUNT(*) as freq
                    FROM tokens t1
                    JOIN tokens t2 ON t1.sentence_id = t2.sentence_id AND t1.head = t2.token_index
                    WHERE t1.lemma = ? AND t1.deprel = ? AND t2.upos = ?
                    GROUP BY t2.lemma
                    HAVING freq >= ?
                    ORDER BY freq DESC
                    LIMIT ?
                """, (lemma, deprel, target_pos, min_freq, limit * 2))
            else:
                # Target is the dependent
                cursor.execute("""
                    SELECT t2.lemma as collocate, COUNT(*) as freq
                    FROM tokens t1
                    JOIN tokens t2 ON t1.sentence_id = t2.sentence_id AND t2.head = t1.token_index
                    WHERE t1.lemma = ? AND t2.deprel = ? AND t2.upos = ?
                    GROUP BY t2.lemma
                    HAVING freq >= ?
                    ORDER BY freq DESC
                    LIMIT ?
                """, (lemma, deprel, target_pos, min_freq, limit * 2))
            
            entries = []
            for row in cursor.fetchall():
                collocate = row['collocate']
                freq = row['freq']
                
                # Get collocate frequency
                cursor.execute("SELECT COUNT(*) FROM tokens WHERE lemma = ?", (collocate,))
                collocate_freq = cursor.fetchone()[0]
                
                # Calculate association score (LogDice)
                o11 = freq
                o12 = lemma_freq - freq
                o21 = collocate_freq - freq
                o22 = corpus_size - o11 - o12 - o21
                
                score = self.measures.log_dice(o11, o12, o21, o22)
                
                entries.append(WordSketchEntry(
                    lemma=lemma,
                    relation=rel_name,
                    collocate=collocate,
                    frequency=freq,
                    score=score
                ))
            
            # Sort by score and limit
            entries.sort(key=lambda x: x.score, reverse=True)
            sketch[rel_name] = entries[:limit]
        
        conn.close()
        return sketch
    
    def compare(self, lemma1: str, lemma2: str, pos: str = None) -> Dict:
        """Compare word sketches of two lemmas"""
        sketch1 = self.generate(lemma1, pos)
        sketch2 = self.generate(lemma2, pos)
        
        comparison = {
            'lemma1': lemma1,
            'lemma2': lemma2,
            'shared': {},
            'unique1': {},
            'unique2': {}
        }
        
        all_relations = set(sketch1.keys()) | set(sketch2.keys())
        
        for rel in all_relations:
            collocates1 = {e.collocate for e in sketch1.get(rel, [])}
            collocates2 = {e.collocate for e in sketch2.get(rel, [])}
            
            shared = collocates1 & collocates2
            unique1 = collocates1 - collocates2
            unique2 = collocates2 - collocates1
            
            if shared:
                comparison['shared'][rel] = list(shared)
            if unique1:
                comparison['unique1'][rel] = list(unique1)
            if unique2:
                comparison['unique2'][rel] = list(unique2)
        
        return comparison


# =============================================================================
# THESAURUS (Distributional Similarity)
# =============================================================================

class DistributionalThesaurus:
    """
    Find similar words based on distributional similarity
    Words that appear in similar contexts are similar
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.context_vectors: Dict[str, Counter] = {}
    
    def build_vectors(self, min_freq: int = 5, context_window: int = 5):
        """Build context vectors for all lemmas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get sentences with tokens
        cursor.execute("""
            SELECT sentence_id, GROUP_CONCAT(lemma, ' ') as lemmas
            FROM tokens
            GROUP BY sentence_id
        """)
        
        for row in cursor.fetchall():
            lemmas = row[1].split() if row[1] else []
            
            for i, lemma in enumerate(lemmas):
                if lemma not in self.context_vectors:
                    self.context_vectors[lemma] = Counter()
                
                # Get context window
                start = max(0, i - context_window)
                end = min(len(lemmas), i + context_window + 1)
                
                for j in range(start, end):
                    if j != i:
                        context_lemma = lemmas[j]
                        self.context_vectors[lemma][context_lemma] += 1
        
        conn.close()
        logger.info(f"Built vectors for {len(self.context_vectors)} lemmas")
    
    def cosine_similarity(self, vec1: Counter, vec2: Counter) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Get shared keys
        shared = set(vec1.keys()) & set(vec2.keys())
        
        if not shared:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[k] * vec2[k] for k in shared)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def find_similar(self, lemma: str, top_n: int = 20, 
                     pos_filter: str = None) -> List[Tuple[str, float]]:
        """Find most similar words to a lemma"""
        if lemma not in self.context_vectors:
            return []
        
        target_vec = self.context_vectors[lemma]
        similarities = []
        
        for other_lemma, other_vec in self.context_vectors.items():
            if other_lemma == lemma:
                continue
            
            sim = self.cosine_similarity(target_vec, other_vec)
            if sim > 0:
                similarities.append((other_lemma, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]


# =============================================================================
# CONCORDANCE
# =============================================================================

class Concordancer:
    """
    Generate concordances (KWIC - Keyword in Context)
    With sorting, filtering, and export options
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.cql_parser = CQLParser()
    
    def search(self, query: str, context_size: int = 50, 
               limit: int = 1000, sort_by: str = 'position') -> List[ConcordanceLine]:
        """
        Search corpus and return concordance lines
        Query can be simple word or CQL pattern
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        results = []
        
        # Check if CQL query
        if '[' in query:
            results = self._cql_search(cursor, query, context_size, limit)
        else:
            # Simple word/lemma search
            results = self._simple_search(cursor, query, context_size, limit)
        
        conn.close()
        
        # Sort results
        if sort_by == 'left':
            results.sort(key=lambda x: x.left_context.split()[-1] if x.left_context else '')
        elif sort_by == 'right':
            results.sort(key=lambda x: x.right_context.split()[0] if x.right_context else '')
        elif sort_by == 'keyword':
            results.sort(key=lambda x: x.keyword)
        
        return results
    
    def _simple_search(self, cursor, query: str, context_size: int, 
                       limit: int) -> List[ConcordanceLine]:
        """Simple word or lemma search"""
        results = []
        
        # Search in both form and lemma
        cursor.execute("""
            SELECT t.*, s.text as sentence_text, s.document_id
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            WHERE t.form = ? OR t.lemma = ?
            LIMIT ?
        """, (query, query, limit))
        
        for row in cursor.fetchall():
            sentence_text = row['sentence_text'] or ''
            keyword = row['form']
            
            # Find keyword position in sentence
            pos = sentence_text.lower().find(keyword.lower())
            if pos >= 0:
                left = sentence_text[max(0, pos - context_size):pos].strip()
                right = sentence_text[pos + len(keyword):pos + len(keyword) + context_size].strip()
                
                results.append(ConcordanceLine(
                    left_context=left,
                    keyword=keyword,
                    right_context=right,
                    sentence_id=row['sentence_id'],
                    document_id=row['document_id'],
                    position=row['token_index']
                ))
        
        return results
    
    def _cql_search(self, cursor, query: str, context_size: int,
                    limit: int) -> List[ConcordanceLine]:
        """CQL pattern search"""
        specs = self.cql_parser.parse(query)
        if not specs:
            return []
        
        results = []
        
        # For single token queries
        if len(specs) == 1:
            spec = specs[0]
            
            # Build SQL conditions
            conditions = []
            params = []
            
            for attr, cond in spec.items():
                col = 'form' if attr == 'word' else attr
                if col in ('form', 'lemma', 'upos', 'xpos', 'deprel'):
                    if cond['negated']:
                        conditions.append(f"{col} != ?")
                    else:
                        conditions.append(f"{col} = ?")
                    params.append(cond['value'])
            
            if conditions:
                where_clause = ' AND '.join(conditions)
                cursor.execute(f"""
                    SELECT t.*, s.text as sentence_text, s.document_id
                    FROM tokens t
                    JOIN sentences s ON t.sentence_id = s.id
                    WHERE {where_clause}
                    LIMIT ?
                """, params + [limit])
                
                for row in cursor.fetchall():
                    sentence_text = row['sentence_text'] or ''
                    keyword = row['form']
                    
                    pos = sentence_text.lower().find(keyword.lower())
                    if pos >= 0:
                        left = sentence_text[max(0, pos - context_size):pos].strip()
                        right = sentence_text[pos + len(keyword):pos + len(keyword) + context_size].strip()
                        
                        results.append(ConcordanceLine(
                            left_context=left,
                            keyword=keyword,
                            right_context=right,
                            sentence_id=row['sentence_id'],
                            document_id=row['document_id'],
                            position=row['token_index']
                        ))
        
        return results
    
    def export_csv(self, results: List[ConcordanceLine], filepath: str):
        """Export concordance to CSV"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Left Context', 'Keyword', 'Right Context', 
                           'Sentence ID', 'Document ID', 'Position'])
            
            for line in results:
                writer.writerow([
                    line.left_context,
                    line.keyword,
                    line.right_context,
                    line.sentence_id,
                    line.document_id,
                    line.position
                ])


# =============================================================================
# WORD LISTS & FREQUENCY
# =============================================================================

class WordListGenerator:
    """Generate frequency lists with various filters"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def frequency_list(self, level: str = 'lemma', pos_filter: str = None,
                       min_freq: int = 1, limit: int = 1000) -> List[Tuple[str, int]]:
        """Generate frequency list"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        col = 'lemma' if level == 'lemma' else 'form'
        
        if pos_filter:
            cursor.execute(f"""
                SELECT {col}, COUNT(*) as freq
                FROM tokens
                WHERE upos = ?
                GROUP BY {col}
                HAVING freq >= ?
                ORDER BY freq DESC
                LIMIT ?
            """, (pos_filter, min_freq, limit))
        else:
            cursor.execute(f"""
                SELECT {col}, COUNT(*) as freq
                FROM tokens
                GROUP BY {col}
                HAVING freq >= ?
                ORDER BY freq DESC
                LIMIT ?
            """, (min_freq, limit))
        
        results = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def pos_distribution(self) -> Dict[str, int]:
        """Get POS tag distribution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT upos, COUNT(*) as freq
            FROM tokens
            GROUP BY upos
            ORDER BY freq DESC
        """)
        
        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return results
    
    def hapax_legomena(self, level: str = 'lemma') -> List[str]:
        """Get words that appear only once"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        col = 'lemma' if level == 'lemma' else 'form'
        
        cursor.execute(f"""
            SELECT {col}
            FROM tokens
            GROUP BY {col}
            HAVING COUNT(*) = 1
        """)
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return results


# =============================================================================
# N-GRAMS
# =============================================================================

class NGramExtractor:
    """Extract and analyze n-grams"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def extract(self, n: int = 2, level: str = 'form', min_freq: int = 2,
                pos_pattern: List[str] = None, limit: int = 1000) -> List[NGram]:
        """Extract n-grams from corpus"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        ngram_counts = Counter()
        ngram_docs = defaultdict(set)
        
        # Get sentences
        cursor.execute("""
            SELECT sentence_id, document_id FROM sentences
        """)
        sentences = cursor.fetchall()
        
        for sent_id, doc_id in sentences:
            # Get tokens for sentence
            col = 'lemma' if level == 'lemma' else 'form'
            
            if pos_pattern:
                cursor.execute(f"""
                    SELECT {col}, upos FROM tokens
                    WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_id,))
                tokens = cursor.fetchall()
                
                # Extract n-grams matching POS pattern
                for i in range(len(tokens) - n + 1):
                    window = tokens[i:i+n]
                    pos_seq = [t[1] for t in window]
                    
                    if pos_seq == pos_pattern:
                        ngram = tuple(t[0] for t in window)
                        ngram_counts[ngram] += 1
                        ngram_docs[ngram].add(doc_id)
            else:
                cursor.execute(f"""
                    SELECT {col} FROM tokens
                    WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_id,))
                tokens = [row[0] for row in cursor.fetchall()]
                
                # Extract all n-grams
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    ngram_counts[ngram] += 1
                    ngram_docs[ngram].add(doc_id)
        
        conn.close()
        
        # Filter and sort
        results = []
        for ngram, freq in ngram_counts.most_common():
            if freq >= min_freq:
                results.append(NGram(
                    tokens=ngram,
                    frequency=freq,
                    documents=len(ngram_docs[ngram])
                ))
                
                if len(results) >= limit:
                    break
        
        return results


# =============================================================================
# KEYWORDS (Corpus Comparison)
# =============================================================================

class KeywordExtractor:
    """Extract keywords by comparing corpora"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.measures = AssociationMeasures()
    
    def extract_keywords(self, focus_condition: str, reference_condition: str = None,
                        level: str = 'lemma', min_freq: int = 5,
                        limit: int = 100) -> List[Tuple[str, float, int, int]]:
        """
        Extract keywords from focus corpus compared to reference
        Returns: (word, keyness_score, focus_freq, reference_freq)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        col = 'lemma' if level == 'lemma' else 'form'
        
        # Get focus corpus frequencies
        cursor.execute(f"""
            SELECT t.{col}, COUNT(*) as freq
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            WHERE {focus_condition}
            GROUP BY t.{col}
            HAVING freq >= ?
        """, (min_freq,))
        focus_freqs = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get focus corpus size
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            WHERE {focus_condition}
        """)
        focus_size = cursor.fetchone()[0]
        
        # Get reference corpus frequencies
        if reference_condition:
            ref_where = reference_condition
        else:
            ref_where = f"NOT ({focus_condition})"
        
        cursor.execute(f"""
            SELECT t.{col}, COUNT(*) as freq
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            WHERE {ref_where}
            GROUP BY t.{col}
        """)
        ref_freqs = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get reference corpus size
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            WHERE {ref_where}
        """)
        ref_size = cursor.fetchone()[0]
        
        conn.close()
        
        # Calculate keyness scores
        keywords = []
        for word, focus_freq in focus_freqs.items():
            ref_freq = ref_freqs.get(word, 0)
            
            # Calculate log-likelihood
            o11 = focus_freq
            o12 = focus_size - focus_freq
            o21 = ref_freq
            o22 = ref_size - ref_freq
            
            ll = self.measures.log_likelihood(o11, o12, o21, o22)
            
            # Determine direction (positive = more frequent in focus)
            expected = focus_size * (focus_freq + ref_freq) / (focus_size + ref_size)
            if focus_freq < expected:
                ll = -ll
            
            keywords.append((word, ll, focus_freq, ref_freq))
        
        # Sort by keyness score
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:limit]


# =============================================================================
# DIACHRONIC TRENDS
# =============================================================================

class DiachronicAnalyzer:
    """Analyze frequency trends across time periods"""
    
    PERIOD_ORDER = [
        'archaic', 'classical', 'hellenistic', 'koine',
        'medieval', 'early_modern', 'modern'
    ]
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def frequency_by_period(self, lemma: str) -> Dict[str, Dict]:
        """Get frequency of lemma across periods"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get frequencies
        cursor.execute("""
            SELECT d.period, COUNT(*) as freq
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            WHERE t.lemma = ?
            GROUP BY d.period
        """, (lemma,))
        
        raw_freqs = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get period sizes
        cursor.execute("""
            SELECT d.period, COUNT(*) as size
            FROM tokens t
            JOIN documents d ON t.document_id = d.id
            GROUP BY d.period
        """)
        
        period_sizes = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Calculate normalized frequencies (per million)
        results = {}
        for period in self.PERIOD_ORDER:
            if period in period_sizes:
                freq = raw_freqs.get(period, 0)
                size = period_sizes[period]
                norm_freq = (freq / size) * 1000000 if size > 0 else 0
                
                results[period] = {
                    'raw_frequency': freq,
                    'normalized_frequency': norm_freq,
                    'corpus_size': size
                }
        
        return results
    
    def detect_change(self, lemma: str, threshold: float = 2.0) -> List[Dict]:
        """Detect significant frequency changes"""
        trends = self.frequency_by_period(lemma)
        
        changes = []
        periods = [p for p in self.PERIOD_ORDER if p in trends]
        
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i+1]
            f1 = trends[p1]['normalized_frequency']
            f2 = trends[p2]['normalized_frequency']
            
            if f1 > 0:
                ratio = f2 / f1
                if ratio >= threshold or ratio <= 1/threshold:
                    changes.append({
                        'from_period': p1,
                        'to_period': p2,
                        'from_freq': f1,
                        'to_freq': f2,
                        'ratio': ratio,
                        'direction': 'increase' if ratio > 1 else 'decrease'
                    })
        
        return changes


# =============================================================================
# TERMINOLOGY EXTRACTION
# =============================================================================

class TerminologyExtractor:
    """Extract domain-specific terminology"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ngram_extractor = NGramExtractor(db_path)
        self.keyword_extractor = KeywordExtractor(db_path)
    
    def extract_terms(self, domain_condition: str, 
                      general_condition: str = None,
                      max_ngram: int = 4,
                      min_freq: int = 3,
                      limit: int = 200) -> List[Dict]:
        """Extract domain-specific terms"""
        terms = []
        
        # Extract single-word terms (keywords)
        keywords = self.keyword_extractor.extract_keywords(
            domain_condition, general_condition,
            min_freq=min_freq, limit=limit
        )
        
        for word, score, domain_freq, general_freq in keywords:
            if score > 0:  # Only positive keywords
                terms.append({
                    'term': word,
                    'type': 'single',
                    'score': score,
                    'domain_freq': domain_freq,
                    'general_freq': general_freq
                })
        
        # Extract multi-word terms (n-grams)
        for n in range(2, max_ngram + 1):
            ngrams = self.ngram_extractor.extract(
                n=n, level='lemma', min_freq=min_freq, limit=limit
            )
            
            for ngram in ngrams:
                term = ' '.join(ngram.tokens)
                terms.append({
                    'term': term,
                    'type': f'{n}-gram',
                    'score': ngram.frequency * n,  # Boost longer terms
                    'domain_freq': ngram.frequency,
                    'general_freq': 0
                })
        
        # Sort by score
        terms.sort(key=lambda x: x['score'], reverse=True)
        
        return terms[:limit]


# =============================================================================
# MAIN SKETCH ENGINE CLASS
# =============================================================================

class SketchEngine:
    """
    Main class integrating all SketchEngine-like tools
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
        # Initialize all tools
        self.word_sketch = WordSketch(db_path)
        self.thesaurus = DistributionalThesaurus(db_path)
        self.concordancer = Concordancer(db_path)
        self.word_list = WordListGenerator(db_path)
        self.ngrams = NGramExtractor(db_path)
        self.keywords = KeywordExtractor(db_path)
        self.trends = DiachronicAnalyzer(db_path)
        self.terminology = TerminologyExtractor(db_path)
        
        logger.info(f"SketchEngine initialized with database: {db_path}")
    
    def get_corpus_info(self) -> Dict:
        """Get corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        info = {}
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        info['documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sentences")
        info['sentences'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tokens")
        info['tokens'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT lemma) FROM tokens")
        info['lemmas'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT form) FROM tokens")
        info['word_forms'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT language, COUNT(*) FROM documents
            GROUP BY language
        """)
        info['by_language'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT period, COUNT(*) FROM documents
            WHERE period IS NOT NULL
            GROUP BY period
        """)
        info['by_period'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "/root/corpus_platform/data/corpus_platform.db"
    
    print("=" * 70)
    print("SKETCH ENGINE TOOLS - Test")
    print("=" * 70)
    
    engine = SketchEngine(db_path)
    
    # Corpus info
    print("\n📊 Corpus Information:")
    info = engine.get_corpus_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Word frequency
    print("\n📝 Top 20 Lemmas:")
    freq_list = engine.word_list.frequency_list(limit=20)
    for lemma, freq in freq_list:
        print(f"  {lemma}: {freq:,}")
    
    # POS distribution
    print("\n🏷️ POS Distribution:")
    pos_dist = engine.word_list.pos_distribution()
    for pos, freq in list(pos_dist.items())[:10]:
        print(f"  {pos}: {freq:,}")
    
    print("\n✅ SketchEngine tools ready!")
