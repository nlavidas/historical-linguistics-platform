"""
Word Cloud Generator - Create word frequency data for visualization

This module generates word frequency data for word cloud visualizations,
supporting multiple languages and filtering options.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WordCloudConfig:
    min_word_length: int = 3
    max_words: int = 100
    exclude_stopwords: bool = True
    language: Optional[str] = None
    period: Optional[str] = None
    genre: Optional[str] = None
    normalize_greek: bool = True
    lemmatize: bool = False


@dataclass
class WordFrequency:
    word: str
    count: int
    normalized: str = ""
    lemma: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'count': self.count,
            'normalized': self.normalized or self.word,
            'lemma': self.lemma or self.word,
        }


class WordCloudGenerator:
    
    GREEK_STOPWORDS = {
        'καί', 'δέ', 'τε', 'γάρ', 'μέν', 'οὖν', 'ἀλλά', 'ἤ', 'εἰ', 'ὡς',
        'ὅτι', 'ἐν', 'ἐπί', 'πρός', 'ἐκ', 'ἀπό', 'διά', 'κατά', 'μετά', 'περί',
        'ὑπό', 'παρά', 'ἀνά', 'σύν', 'εἰς', 'ὁ', 'ἡ', 'τό', 'τοῦ', 'τῆς',
        'τῷ', 'τήν', 'τόν', 'οἱ', 'αἱ', 'τά', 'τῶν', 'τοῖς', 'ταῖς', 'τούς',
        'τάς', 'αὐτός', 'αὐτή', 'αὐτό', 'αὐτοῦ', 'αὐτῆς', 'αὐτῷ', 'αὐτήν',
        'αὐτόν', 'οὗτος', 'αὕτη', 'τοῦτο', 'ἐκεῖνος', 'ὅς', 'ἥ', 'ὅ', 'τίς',
        'τί', 'πᾶς', 'πᾶσα', 'πᾶν', 'εἰμί', 'ἐστί', 'ἐστίν', 'εἶναι', 'ἦν',
        'οὐ', 'οὐκ', 'οὐχ', 'μή', 'ἄν', 'δή', 'γε', 'τοι', 'νῦν', 'ἔτι',
        'και', 'δε', 'τε', 'γαρ', 'μεν', 'ουν', 'αλλα', 'η', 'ει', 'ως',
        'οτι', 'εν', 'επι', 'προς', 'εκ', 'απο', 'δια', 'κατα', 'μετα', 'περι',
        'υπο', 'παρα', 'ανα', 'συν', 'εις', 'ο', 'η', 'το', 'του', 'της',
        'τω', 'την', 'τον', 'οι', 'αι', 'τα', 'των', 'τοις', 'ταις', 'τους',
        'τας', 'αυτος', 'αυτη', 'αυτο', 'αυτου', 'αυτης', 'αυτω', 'αυτην',
        'αυτον', 'ουτος', 'αυτη', 'τουτο', 'εκεινος', 'ος', 'η', 'ο', 'τις',
        'τι', 'πας', 'πασα', 'παν', 'ειμι', 'εστι', 'εστιν', 'ειναι', 'ην',
        'ου', 'ουκ', 'ουχ', 'μη', 'αν', 'δη', 'γε', 'τοι', 'νυν', 'ετι',
    }
    
    LATIN_STOPWORDS = {
        'et', 'in', 'est', 'non', 'cum', 'ad', 'ut', 'sed', 'si', 'quod',
        'qui', 'quae', 'quam', 'per', 'ex', 'de', 'ab', 'pro', 'sub', 'inter',
        'hic', 'haec', 'hoc', 'ille', 'illa', 'illud', 'is', 'ea', 'id',
        'ipse', 'ipsa', 'ipsum', 'idem', 'eadem', 'quis', 'quid', 'omnis',
        'omne', 'sum', 'esse', 'fui', 'erat', 'sunt', 'eram', 'erit',
        'ne', 'nec', 'neque', 'aut', 'vel', 'atque', 'ac', 'autem', 'enim',
        'nam', 'tamen', 'ergo', 'igitur', 'itaque', 'iam', 'nunc', 'tum',
        'a', 'e', 'o', 'i', 'u', 'me', 'te', 'se', 'nos', 'vos',
    }
    
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'while', 'about', 'against',
        'thee', 'thou', 'thy', 'thine', 'ye', 'hath', 'doth', 'art', 'wilt',
        'shalt', 'hast', 'didst', 'wouldst', 'couldst', 'shouldst', 'unto',
        'upon', 'thereof', 'whereof', 'wherein', 'wherefore', 'thereby',
    }
    
    OLD_ENGLISH_STOPWORDS = {
        'se', 'seo', 'þæt', 'þa', 'þone', 'þære', 'þæs', 'þam', 'þy', 'þe',
        'and', 'ond', 'oþþe', 'ac', 'ne', 'na', 'no', 'gif', 'þonne', 'þa',
        'nu', 'her', 'þær', 'hwær', 'hwa', 'hwæt', 'hu', 'for', 'mid', 'on',
        'in', 'to', 'æt', 'of', 'fram', 'be', 'æfter', 'ofer', 'under',
        'ic', 'þu', 'he', 'heo', 'hit', 'we', 'ge', 'hie', 'me', 'þe', 'him',
        'hire', 'us', 'eow', 'min', 'þin', 'his', 'hire', 'ure', 'eower',
        'is', 'wæs', 'beon', 'wesan', 'hæfde', 'habban', 'wolde', 'willan',
        'sceolde', 'sculan', 'mihte', 'magan', 'moste', 'motan',
    }
    
    def __init__(self, db_path: str = "data/corpus_platform.db"):
        self.db_path = Path(db_path)
        self.lemmatizer = None
        self._init_lemmatizer()
    
    def _init_lemmatizer(self):
        try:
            from hlp_collection.arcas_tools import GreekLemmatizer
            self.lemmatizer = GreekLemmatizer()
        except ImportError:
            logger.warning("Lemmatizer not available")
    
    def get_stopwords(self, language: str) -> set:
        stopword_map = {
            'grc': self.GREEK_STOPWORDS,
            'lat': self.LATIN_STOPWORDS,
            'en': self.ENGLISH_STOPWORDS,
            'ang': self.OLD_ENGLISH_STOPWORDS,
            'enm': self.ENGLISH_STOPWORDS,
        }
        return stopword_map.get(language, set())
    
    def normalize_greek(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = unicodedata.normalize('NFC', text)
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def generate_word_cloud(
        self,
        text: Optional[str] = None,
        text_id: Optional[int] = None,
        config: Optional[WordCloudConfig] = None
    ) -> List[WordFrequency]:
        config = config or WordCloudConfig()
        
        if text is None and text_id is not None:
            text = self._get_text_content(text_id)
        
        if text is None:
            text = self._get_corpus_text(config)
        
        if not text:
            return []
        
        tokens = self.tokenize(text)
        
        if config.normalize_greek:
            tokens = [self.normalize_greek(t) for t in tokens]
        else:
            tokens = [t.lower() for t in tokens]
        
        tokens = [t for t in tokens if len(t) >= config.min_word_length]
        
        if config.exclude_stopwords:
            stopwords = self.get_stopwords(config.language or 'grc')
            stopwords = stopwords.union(self.ENGLISH_STOPWORDS)
            if config.normalize_greek:
                stopwords = {self.normalize_greek(s) for s in stopwords}
            tokens = [t for t in tokens if t not in stopwords]
        
        word_counts = Counter(tokens)
        
        frequencies = []
        for word, count in word_counts.most_common(config.max_words):
            lemma = word
            if config.lemmatize and self.lemmatizer:
                lemma = self.lemmatizer.lemmatize(word)
            
            frequencies.append(WordFrequency(
                word=word,
                count=count,
                normalized=word,
                lemma=lemma,
            ))
        
        return frequencies
    
    def _get_text_content(self, text_id: int) -> Optional[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT content FROM corpus_items WHERE id = ?",
                    (text_id,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logger.error(f"Error getting text content: {e}")
        return None
    
    def _get_corpus_text(self, config: WordCloudConfig) -> str:
        texts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conditions = []
                params = []
                
                if config.language:
                    conditions.append("language = ?")
                    params.append(config.language)
                
                if config.period:
                    conditions.append("period = ?")
                    params.append(config.period)
                
                if config.genre:
                    conditions.append("genre = ?")
                    params.append(config.genre)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                cursor = conn.execute(
                    f"SELECT content FROM corpus_items WHERE {where_clause} LIMIT 50",
                    params
                )
                
                for row in cursor:
                    if row[0]:
                        texts.append(row[0][:10000])
        except Exception as e:
            logger.error(f"Error getting corpus text: {e}")
        
        return ' '.join(texts)
    
    def generate_comparison_clouds(
        self,
        language1: str,
        language2: str,
        config: Optional[WordCloudConfig] = None
    ) -> Dict[str, List[WordFrequency]]:
        config = config or WordCloudConfig()
        
        config1 = WordCloudConfig(
            min_word_length=config.min_word_length,
            max_words=config.max_words,
            exclude_stopwords=config.exclude_stopwords,
            language=language1,
            normalize_greek=config.normalize_greek,
            lemmatize=config.lemmatize,
        )
        
        config2 = WordCloudConfig(
            min_word_length=config.min_word_length,
            max_words=config.max_words,
            exclude_stopwords=config.exclude_stopwords,
            language=language2,
            normalize_greek=config.normalize_greek,
            lemmatize=config.lemmatize,
        )
        
        return {
            language1: self.generate_word_cloud(config=config1),
            language2: self.generate_word_cloud(config=config2),
        }
    
    def generate_period_comparison(
        self,
        period1: str,
        period2: str,
        language: str = 'grc',
        config: Optional[WordCloudConfig] = None
    ) -> Dict[str, List[WordFrequency]]:
        config = config or WordCloudConfig()
        
        config1 = WordCloudConfig(
            min_word_length=config.min_word_length,
            max_words=config.max_words,
            exclude_stopwords=config.exclude_stopwords,
            language=language,
            period=period1,
            normalize_greek=config.normalize_greek,
            lemmatize=config.lemmatize,
        )
        
        config2 = WordCloudConfig(
            min_word_length=config.min_word_length,
            max_words=config.max_words,
            exclude_stopwords=config.exclude_stopwords,
            language=language,
            period=period2,
            normalize_greek=config.normalize_greek,
            lemmatize=config.lemmatize,
        )
        
        return {
            period1: self.generate_word_cloud(config=config1),
            period2: self.generate_word_cloud(config=config2),
        }
    
    def to_d3_format(self, frequencies: List[WordFrequency]) -> List[Dict[str, Any]]:
        if not frequencies:
            return []
        
        max_count = max(f.count for f in frequencies)
        
        return [
            {
                'text': f.word,
                'size': 10 + (f.count / max_count) * 90,
                'count': f.count,
            }
            for f in frequencies
        ]
    
    def to_json(self, frequencies: List[WordFrequency]) -> str:
        return json.dumps([f.to_dict() for f in frequencies], indent=2)
