"""
Prompt Engine - Process natural language queries for linguistic analysis

This module provides a Devin-style prompt interface that:
- Parses natural language queries
- Routes to appropriate analysis modules
- Returns structured results
- Supports follow-up queries

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptType(Enum):
    SEARCH = "search"
    ANALYZE = "analyze"
    COMPARE = "compare"
    EXTRACT = "extract"
    TRANSLATE = "translate"
    VISUALIZE = "visualize"
    EXPORT = "export"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class PromptResult:
    success: bool
    prompt_type: PromptType
    query: str
    response: str
    data: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'prompt_type': self.prompt_type.value,
            'query': self.query,
            'response': self.response,
            'data': self.data,
            'suggestions': self.suggestions,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp,
        }


class PromptEngine:
    
    SEARCH_KEYWORDS = [
        'find', 'search', 'look for', 'show', 'list', 'get', 'retrieve',
        'where', 'which', 'what texts', 'locate', 'discover',
    ]
    
    ANALYZE_KEYWORDS = [
        'analyze', 'analyse', 'examine', 'study', 'investigate', 'parse',
        'annotate', 'tag', 'morphology', 'syntax', 'semantics', 'lemmatize',
    ]
    
    COMPARE_KEYWORDS = [
        'compare', 'contrast', 'difference', 'similar', 'between', 'versus',
        'vs', 'diachronic', 'change', 'evolution', 'development',
    ]
    
    EXTRACT_KEYWORDS = [
        'extract', 'valency', 'pattern', 'argument', 'structure', 'frame',
        'subcategorization', 'complement', 'verb', 'noun',
    ]
    
    TRANSLATE_KEYWORDS = [
        'translate', 'translation', 'intralingual', 'interlingual', 'planudes',
        'retranslation', 'version', 'rendering',
    ]
    
    VISUALIZE_KEYWORDS = [
        'visualize', 'chart', 'graph', 'plot', 'timeline', 'wordcloud',
        'distribution', 'statistics', 'stats', 'show me',
    ]
    
    EXPORT_KEYWORDS = [
        'export', 'download', 'save', 'conllu', 'proiel', 'xml', 'json',
        'bibtex', 'citation', 'cite',
    ]
    
    HELP_KEYWORDS = [
        'help', 'how to', 'what can', 'guide', 'tutorial', 'example',
        'documentation', 'docs',
    ]
    
    def __init__(self):
        self.search_engine = None
        self.translation_tracker = None
        self.chart_generator = None
        self.timeline_generator = None
        self.wordcloud_generator = None
        self.arcas_toolkit = None
        self.influential_texts = None
        
        self._init_modules()
        self.history: List[PromptResult] = []
    
    def _init_modules(self):
        try:
            from hlp_search.text_search import TextSearchEngine
            self.search_engine = TextSearchEngine()
        except ImportError:
            logger.warning("TextSearchEngine not available")
        
        try:
            from hlp_search.translation_tracker import TranslationTracker
            self.translation_tracker = TranslationTracker()
        except ImportError:
            logger.warning("TranslationTracker not available")
        
        try:
            from hlp_search.influential_texts import InfluentialTextsRegistry
            self.influential_texts = InfluentialTextsRegistry()
        except ImportError:
            logger.warning("InfluentialTextsRegistry not available")
        
        try:
            from hlp_visualizations.charts import ChartGenerator
            self.chart_generator = ChartGenerator()
        except ImportError:
            logger.warning("ChartGenerator not available")
        
        try:
            from hlp_visualizations.timelines import TimelineGenerator
            self.timeline_generator = TimelineGenerator()
        except ImportError:
            logger.warning("TimelineGenerator not available")
        
        try:
            from hlp_visualizations.wordclouds import WordCloudGenerator
            self.wordcloud_generator = WordCloudGenerator()
        except ImportError:
            logger.warning("WordCloudGenerator not available")
        
        try:
            from hlp_collection.arcas_tools import ARCASToolkit
            self.arcas_toolkit = ARCASToolkit()
        except ImportError:
            logger.warning("ARCASToolkit not available")
    
    def process(self, query: str) -> PromptResult:
        start_time = datetime.now()
        
        query = query.strip()
        if not query:
            return PromptResult(
                success=False,
                prompt_type=PromptType.UNKNOWN,
                query=query,
                response="Please enter a query.",
                suggestions=self._get_example_queries(),
            )
        
        prompt_type = self._classify_query(query)
        
        try:
            if prompt_type == PromptType.SEARCH:
                result = self._handle_search(query)
            elif prompt_type == PromptType.ANALYZE:
                result = self._handle_analyze(query)
            elif prompt_type == PromptType.COMPARE:
                result = self._handle_compare(query)
            elif prompt_type == PromptType.EXTRACT:
                result = self._handle_extract(query)
            elif prompt_type == PromptType.TRANSLATE:
                result = self._handle_translate(query)
            elif prompt_type == PromptType.VISUALIZE:
                result = self._handle_visualize(query)
            elif prompt_type == PromptType.EXPORT:
                result = self._handle_export(query)
            elif prompt_type == PromptType.HELP:
                result = self._handle_help(query)
            else:
                result = self._handle_unknown(query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            result = PromptResult(
                success=False,
                prompt_type=prompt_type,
                query=query,
                response=f"Error processing query: {str(e)}",
                suggestions=self._get_example_queries(),
            )
        
        end_time = datetime.now()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        self.history.append(result)
        
        return result
    
    def _classify_query(self, query: str) -> PromptType:
        query_lower = query.lower()
        
        for keyword in self.HELP_KEYWORDS:
            if keyword in query_lower:
                return PromptType.HELP
        
        for keyword in self.SEARCH_KEYWORDS:
            if keyword in query_lower:
                return PromptType.SEARCH
        
        for keyword in self.ANALYZE_KEYWORDS:
            if keyword in query_lower:
                return PromptType.ANALYZE
        
        for keyword in self.COMPARE_KEYWORDS:
            if keyword in query_lower:
                return PromptType.COMPARE
        
        for keyword in self.EXTRACT_KEYWORDS:
            if keyword in query_lower:
                return PromptType.EXTRACT
        
        for keyword in self.TRANSLATE_KEYWORDS:
            if keyword in query_lower:
                return PromptType.TRANSLATE
        
        for keyword in self.VISUALIZE_KEYWORDS:
            if keyword in query_lower:
                return PromptType.VISUALIZE
        
        for keyword in self.EXPORT_KEYWORDS:
            if keyword in query_lower:
                return PromptType.EXPORT
        
        return PromptType.SEARCH
    
    def _handle_search(self, query: str) -> PromptResult:
        if not self.search_engine:
            return PromptResult(
                success=False,
                prompt_type=PromptType.SEARCH,
                query=query,
                response="Search engine not available.",
            )
        
        search_term = self._extract_search_term(query)
        language = self._extract_language(query)
        period = self._extract_period(query)
        
        from hlp_search.text_search import SearchQuery, SearchFilter, SearchMode
        
        filters = SearchFilter(
            languages=[language] if language else [],
            periods=[period] if period else [],
        )
        
        search_query = SearchQuery(
            query=search_term,
            mode=SearchMode.EXACT,
            filters=filters,
            limit=20,
        )
        
        results = self.search_engine.search(search_query)
        
        if results:
            response = f"Found {len(results)} texts matching '{search_term}':\n\n"
            for i, r in enumerate(results[:10], 1):
                response += f"{i}. {r.title} ({r.language}, {r.period})\n"
                if r.snippet:
                    response += f"   {r.snippet[:100]}...\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.SEARCH,
                query=query,
                response=response,
                data={'results': [r.to_dict() for r in results]},
                suggestions=[
                    f"Analyze the morphology of '{search_term}'",
                    f"Show word cloud for {language or 'Greek'} texts",
                    f"Compare '{search_term}' across periods",
                ],
            )
        else:
            return PromptResult(
                success=True,
                prompt_type=PromptType.SEARCH,
                query=query,
                response=f"No texts found matching '{search_term}'.",
                suggestions=[
                    "Try a different search term",
                    "Search in a specific language: 'find X in Greek'",
                    "List all available texts",
                ],
            )
    
    def _handle_analyze(self, query: str) -> PromptResult:
        if not self.arcas_toolkit:
            return PromptResult(
                success=False,
                prompt_type=PromptType.ANALYZE,
                query=query,
                response="Analysis toolkit not available.",
            )
        
        text_to_analyze = self._extract_text_to_analyze(query)
        
        if not text_to_analyze:
            return PromptResult(
                success=False,
                prompt_type=PromptType.ANALYZE,
                query=query,
                response="Please specify text to analyze. Example: 'analyze the word logos'",
                suggestions=[
                    "Analyze the word 'logos'",
                    "Parse the sentence 'ho anthropos legei'",
                    "Lemmatize 'legousin'",
                ],
            )
        
        result = self.arcas_toolkit.process_text(text_to_analyze)
        
        response = f"Analysis of '{text_to_analyze}':\n\n"
        
        for token in result.get('tokens', [])[:10]:
            response += f"- {token.get('form', '')}: "
            response += f"lemma={token.get('lemma', '')}, "
            response += f"pos={token.get('pos', '')}\n"
        
        stats = result.get('statistics', {})
        if stats:
            response += f"\nStatistics: {stats.get('token_count', 0)} tokens, "
            response += f"{stats.get('type_count', 0)} types\n"
        
        return PromptResult(
            success=True,
            prompt_type=PromptType.ANALYZE,
            query=query,
            response=response,
            data=result,
            suggestions=[
                f"Export analysis as CoNLL-U",
                f"Compare with Latin equivalent",
                f"Extract valency patterns",
            ],
        )
    
    def _handle_compare(self, query: str) -> PromptResult:
        languages = self._extract_languages_for_comparison(query)
        periods = self._extract_periods_for_comparison(query)
        
        if languages and len(languages) >= 2:
            response = f"Comparing {languages[0]} and {languages[1]}:\n\n"
            
            if self.search_engine:
                stats = self.search_engine.get_statistics()
                for lang in languages:
                    count = stats.get('by_language', {}).get(lang, 0)
                    response += f"- {lang}: {count} texts\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.COMPARE,
                query=query,
                response=response,
                data={'languages': languages},
                suggestions=[
                    f"Show word cloud comparison for {languages[0]} vs {languages[1]}",
                    f"Compare valency patterns",
                    f"Show timeline of both languages",
                ],
            )
        
        if periods and len(periods) >= 2:
            response = f"Comparing {periods[0]} and {periods[1]} periods:\n\n"
            
            if self.search_engine:
                stats = self.search_engine.get_statistics()
                for period in periods:
                    count = stats.get('by_period', {}).get(period, 0)
                    response += f"- {period}: {count} texts\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.COMPARE,
                query=query,
                response=response,
                data={'periods': periods},
                suggestions=[
                    f"Show diachronic changes between periods",
                    f"Compare word frequencies",
                    f"Extract valency changes",
                ],
            )
        
        return PromptResult(
            success=False,
            prompt_type=PromptType.COMPARE,
            query=query,
            response="Please specify what to compare. Example: 'compare Greek and Latin' or 'compare Ancient and Byzantine periods'",
            suggestions=[
                "Compare Greek and Latin",
                "Compare Ancient and Medieval periods",
                "Compare Homer and New Testament vocabulary",
            ],
        )
    
    def _handle_extract(self, query: str) -> PromptResult:
        verb = self._extract_verb(query)
        
        if verb:
            response = f"Valency patterns for '{verb}':\n\n"
            
            response += "Common patterns:\n"
            response += f"- {verb} + NOM (intransitive)\n"
            response += f"- {verb} + NOM + ACC (transitive)\n"
            response += f"- {verb} + NOM + DAT (dative argument)\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.EXTRACT,
                query=query,
                response=response,
                data={'verb': verb},
                suggestions=[
                    f"Compare valency of '{verb}' across periods",
                    f"Find all occurrences of '{verb}'",
                    f"Export valency lexicon entry",
                ],
            )
        
        return PromptResult(
            success=False,
            prompt_type=PromptType.EXTRACT,
            query=query,
            response="Please specify a verb to extract valency patterns. Example: 'extract valency for didomi'",
            suggestions=[
                "Extract valency for 'didomi' (give)",
                "Extract valency for 'lego' (say)",
                "Extract all dative verbs",
            ],
        )
    
    def _handle_translate(self, query: str) -> PromptResult:
        if not self.translation_tracker:
            return PromptResult(
                success=False,
                prompt_type=PromptType.TRANSLATE,
                query=query,
                response="Translation tracker not available.",
            )
        
        if 'planudes' in query.lower():
            translations = self.translation_tracker.get_planudes_translations()
            
            response = "Planudes translations:\n\n"
            for t in translations:
                response += f"- {t.source_title} -> {t.target_title}\n"
                response += f"  ({t.source_language} -> {t.target_language}, {t.translation_year})\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.TRANSLATE,
                query=query,
                response=response,
                data={'translations': [t.to_dict() for t in translations]},
                suggestions=[
                    "Show all intralingual translations",
                    "Compare Planudes with modern translations",
                    "Show translation timeline",
                ],
            )
        
        if 'intralingual' in query.lower():
            translations = self.translation_tracker.get_intralingual_translations()
            
            response = "Intralingual translations (same language, different periods):\n\n"
            for t in translations[:10]:
                response += f"- {t.source_title}: {t.source_period} -> {t.target_period}\n"
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.TRANSLATE,
                query=query,
                response=response,
                data={'translations': [t.to_dict() for t in translations]},
            )
        
        chains = self.translation_tracker.get_all_chains()
        
        response = "Known translation chains:\n\n"
        for chain in chains[:10]:
            response += f"- {chain.original_title} ({chain.original_language})\n"
            response += f"  {len(chain.translations)} translations, {chain.total_languages} languages\n"
        
        return PromptResult(
            success=True,
            prompt_type=PromptType.TRANSLATE,
            query=query,
            response=response,
            data={'chains': [c.to_dict() for c in chains]},
            suggestions=[
                "Show Planudes translations",
                "Show intralingual translations",
                "Show Bible translation chain",
            ],
        )
    
    def _handle_visualize(self, query: str) -> PromptResult:
        query_lower = query.lower()
        
        if 'timeline' in query_lower:
            if not self.timeline_generator:
                return PromptResult(
                    success=False,
                    prompt_type=PromptType.VISUALIZE,
                    query=query,
                    response="Timeline generator not available.",
                )
            
            if 'greek' in query_lower:
                timeline = self.timeline_generator.get_greek_timeline()
            elif 'english' in query_lower:
                timeline = self.timeline_generator.get_english_timeline()
            elif 'byzantine' in query_lower:
                timeline = self.timeline_generator.get_byzantine_timeline()
            else:
                timeline = self.timeline_generator.get_full_timeline()
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.VISUALIZE,
                query=query,
                response=f"Timeline: {timeline['title']} with {len(timeline['events'])} events",
                data={'timeline': timeline},
                suggestions=[
                    "Show Greek timeline",
                    "Show Byzantine timeline",
                    "Show translation timeline",
                ],
            )
        
        if 'wordcloud' in query_lower or 'word cloud' in query_lower:
            if not self.wordcloud_generator:
                return PromptResult(
                    success=False,
                    prompt_type=PromptType.VISUALIZE,
                    query=query,
                    response="Word cloud generator not available.",
                )
            
            from hlp_visualizations.wordclouds import WordCloudConfig
            
            language = self._extract_language(query)
            config = WordCloudConfig(language=language)
            
            frequencies = self.wordcloud_generator.generate_word_cloud(config=config)
            d3_data = self.wordcloud_generator.to_d3_format(frequencies)
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.VISUALIZE,
                query=query,
                response=f"Word cloud generated with {len(frequencies)} words",
                data={'wordcloud': d3_data},
                suggestions=[
                    "Compare word clouds for Greek vs Latin",
                    "Show word cloud for Byzantine texts",
                    "Export word frequencies",
                ],
            )
        
        if 'chart' in query_lower or 'distribution' in query_lower or 'stats' in query_lower:
            if not self.chart_generator:
                return PromptResult(
                    success=False,
                    prompt_type=PromptType.VISUALIZE,
                    query=query,
                    response="Chart generator not available.",
                )
            
            charts = self.chart_generator.generate_all_charts()
            
            return PromptResult(
                success=True,
                prompt_type=PromptType.VISUALIZE,
                query=query,
                response=f"Generated {len(charts)} charts: {', '.join(charts.keys())}",
                data={'charts': {k: v.to_chartjs_config() for k, v in charts.items()}},
                suggestions=[
                    "Show language distribution",
                    "Show period distribution",
                    "Show collection timeline",
                ],
            )
        
        return PromptResult(
            success=False,
            prompt_type=PromptType.VISUALIZE,
            query=query,
            response="Please specify what to visualize.",
            suggestions=[
                "Show timeline of Greek texts",
                "Show word cloud for Byzantine Greek",
                "Show language distribution chart",
            ],
        )
    
    def _handle_export(self, query: str) -> PromptResult:
        query_lower = query.lower()
        
        if 'conllu' in query_lower or 'conll-u' in query_lower:
            return PromptResult(
                success=True,
                prompt_type=PromptType.EXPORT,
                query=query,
                response="CoNLL-U export ready. Use the Export tab in the dashboard to download.",
                data={'format': 'conllu'},
                suggestions=[
                    "Export as PROIEL XML",
                    "Export citations as BibTeX",
                ],
            )
        
        if 'proiel' in query_lower or 'xml' in query_lower:
            return PromptResult(
                success=True,
                prompt_type=PromptType.EXPORT,
                query=query,
                response="PROIEL XML export ready. Use the Export tab in the dashboard to download.",
                data={'format': 'proiel'},
                suggestions=[
                    "Export as CoNLL-U",
                    "Export citations as BibTeX",
                ],
            )
        
        if 'bibtex' in query_lower or 'citation' in query_lower:
            return PromptResult(
                success=True,
                prompt_type=PromptType.EXPORT,
                query=query,
                response="BibTeX export ready. Use the Citations tab in the dashboard to download.",
                data={'format': 'bibtex'},
                suggestions=[
                    "Export as APA format",
                    "Export as MLA format",
                ],
            )
        
        return PromptResult(
            success=False,
            prompt_type=PromptType.EXPORT,
            query=query,
            response="Please specify export format.",
            suggestions=[
                "Export as CoNLL-U",
                "Export as PROIEL XML",
                "Export citations as BibTeX",
            ],
        )
    
    def _handle_help(self, query: str) -> PromptResult:
        help_text = """
Historical Linguistics Platform - Prompt Guide

SEARCH QUERIES:
- "Find texts about [topic]"
- "Search for [word] in Greek texts"
- "List all Byzantine texts"
- "Show texts by Homer"

ANALYSIS QUERIES:
- "Analyze the word [word]"
- "Parse the sentence [sentence]"
- "Lemmatize [word]"
- "Show morphology of [word]"

COMPARISON QUERIES:
- "Compare Greek and Latin"
- "Compare Ancient and Byzantine periods"
- "Show diachronic changes in [word]"

VALENCY QUERIES:
- "Extract valency for [verb]"
- "Show argument structure of [verb]"
- "List dative verbs"

TRANSLATION QUERIES:
- "Show Planudes translations"
- "List intralingual translations"
- "Show Bible translation chain"

VISUALIZATION QUERIES:
- "Show timeline of Greek texts"
- "Show word cloud for [language]"
- "Show language distribution"

EXPORT QUERIES:
- "Export as CoNLL-U"
- "Export as PROIEL XML"
- "Export citations as BibTeX"
"""
        
        return PromptResult(
            success=True,
            prompt_type=PromptType.HELP,
            query=query,
            response=help_text,
            suggestions=self._get_example_queries(),
        )
    
    def _handle_unknown(self, query: str) -> PromptResult:
        return PromptResult(
            success=False,
            prompt_type=PromptType.UNKNOWN,
            query=query,
            response=f"I'm not sure how to handle '{query}'. Try one of these:",
            suggestions=self._get_example_queries(),
        )
    
    def _extract_search_term(self, query: str) -> str:
        for keyword in self.SEARCH_KEYWORDS:
            if keyword in query.lower():
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    term = parts[1].strip()
                    term = re.sub(r'\b(in|for|about|from|by)\b.*$', '', term).strip()
                    return term
        
        return query
    
    def _extract_text_to_analyze(self, query: str) -> str:
        patterns = [
            r"analyze\s+(?:the\s+)?(?:word\s+)?['\"]?([^'\"]+)['\"]?",
            r"parse\s+(?:the\s+)?(?:sentence\s+)?['\"]?([^'\"]+)['\"]?",
            r"lemmatize\s+['\"]?([^'\"]+)['\"]?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_language(self, query: str) -> Optional[str]:
        language_map = {
            'greek': 'grc',
            'ancient greek': 'grc',
            'classical greek': 'grc',
            'byzantine greek': 'grc',
            'koine': 'grc',
            'latin': 'lat',
            'english': 'en',
            'old english': 'ang',
            'middle english': 'enm',
            'modern greek': 'el',
        }
        
        query_lower = query.lower()
        for name, code in language_map.items():
            if name in query_lower:
                return code
        
        return None
    
    def _extract_period(self, query: str) -> Optional[str]:
        periods = [
            'Ancient', 'Classical', 'Hellenistic', 'Koine', 'Late Ancient',
            'Byzantine', 'Medieval', 'Early Modern', 'Modern', 'Renaissance',
        ]
        
        query_lower = query.lower()
        for period in periods:
            if period.lower() in query_lower:
                return period
        
        return None
    
    def _extract_languages_for_comparison(self, query: str) -> List[str]:
        languages = []
        language_map = {
            'greek': 'grc',
            'latin': 'lat',
            'english': 'en',
            'old english': 'ang',
        }
        
        query_lower = query.lower()
        for name, code in language_map.items():
            if name in query_lower:
                languages.append(code)
        
        return languages
    
    def _extract_periods_for_comparison(self, query: str) -> List[str]:
        periods = []
        period_names = [
            'Ancient', 'Classical', 'Hellenistic', 'Byzantine', 'Medieval',
            'Early Modern', 'Modern',
        ]
        
        query_lower = query.lower()
        for period in period_names:
            if period.lower() in query_lower:
                periods.append(period)
        
        return periods
    
    def _extract_verb(self, query: str) -> Optional[str]:
        patterns = [
            r"valency\s+(?:for\s+)?(?:the\s+)?(?:verb\s+)?['\"]?(\w+)['\"]?",
            r"extract\s+(?:valency\s+)?(?:for\s+)?['\"]?(\w+)['\"]?",
            r"argument\s+structure\s+(?:of\s+)?['\"]?(\w+)['\"]?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _get_example_queries(self) -> List[str]:
        return [
            "Find texts about the Trojan War",
            "Analyze the word 'logos'",
            "Compare Greek and Latin",
            "Show Planudes translations",
            "Show timeline of Greek texts",
            "Extract valency for 'didomi'",
            "Export as CoNLL-U",
        ]
    
    def get_history(self) -> List[PromptResult]:
        return self.history
    
    def clear_history(self):
        self.history = []
