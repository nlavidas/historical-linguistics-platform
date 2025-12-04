"""
Linguistic Prompts Library - Pre-built prompts for common linguistic queries

This module provides a library of prompt templates for:
- Morphological analysis
- Syntactic parsing
- Valency extraction
- Diachronic comparison
- Translation analysis

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    MORPHOLOGY = "morphology"
    SYNTAX = "syntax"
    SEMANTICS = "semantics"
    VALENCY = "valency"
    DIACHRONIC = "diachronic"
    TRANSLATION = "translation"
    CORPUS = "corpus"
    VISUALIZATION = "visualization"


@dataclass
class PromptTemplate:
    id: str
    name: str
    description: str
    category: PromptCategory
    template: str
    parameters: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    def format(self, **kwargs) -> str:
        result = self.template
        for param in self.parameters:
            if param in kwargs:
                result = result.replace(f"{{{param}}}", str(kwargs[param]))
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'template': self.template,
            'parameters': self.parameters,
            'examples': self.examples,
        }


class LinguisticPromptLibrary:
    
    def __init__(self):
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        self._add_morphology_prompts()
        self._add_syntax_prompts()
        self._add_valency_prompts()
        self._add_diachronic_prompts()
        self._add_translation_prompts()
        self._add_corpus_prompts()
        self._add_visualization_prompts()
    
    def _add_morphology_prompts(self):
        self.prompts['analyze_word'] = PromptTemplate(
            id='analyze_word',
            name='Analyze Word',
            description='Perform morphological analysis on a single word',
            category=PromptCategory.MORPHOLOGY,
            template='Analyze the morphology of the word "{word}" in {language}',
            parameters=['word', 'language'],
            examples=[
                'Analyze the morphology of the word "logos" in Greek',
                'Analyze the morphology of the word "fecit" in Latin',
            ],
        )
        
        self.prompts['lemmatize_text'] = PromptTemplate(
            id='lemmatize_text',
            name='Lemmatize Text',
            description='Find the dictionary form (lemma) of words in a text',
            category=PromptCategory.MORPHOLOGY,
            template='Lemmatize the following {language} text: "{text}"',
            parameters=['text', 'language'],
            examples=[
                'Lemmatize the following Greek text: "ho anthropos legei"',
                'Lemmatize the following Latin text: "veni vidi vici"',
            ],
        )
        
        self.prompts['parse_verb'] = PromptTemplate(
            id='parse_verb',
            name='Parse Verb Form',
            description='Analyze a verb form for person, number, tense, mood, voice',
            category=PromptCategory.MORPHOLOGY,
            template='Parse the {language} verb form "{verb}" (person, number, tense, mood, voice)',
            parameters=['verb', 'language'],
            examples=[
                'Parse the Greek verb form "legousin" (person, number, tense, mood, voice)',
                'Parse the Latin verb form "amaverat" (person, number, tense, mood, voice)',
            ],
        )
        
        self.prompts['parse_noun'] = PromptTemplate(
            id='parse_noun',
            name='Parse Noun Form',
            description='Analyze a noun form for case, number, gender',
            category=PromptCategory.MORPHOLOGY,
            template='Parse the {language} noun form "{noun}" (case, number, gender)',
            parameters=['noun', 'language'],
            examples=[
                'Parse the Greek noun form "anthropou" (case, number, gender)',
                'Parse the Latin noun form "regum" (case, number, gender)',
            ],
        )
        
        self.prompts['find_cognates'] = PromptTemplate(
            id='find_cognates',
            name='Find Cognates',
            description='Find cognate words across Indo-European languages',
            category=PromptCategory.MORPHOLOGY,
            template='Find cognates of "{word}" across Indo-European languages',
            parameters=['word'],
            examples=[
                'Find cognates of "pater" across Indo-European languages',
                'Find cognates of "mother" across Indo-European languages',
            ],
        )
    
    def _add_syntax_prompts(self):
        self.prompts['parse_sentence'] = PromptTemplate(
            id='parse_sentence',
            name='Parse Sentence',
            description='Perform syntactic analysis on a sentence',
            category=PromptCategory.SYNTAX,
            template='Parse the {language} sentence: "{sentence}"',
            parameters=['sentence', 'language'],
            examples=[
                'Parse the Greek sentence: "ho anthropos ton logon legei"',
                'Parse the Latin sentence: "Caesar Galliam vicit"',
            ],
        )
        
        self.prompts['identify_clauses'] = PromptTemplate(
            id='identify_clauses',
            name='Identify Clauses',
            description='Identify main and subordinate clauses in a sentence',
            category=PromptCategory.SYNTAX,
            template='Identify the clause structure in: "{sentence}"',
            parameters=['sentence'],
            examples=[
                'Identify the clause structure in: "When Caesar arrived, the Gauls fled"',
            ],
        )
        
        self.prompts['word_order'] = PromptTemplate(
            id='word_order',
            name='Analyze Word Order',
            description='Analyze word order patterns in a text',
            category=PromptCategory.SYNTAX,
            template='Analyze the word order patterns in {language} {text_type}',
            parameters=['language', 'text_type'],
            examples=[
                'Analyze the word order patterns in Greek prose',
                'Analyze the word order patterns in Latin poetry',
            ],
        )
    
    def _add_valency_prompts(self):
        self.prompts['extract_valency'] = PromptTemplate(
            id='extract_valency',
            name='Extract Valency',
            description='Extract valency patterns for a verb',
            category=PromptCategory.VALENCY,
            template='Extract valency patterns for the {language} verb "{verb}"',
            parameters=['verb', 'language'],
            examples=[
                'Extract valency patterns for the Greek verb "didomi"',
                'Extract valency patterns for the Latin verb "dare"',
            ],
        )
        
        self.prompts['compare_valency'] = PromptTemplate(
            id='compare_valency',
            name='Compare Valency',
            description='Compare valency patterns of a verb across periods',
            category=PromptCategory.VALENCY,
            template='Compare valency patterns of "{verb}" in {period1} vs {period2} {language}',
            parameters=['verb', 'period1', 'period2', 'language'],
            examples=[
                'Compare valency patterns of "didomi" in Classical vs Koine Greek',
                'Compare valency patterns of "give" in Old vs Modern English',
            ],
        )
        
        self.prompts['dative_verbs'] = PromptTemplate(
            id='dative_verbs',
            name='Find Dative Verbs',
            description='Find verbs that take dative arguments',
            category=PromptCategory.VALENCY,
            template='List {language} verbs that take dative arguments in {period}',
            parameters=['language', 'period'],
            examples=[
                'List Greek verbs that take dative arguments in Classical period',
                'List Latin verbs that take dative arguments in Classical period',
            ],
        )
        
        self.prompts['argument_alternation'] = PromptTemplate(
            id='argument_alternation',
            name='Argument Alternation',
            description='Find argument structure alternations for a verb',
            category=PromptCategory.VALENCY,
            template='Show argument alternations for "{verb}" in {language}',
            parameters=['verb', 'language'],
            examples=[
                'Show argument alternations for "break" in English',
                'Show argument alternations for "open" in Greek',
            ],
        )
    
    def _add_diachronic_prompts(self):
        self.prompts['trace_change'] = PromptTemplate(
            id='trace_change',
            name='Trace Linguistic Change',
            description='Trace the diachronic development of a linguistic feature',
            category=PromptCategory.DIACHRONIC,
            template='Trace the development of {feature} from {period1} to {period2} in {language}',
            parameters=['feature', 'period1', 'period2', 'language'],
            examples=[
                'Trace the development of the dative case from Classical to Modern Greek',
                'Trace the development of word order from Old to Modern English',
            ],
        )
        
        self.prompts['compare_periods'] = PromptTemplate(
            id='compare_periods',
            name='Compare Periods',
            description='Compare linguistic features across historical periods',
            category=PromptCategory.DIACHRONIC,
            template='Compare {feature} in {period1} vs {period2} {language}',
            parameters=['feature', 'period1', 'period2', 'language'],
            examples=[
                'Compare verb morphology in Classical vs Byzantine Greek',
                'Compare vocabulary in Old vs Middle English',
            ],
        )
        
        self.prompts['grammaticalization'] = PromptTemplate(
            id='grammaticalization',
            name='Grammaticalization Path',
            description='Trace grammaticalization of a lexical item',
            category=PromptCategory.DIACHRONIC,
            template='Trace the grammaticalization of "{word}" in {language}',
            parameters=['word', 'language'],
            examples=[
                'Trace the grammaticalization of "have" in English',
                'Trace the grammaticalization of "echo" in Greek',
            ],
        )
    
    def _add_translation_prompts(self):
        self.prompts['find_translations'] = PromptTemplate(
            id='find_translations',
            name='Find Translations',
            description='Find translations of a text',
            category=PromptCategory.TRANSLATION,
            template='Find all translations of "{title}" into {target_language}',
            parameters=['title', 'target_language'],
            examples=[
                'Find all translations of "Iliad" into English',
                'Find all translations of "Bible" into English',
            ],
        )
        
        self.prompts['compare_translations'] = PromptTemplate(
            id='compare_translations',
            name='Compare Translations',
            description='Compare different translations of the same text',
            category=PromptCategory.TRANSLATION,
            template='Compare translations of "{title}" by {translator1} and {translator2}',
            parameters=['title', 'translator1', 'translator2'],
            examples=[
                'Compare translations of "Iliad" by Chapman and Pope',
                'Compare translations of "Bible" by Wycliffe and Tyndale',
            ],
        )
        
        self.prompts['planudes_analysis'] = PromptTemplate(
            id='planudes_analysis',
            name='Planudes Translation Analysis',
            description='Analyze Planudes intralingual/interlingual translations',
            category=PromptCategory.TRANSLATION,
            template='Analyze Planudes translation of "{title}" from {source_lang} to {target_lang}',
            parameters=['title', 'source_lang', 'target_lang'],
            examples=[
                'Analyze Planudes translation of "Metamorphoses" from Latin to Greek',
                'Analyze Planudes translation of "Boethius" from Latin to Greek',
            ],
        )
        
        self.prompts['intralingual'] = PromptTemplate(
            id='intralingual',
            name='Intralingual Translation',
            description='Find intralingual translations (same language, different periods)',
            category=PromptCategory.TRANSLATION,
            template='Find intralingual translations of "{title}" in {language}',
            parameters=['title', 'language'],
            examples=[
                'Find intralingual translations of "Homer" in Greek',
                'Find intralingual translations of "Beowulf" in English',
            ],
        )
    
    def _add_corpus_prompts(self):
        self.prompts['search_corpus'] = PromptTemplate(
            id='search_corpus',
            name='Search Corpus',
            description='Search for texts in the corpus',
            category=PromptCategory.CORPUS,
            template='Find {language} texts about "{topic}" from {period}',
            parameters=['language', 'topic', 'period'],
            examples=[
                'Find Greek texts about "war" from Classical period',
                'Find Latin texts about "philosophy" from Ancient period',
            ],
        )
        
        self.prompts['list_texts'] = PromptTemplate(
            id='list_texts',
            name='List Texts',
            description='List texts by various criteria',
            category=PromptCategory.CORPUS,
            template='List all {language} texts by {author}',
            parameters=['language', 'author'],
            examples=[
                'List all Greek texts by Homer',
                'List all Latin texts by Virgil',
            ],
        )
        
        self.prompts['corpus_stats'] = PromptTemplate(
            id='corpus_stats',
            name='Corpus Statistics',
            description='Get statistics about the corpus',
            category=PromptCategory.CORPUS,
            template='Show statistics for {language} texts in {period}',
            parameters=['language', 'period'],
            examples=[
                'Show statistics for Greek texts in Byzantine period',
                'Show statistics for English texts in Medieval period',
            ],
        )
    
    def _add_visualization_prompts(self):
        self.prompts['timeline'] = PromptTemplate(
            id='timeline',
            name='Show Timeline',
            description='Display a timeline of texts or events',
            category=PromptCategory.VISUALIZATION,
            template='Show timeline of {language} texts from {period}',
            parameters=['language', 'period'],
            examples=[
                'Show timeline of Greek texts from all periods',
                'Show timeline of English texts from Medieval period',
            ],
        )
        
        self.prompts['wordcloud'] = PromptTemplate(
            id='wordcloud',
            name='Word Cloud',
            description='Generate a word cloud for texts',
            category=PromptCategory.VISUALIZATION,
            template='Show word cloud for {language} {genre} texts',
            parameters=['language', 'genre'],
            examples=[
                'Show word cloud for Greek epic texts',
                'Show word cloud for Latin philosophical texts',
            ],
        )
        
        self.prompts['distribution'] = PromptTemplate(
            id='distribution',
            name='Distribution Chart',
            description='Show distribution of texts by various criteria',
            category=PromptCategory.VISUALIZATION,
            template='Show distribution of texts by {criterion}',
            parameters=['criterion'],
            examples=[
                'Show distribution of texts by language',
                'Show distribution of texts by period',
                'Show distribution of texts by genre',
            ],
        )
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        return self.prompts.get(prompt_id)
    
    def get_all_prompts(self) -> List[PromptTemplate]:
        return list(self.prompts.values())
    
    def get_prompts_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        return [p for p in self.prompts.values() if p.category == category]
    
    def search_prompts(self, query: str) -> List[PromptTemplate]:
        query_lower = query.lower()
        results = []
        
        for prompt in self.prompts.values():
            if (query_lower in prompt.name.lower() or
                query_lower in prompt.description.lower() or
                any(query_lower in ex.lower() for ex in prompt.examples)):
                results.append(prompt)
        
        return results
    
    def get_categories(self) -> List[str]:
        return [c.value for c in PromptCategory]
    
    def get_examples(self, limit: int = 20) -> List[str]:
        examples = []
        for prompt in self.prompts.values():
            examples.extend(prompt.examples)
        return examples[:limit]
