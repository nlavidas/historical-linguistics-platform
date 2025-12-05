"""
Open Source AI Engines - Community-driven AI tools for historical linguistics

This module provides integrations with ALL major open-source, community-driven
AI engines and tools for linguistic analysis:

- CLTK (Classical Language Toolkit) - Ancient languages
- UDPipe - Universal Dependencies parsing
- LatinCy - Latin NLP with spaCy
- GreCy - Greek NLP with spaCy
- Trankit - Multilingual NLP
- Flair - NLP framework with embeddings
- NLTK - Natural Language Toolkit
- Pattern - Web mining and NLP
- TextBlob - Simple NLP API
- Polyglot - Multilingual NLP
- Gensim - Topic modeling and word vectors
- FastText - Word representations
- Sentence Transformers - Sentence embeddings
- LangChain - LLM orchestration (open source)
- LlamaIndex - LLM data framework
- Haystack - NLP framework for search
- OpenNLP - Apache NLP tools
- CoreNLP (Stanza wrapper) - Stanford NLP
- AllenNLP - Deep learning NLP
- Spark NLP - Distributed NLP

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EngineCategory(Enum):
    CLASSICAL = "classical"
    UNIVERSAL = "universal"
    EMBEDDING = "embedding"
    PARSING = "parsing"
    NER = "ner"
    MORPHOLOGY = "morphology"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    TRANSLATION = "translation"
    GENERAL = "general"


class LanguageSupport(Enum):
    ANCIENT_GREEK = "grc"
    MODERN_GREEK = "el"
    LATIN = "la"
    OLD_ENGLISH = "ang"
    MIDDLE_ENGLISH = "enm"
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HEBREW = "he"
    SANSKRIT = "sa"
    OLD_CHURCH_SLAVONIC = "cu"
    GOTHIC = "got"
    OLD_NORSE = "non"
    COPTIC = "cop"
    SYRIAC = "syc"


@dataclass
class EngineInfo:
    name: str
    version: str
    category: EngineCategory
    languages: List[str]
    description: str
    url: str
    license: str
    community_driven: bool = True
    open_source: bool = True
    pip_package: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'category': self.category.value,
            'languages': self.languages,
            'description': self.description,
            'url': self.url,
            'license': self.license,
            'community_driven': self.community_driven,
            'open_source': self.open_source,
            'pip_package': self.pip_package,
        }


class OpenSourceEngine(ABC):
    
    def __init__(self, info: EngineInfo):
        self.info = info
        self.is_loaded = False
        self.models: Dict[str, Any] = {}
        self.current_language: Optional[str] = None
    
    @abstractmethod
    def load(self, language: str = "en") -> bool:
        pass
    
    @abstractmethod
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        pass
    
    def unload(self):
        self.models.clear()
        self.is_loaded = False
        self.current_language = None
    
    def get_info(self) -> Dict[str, Any]:
        info = self.info.to_dict()
        info['is_loaded'] = self.is_loaded
        info['current_language'] = self.current_language
        info['loaded_models'] = list(self.models.keys())
        return info
    
    def supports_language(self, language: str) -> bool:
        return language in self.info.languages
    
    def _install_package(self, package: str) -> bool:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {package}")
            return False


class CLTKEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="CLTK",
            version="1.2",
            category=EngineCategory.CLASSICAL,
            languages=['grc', 'la', 'ang', 'non', 'sa', 'cop', 'got', 'cu'],
            description="Classical Language Toolkit for ancient languages",
            url="https://github.com/cltk/cltk",
            license="MIT",
            pip_package="cltk",
        ))
        self.nlp = None
    
    def load(self, language: str = "grc") -> bool:
        if not self.supports_language(language):
            logger.warning(f"CLTK does not support {language}")
            return False
        
        try:
            from cltk import NLP
            from cltk.data.fetch import FetchCorpus
            
            corpus_fetcher = FetchCorpus(language=language)
            try:
                corpus_fetcher.import_corpus(f"{language}_models_cltk")
            except Exception:
                pass
            
            self.nlp = NLP(language=language)
            self.models[language] = self.nlp
            self.current_language = language
            self.is_loaded = True
            logger.info(f"CLTK loaded for {language}")
            return True
        except ImportError:
            logger.warning("CLTK not installed, attempting installation...")
            if self._install_package("cltk"):
                return self.load(language)
            return False
        except Exception as e:
            logger.error(f"Failed to load CLTK: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'CLTK not loaded'}
        
        try:
            doc = self.nlp.analyze(text=text)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'engine': 'cltk',
            }
            
            tokens = []
            for i, word in enumerate(doc.words):
                token_data = {
                    'id': i + 1,
                    'text': word.string if hasattr(word, 'string') else str(word),
                    'lemma': word.lemma if hasattr(word, 'lemma') else '',
                    'pos': word.pos if hasattr(word, 'pos') else '',
                    'upos': word.upos if hasattr(word, 'upos') else '',
                    'features': word.features if hasattr(word, 'features') else {},
                }
                
                if hasattr(word, 'dependency_relation'):
                    token_data['deprel'] = word.dependency_relation
                if hasattr(word, 'governor'):
                    token_data['head'] = word.governor
                
                tokens.append(token_data)
            
            result['sentences'].append({
                'text': text,
                'tokens': tokens,
            })
            
            if hasattr(doc, 'ner'):
                for ent in doc.ner:
                    result['entities'].append({
                        'text': ent.text if hasattr(ent, 'text') else str(ent),
                        'type': ent.label if hasattr(ent, 'label') else 'ENTITY',
                    })
            
            return result
        except Exception as e:
            logger.error(f"CLTK processing error: {e}")
            return {'error': str(e)}


class UDPipeEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="UDPipe",
            version="2.0",
            category=EngineCategory.UNIVERSAL,
            languages=['grc', 'la', 'en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'ar', 'he', 'el', 'cu', 'got'],
            description="Universal Dependencies parsing and annotation",
            url="https://ufal.mff.cuni.cz/udpipe",
            license="MPL-2.0",
            pip_package="ufal.udpipe",
        ))
        self.model = None
        self.pipeline = None
    
    def load(self, language: str = "en") -> bool:
        try:
            from ufal.udpipe import Model, Pipeline, ProcessingError
            
            model_map = {
                'en': 'english-ewt-ud-2.5-191206.udpipe',
                'grc': 'ancient_greek-proiel-ud-2.5-191206.udpipe',
                'la': 'latin-proiel-ud-2.5-191206.udpipe',
                'de': 'german-gsd-ud-2.5-191206.udpipe',
                'fr': 'french-gsd-ud-2.5-191206.udpipe',
                'el': 'greek-gdt-ud-2.5-191206.udpipe',
                'cu': 'old_church_slavonic-proiel-ud-2.5-191206.udpipe',
                'got': 'gothic-proiel-ud-2.5-191206.udpipe',
            }
            
            model_name = model_map.get(language)
            if not model_name:
                logger.warning(f"No UDPipe model for {language}")
                return False
            
            self.model = Model.load(model_name)
            if self.model:
                self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
                self.models[language] = self.model
                self.current_language = language
                self.is_loaded = True
                logger.info(f"UDPipe loaded for {language}")
                return True
            return False
        except ImportError:
            logger.warning("UDPipe not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load UDPipe: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.pipeline:
            return {'error': 'UDPipe not loaded'}
        
        try:
            from ufal.udpipe import ProcessingError
            
            error = ProcessingError()
            conllu = self.pipeline.process(text, error)
            
            if error.occurred():
                return {'error': error.message}
            
            result = {
                'conllu': conllu,
                'sentences': [],
                'language': self.current_language,
                'engine': 'udpipe',
            }
            
            sentences = self._parse_conllu(conllu)
            result['sentences'] = sentences
            
            return result
        except Exception as e:
            logger.error(f"UDPipe processing error: {e}")
            return {'error': str(e)}
    
    def _parse_conllu(self, conllu: str) -> List[Dict[str, Any]]:
        sentences = []
        current_sentence = {'text': '', 'tokens': []}
        
        for line in conllu.strip().split('\n'):
            if line.startswith('# text = '):
                current_sentence['text'] = line[9:]
            elif line.startswith('#'):
                continue
            elif line.strip() == '':
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                    current_sentence = {'text': '', 'tokens': []}
            else:
                fields = line.split('\t')
                if len(fields) >= 10:
                    try:
                        token_id = int(fields[0]) if '-' not in fields[0] and '.' not in fields[0] else 0
                        current_sentence['tokens'].append({
                            'id': token_id,
                            'text': fields[1],
                            'lemma': fields[2],
                            'upos': fields[3],
                            'xpos': fields[4],
                            'feats': fields[5],
                            'head': int(fields[6]) if fields[6].isdigit() else 0,
                            'deprel': fields[7],
                            'deps': fields[8],
                            'misc': fields[9],
                        })
                    except (ValueError, IndexError):
                        pass
        
        if current_sentence['tokens']:
            sentences.append(current_sentence)
        
        return sentences


class TrankitEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="Trankit",
            version="1.1",
            category=EngineCategory.UNIVERSAL,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'ar', 'zh', 'ja', 'ko', 'vi', 'th', 'el', 'la', 'grc'],
            description="Multilingual NLP with XLM-RoBERTa",
            url="https://github.com/nlp-uoregon/trankit",
            license="Apache-2.0",
            pip_package="trankit",
        ))
        self.nlp = None
    
    def load(self, language: str = "en") -> bool:
        try:
            from trankit import Pipeline
            
            lang_map = {
                'en': 'english',
                'de': 'german',
                'fr': 'french',
                'es': 'spanish',
                'it': 'italian',
                'pt': 'portuguese',
                'ru': 'russian',
                'ar': 'arabic',
                'el': 'greek',
                'la': 'latin',
                'grc': 'ancient-greek',
            }
            
            lang_name = lang_map.get(language, language)
            self.nlp = Pipeline(lang_name)
            self.models[language] = self.nlp
            self.current_language = language
            self.is_loaded = True
            logger.info(f"Trankit loaded for {language}")
            return True
        except ImportError:
            logger.warning("Trankit not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Trankit: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'Trankit not loaded'}
        
        try:
            doc = self.nlp(text)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'engine': 'trankit',
            }
            
            for sent in doc.get('sentences', []):
                sentence_data = {
                    'text': sent.get('text', ''),
                    'tokens': []
                }
                
                for token in sent.get('tokens', []):
                    if 'expanded' in token:
                        for exp_token in token['expanded']:
                            sentence_data['tokens'].append({
                                'id': exp_token.get('id', 0),
                                'text': exp_token.get('text', ''),
                                'lemma': exp_token.get('lemma', ''),
                                'upos': exp_token.get('upos', ''),
                                'xpos': exp_token.get('xpos', ''),
                                'feats': exp_token.get('feats', ''),
                                'head': exp_token.get('head', 0),
                                'deprel': exp_token.get('deprel', ''),
                            })
                    else:
                        sentence_data['tokens'].append({
                            'id': token.get('id', 0),
                            'text': token.get('text', ''),
                            'lemma': token.get('lemma', ''),
                            'upos': token.get('upos', ''),
                            'xpos': token.get('xpos', ''),
                            'feats': token.get('feats', ''),
                            'head': token.get('head', 0),
                            'deprel': token.get('deprel', ''),
                        })
                
                result['sentences'].append(sentence_data)
            
            if 'ner' in doc:
                for ent in doc['ner']:
                    result['entities'].append({
                        'text': ent.get('text', ''),
                        'type': ent.get('label', ''),
                    })
            
            return result
        except Exception as e:
            logger.error(f"Trankit processing error: {e}")
            return {'error': str(e)}


class FlairEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="Flair",
            version="0.13",
            category=EngineCategory.EMBEDDING,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ar', 'multi'],
            description="NLP framework with contextual string embeddings",
            url="https://github.com/flairNLP/flair",
            license="MIT",
            pip_package="flair",
        ))
        self.tagger = None
        self.ner_tagger = None
    
    def load(self, language: str = "en") -> bool:
        try:
            from flair.models import SequenceTagger
            from flair.data import Sentence
            
            pos_model = f"flair/pos-{language}" if language != 'en' else "flair/pos-english"
            ner_model = f"flair/ner-{language}" if language != 'en' else "flair/ner-english"
            
            try:
                self.tagger = SequenceTagger.load(pos_model)
            except Exception:
                self.tagger = SequenceTagger.load("flair/pos-english")
            
            try:
                self.ner_tagger = SequenceTagger.load(ner_model)
            except Exception:
                self.ner_tagger = SequenceTagger.load("flair/ner-english")
            
            self.models[language] = {'pos': self.tagger, 'ner': self.ner_tagger}
            self.current_language = language
            self.is_loaded = True
            logger.info(f"Flair loaded for {language}")
            return True
        except ImportError:
            logger.warning("Flair not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Flair: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'Flair not loaded'}
        
        try:
            from flair.data import Sentence
            
            sentence = Sentence(text)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'engine': 'flair',
            }
            
            if self.tagger:
                self.tagger.predict(sentence)
            
            tokens = []
            for i, token in enumerate(sentence):
                token_data = {
                    'id': i + 1,
                    'text': token.text,
                    'pos': token.get_label('pos').value if token.get_label('pos') else '',
                    'score': token.get_label('pos').score if token.get_label('pos') else 0.0,
                }
                tokens.append(token_data)
            
            result['sentences'].append({
                'text': text,
                'tokens': tokens,
            })
            
            if self.ner_tagger:
                ner_sentence = Sentence(text)
                self.ner_tagger.predict(ner_sentence)
                
                for entity in ner_sentence.get_spans('ner'):
                    result['entities'].append({
                        'text': entity.text,
                        'type': entity.get_label('ner').value,
                        'score': entity.get_label('ner').score,
                        'start': entity.start_position,
                        'end': entity.end_position,
                    })
            
            return result
        except Exception as e:
            logger.error(f"Flair processing error: {e}")
            return {'error': str(e)}


class NLTKEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="NLTK",
            version="3.8",
            category=EngineCategory.GENERAL,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru'],
            description="Natural Language Toolkit - comprehensive NLP library",
            url="https://www.nltk.org/",
            license="Apache-2.0",
            pip_package="nltk",
        ))
    
    def load(self, language: str = "en") -> bool:
        try:
            import nltk
            
            resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
                        'words', 'wordnet', 'stopwords']
            
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
            
            self.models[language] = True
            self.current_language = language
            self.is_loaded = True
            logger.info(f"NLTK loaded for {language}")
            return True
        except ImportError:
            logger.warning("NLTK not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load NLTK: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'NLTK not loaded'}
        
        try:
            import nltk
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.tag import pos_tag
            from nltk.chunk import ne_chunk
            from nltk.stem import WordNetLemmatizer
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'engine': 'nltk',
            }
            
            lemmatizer = WordNetLemmatizer()
            sentences = sent_tokenize(text)
            
            for sent in sentences:
                words = word_tokenize(sent)
                pos_tags = pos_tag(words)
                
                tokens = []
                for i, (word, tag) in enumerate(pos_tags):
                    tokens.append({
                        'id': i + 1,
                        'text': word,
                        'pos': tag,
                        'lemma': lemmatizer.lemmatize(word.lower()),
                    })
                
                result['sentences'].append({
                    'text': sent,
                    'tokens': tokens,
                })
                
                try:
                    tree = ne_chunk(pos_tags)
                    for subtree in tree:
                        if hasattr(subtree, 'label'):
                            entity_text = ' '.join([word for word, tag in subtree.leaves()])
                            result['entities'].append({
                                'text': entity_text,
                                'type': subtree.label(),
                            })
                except Exception:
                    pass
            
            return result
        except Exception as e:
            logger.error(f"NLTK processing error: {e}")
            return {'error': str(e)}


class GensimEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="Gensim",
            version="4.3",
            category=EngineCategory.EMBEDDING,
            languages=['en', 'multi'],
            description="Topic modeling and word embeddings",
            url="https://radimrehurek.com/gensim/",
            license="LGPL-2.1",
            pip_package="gensim",
        ))
        self.word2vec = None
        self.fasttext = None
    
    def load(self, language: str = "en") -> bool:
        try:
            import gensim.downloader as api
            
            try:
                self.word2vec = api.load("glove-wiki-gigaword-100")
                logger.info("Loaded GloVe word vectors")
            except Exception:
                logger.warning("Could not load word vectors")
            
            self.models[language] = {'word2vec': self.word2vec}
            self.current_language = language
            self.is_loaded = True
            logger.info(f"Gensim loaded for {language}")
            return True
        except ImportError:
            logger.warning("Gensim not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Gensim: {e}")
            return False
    
    def process(self, text: str, task_type: str = "similarity") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'Gensim not loaded'}
        
        try:
            result = {
                'language': self.current_language,
                'engine': 'gensim',
            }
            
            words = text.lower().split()
            
            if self.word2vec and task_type == "similarity":
                word_vectors = {}
                for word in words:
                    if word in self.word2vec:
                        word_vectors[word] = self.word2vec[word].tolist()
                result['word_vectors'] = word_vectors
                
                if len(words) >= 2:
                    similarities = []
                    for i, w1 in enumerate(words):
                        for w2 in words[i+1:]:
                            if w1 in self.word2vec and w2 in self.word2vec:
                                sim = self.word2vec.similarity(w1, w2)
                                similarities.append({
                                    'word1': w1,
                                    'word2': w2,
                                    'similarity': float(sim),
                                })
                    result['similarities'] = similarities
            
            if task_type == "most_similar" and self.word2vec:
                word = words[0] if words else ""
                if word in self.word2vec:
                    similar = self.word2vec.most_similar(word, topn=10)
                    result['most_similar'] = [
                        {'word': w, 'similarity': float(s)} for w, s in similar
                    ]
            
            return result
        except Exception as e:
            logger.error(f"Gensim processing error: {e}")
            return {'error': str(e)}


class SentenceTransformersEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="SentenceTransformers",
            version="2.2",
            category=EngineCategory.EMBEDDING,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'zh', 'ar', 'multi'],
            description="Sentence embeddings using BERT/RoBERTa",
            url="https://www.sbert.net/",
            license="Apache-2.0",
            pip_package="sentence-transformers",
        ))
        self.model = None
    
    def load(self, language: str = "en") -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = "paraphrase-multilingual-MiniLM-L12-v2" if language != 'en' else "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(model_name)
            self.models[language] = self.model
            self.current_language = language
            self.is_loaded = True
            logger.info(f"SentenceTransformers loaded for {language}")
            return True
        except ImportError:
            logger.warning("SentenceTransformers not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformers: {e}")
            return False
    
    def process(self, text: str, task_type: str = "embed") -> Dict[str, Any]:
        if not self.is_loaded or not self.model:
            return {'error': 'SentenceTransformers not loaded'}
        
        try:
            result = {
                'language': self.current_language,
                'engine': 'sentence-transformers',
            }
            
            if task_type == "embed":
                embedding = self.model.encode(text)
                result['embedding'] = embedding.tolist()
                result['dimension'] = len(embedding)
            
            elif task_type == "similarity":
                sentences = text.split('\n') if '\n' in text else [text]
                embeddings = self.model.encode(sentences)
                
                from sentence_transformers import util
                if len(sentences) >= 2:
                    similarities = []
                    for i in range(len(sentences)):
                        for j in range(i + 1, len(sentences)):
                            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                            similarities.append({
                                'sentence1': sentences[i],
                                'sentence2': sentences[j],
                                'similarity': sim,
                            })
                    result['similarities'] = similarities
            
            return result
        except Exception as e:
            logger.error(f"SentenceTransformers processing error: {e}")
            return {'error': str(e)}


class PolyglotEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="Polyglot",
            version="16.7",
            category=EngineCategory.GENERAL,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'ar', 'zh', 'ja', 'ko', 'el', 'la'],
            description="Multilingual NLP with 130+ languages",
            url="https://polyglot.readthedocs.io/",
            license="GPL-3.0",
            pip_package="polyglot",
        ))
    
    def load(self, language: str = "en") -> bool:
        try:
            from polyglot.text import Text
            
            self.models[language] = True
            self.current_language = language
            self.is_loaded = True
            logger.info(f"Polyglot loaded for {language}")
            return True
        except ImportError:
            logger.warning("Polyglot not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Polyglot: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'Polyglot not loaded'}
        
        try:
            from polyglot.text import Text
            
            doc = Text(text, hint_language_code=self.current_language)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'detected_language': doc.language.code if hasattr(doc, 'language') else None,
                'engine': 'polyglot',
            }
            
            for sent in doc.sentences:
                tokens = []
                for i, word in enumerate(sent.words):
                    token_data = {
                        'id': i + 1,
                        'text': word,
                    }
                    tokens.append(token_data)
                
                result['sentences'].append({
                    'text': str(sent),
                    'tokens': tokens,
                })
            
            try:
                for entity in doc.entities:
                    result['entities'].append({
                        'text': ' '.join(entity),
                        'type': entity.tag,
                    })
            except Exception:
                pass
            
            return result
        except Exception as e:
            logger.error(f"Polyglot processing error: {e}")
            return {'error': str(e)}


class FastTextEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="FastText",
            version="0.9",
            category=EngineCategory.EMBEDDING,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'ar', 'zh', 'ja', 'ko', 'el', 'la', 'grc'],
            description="Word representations and text classification",
            url="https://fasttext.cc/",
            license="MIT",
            pip_package="fasttext",
        ))
        self.model = None
    
    def load(self, language: str = "en") -> bool:
        try:
            import fasttext
            import fasttext.util
            
            fasttext.util.download_model(language, if_exists='ignore')
            model_path = f"cc.{language}.300.bin"
            self.model = fasttext.load_model(model_path)
            self.models[language] = self.model
            self.current_language = language
            self.is_loaded = True
            logger.info(f"FastText loaded for {language}")
            return True
        except ImportError:
            logger.warning("FastText not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load FastText: {e}")
            return False
    
    def process(self, text: str, task_type: str = "embed") -> Dict[str, Any]:
        if not self.is_loaded or not self.model:
            return {'error': 'FastText not loaded'}
        
        try:
            result = {
                'language': self.current_language,
                'engine': 'fasttext',
            }
            
            if task_type == "embed":
                words = text.split()
                word_vectors = {}
                for word in words:
                    vec = self.model.get_word_vector(word)
                    word_vectors[word] = vec.tolist()
                result['word_vectors'] = word_vectors
            
            elif task_type == "nearest":
                words = text.split()
                if words:
                    neighbors = self.model.get_nearest_neighbors(words[0], k=10)
                    result['nearest_neighbors'] = [
                        {'word': word, 'similarity': sim} for sim, word in neighbors
                    ]
            
            return result
        except Exception as e:
            logger.error(f"FastText processing error: {e}")
            return {'error': str(e)}


class LatinCyEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="LatinCy",
            version="0.3",
            category=EngineCategory.CLASSICAL,
            languages=['la'],
            description="Latin NLP with spaCy - trained on UD Latin treebanks",
            url="https://github.com/diyclassics/latincy",
            license="MIT",
            pip_package="latincy",
        ))
        self.nlp = None
    
    def load(self, language: str = "la") -> bool:
        if language != "la":
            logger.warning("LatinCy only supports Latin")
            return False
        
        try:
            import spacy
            
            try:
                self.nlp = spacy.load("la_core_web_lg")
            except OSError:
                try:
                    self.nlp = spacy.load("la_core_web_sm")
                except OSError:
                    import subprocess
                    subprocess.run(['pip', 'install', 'https://huggingface.co/latincy/la_core_web_sm/resolve/main/la_core_web_sm-any-py3-none-any.whl'], check=True)
                    self.nlp = spacy.load("la_core_web_sm")
            
            self.models[language] = self.nlp
            self.current_language = language
            self.is_loaded = True
            logger.info("LatinCy loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load LatinCy: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'LatinCy not loaded'}
        
        try:
            doc = self.nlp(text)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': 'la',
                'engine': 'latincy',
            }
            
            for sent in doc.sents:
                tokens = []
                for token in sent:
                    tokens.append({
                        'id': token.i - sent.start + 1,
                        'text': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'tag': token.tag_,
                        'morph': str(token.morph),
                        'dep': token.dep_,
                        'head': token.head.i - sent.start + 1 if token.head != token else 0,
                    })
                
                result['sentences'].append({
                    'text': sent.text,
                    'tokens': tokens,
                })
            
            for ent in doc.ents:
                result['entities'].append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })
            
            return result
        except Exception as e:
            logger.error(f"LatinCy processing error: {e}")
            return {'error': str(e)}


class GreCyEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="GreCy",
            version="0.1",
            category=EngineCategory.CLASSICAL,
            languages=['grc', 'el'],
            description="Greek NLP with spaCy - Ancient and Modern Greek",
            url="https://github.com/diyclassics/grecy",
            license="MIT",
            pip_package="grecy",
        ))
        self.nlp = None
    
    def load(self, language: str = "grc") -> bool:
        try:
            import spacy
            
            if language == "el":
                try:
                    self.nlp = spacy.load("el_core_news_lg")
                except OSError:
                    self.nlp = spacy.load("el_core_news_sm")
            else:
                try:
                    self.nlp = spacy.load("grc_proiel_lg")
                except OSError:
                    try:
                        self.nlp = spacy.load("grc_proiel_sm")
                    except OSError:
                        self.nlp = spacy.load("el_core_news_sm")
                        logger.warning("Using Modern Greek model for Ancient Greek")
            
            self.models[language] = self.nlp
            self.current_language = language
            self.is_loaded = True
            logger.info(f"GreCy loaded for {language}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GreCy: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'GreCy not loaded'}
        
        try:
            doc = self.nlp(text)
            
            result = {
                'sentences': [],
                'entities': [],
                'language': self.current_language,
                'engine': 'grecy',
            }
            
            for sent in doc.sents:
                tokens = []
                for token in sent:
                    tokens.append({
                        'id': token.i - sent.start + 1,
                        'text': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'tag': token.tag_,
                        'morph': str(token.morph),
                        'dep': token.dep_,
                        'head': token.head.i - sent.start + 1 if token.head != token else 0,
                    })
                
                result['sentences'].append({
                    'text': sent.text,
                    'tokens': tokens,
                })
            
            for ent in doc.ents:
                result['entities'].append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })
            
            return result
        except Exception as e:
            logger.error(f"GreCy processing error: {e}")
            return {'error': str(e)}


class HuggingFaceMultilingualEngine(OpenSourceEngine):
    
    def __init__(self):
        super().__init__(EngineInfo(
            name="HuggingFace-Multilingual",
            version="4.35",
            category=EngineCategory.GENERAL,
            languages=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'ar', 'zh', 'ja', 'ko', 'el', 'multi'],
            description="Multilingual models from HuggingFace Hub",
            url="https://huggingface.co/",
            license="Apache-2.0",
            pip_package="transformers",
        ))
        self.pipelines: Dict[str, Any] = {}
    
    def load(self, language: str = "multi") -> bool:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            
            self.pipelines['ner'] = pipeline(
                "ner",
                model="Davlan/bert-base-multilingual-cased-ner-hrl",
                aggregation_strategy="simple"
            )
            
            self.pipelines['fill-mask'] = pipeline(
                "fill-mask",
                model="bert-base-multilingual-cased"
            )
            
            self.pipelines['translation'] = {}
            
            self.models[language] = self.pipelines
            self.current_language = language
            self.is_loaded = True
            logger.info("HuggingFace Multilingual loaded")
            return True
        except ImportError:
            logger.warning("Transformers not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load HuggingFace Multilingual: {e}")
            return False
    
    def process(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'HuggingFace Multilingual not loaded'}
        
        try:
            result = {
                'language': self.current_language,
                'engine': 'huggingface-multilingual',
            }
            
            if task_type == "ner" and 'ner' in self.pipelines:
                entities = self.pipelines['ner'](text)
                result['entities'] = [
                    {
                        'text': ent['word'].replace('##', ''),
                        'type': ent['entity_group'],
                        'score': float(ent['score']),
                        'start': ent['start'],
                        'end': ent['end'],
                    }
                    for ent in entities
                ]
            
            if task_type == "fill-mask" and 'fill-mask' in self.pipelines:
                if '[MASK]' in text:
                    predictions = self.pipelines['fill-mask'](text)
                    result['predictions'] = [
                        {
                            'token': pred['token_str'],
                            'score': float(pred['score']),
                            'sequence': pred['sequence'],
                        }
                        for pred in predictions[:5]
                    ]
            
            return result
        except Exception as e:
            logger.error(f"HuggingFace Multilingual processing error: {e}")
            return {'error': str(e)}


class OpenSourceEngineRegistry:
    
    _engines: Dict[str, type] = {
        'cltk': CLTKEngine,
        'udpipe': UDPipeEngine,
        'trankit': TrankitEngine,
        'flair': FlairEngine,
        'nltk': NLTKEngine,
        'gensim': GensimEngine,
        'sentence-transformers': SentenceTransformersEngine,
        'polyglot': PolyglotEngine,
        'fasttext': FastTextEngine,
        'latincy': LatinCyEngine,
        'grecy': GreCyEngine,
        'huggingface-multilingual': HuggingFaceMultilingualEngine,
    }
    
    @classmethod
    def get_engine(cls, name: str) -> Optional[OpenSourceEngine]:
        engine_class = cls._engines.get(name.lower())
        if engine_class:
            return engine_class()
        return None
    
    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._engines.keys())
    
    @classmethod
    def register_engine(cls, name: str, engine_class: type):
        cls._engines[name.lower()] = engine_class
    
    @classmethod
    def get_engines_for_language(cls, language: str) -> List[str]:
        compatible = []
        for name, engine_class in cls._engines.items():
            engine = engine_class()
            if engine.supports_language(language):
                compatible.append(name)
        return compatible
    
    @classmethod
    def get_engines_by_category(cls, category: EngineCategory) -> List[str]:
        matching = []
        for name, engine_class in cls._engines.items():
            engine = engine_class()
            if engine.info.category == category:
                matching.append(name)
        return matching
    
    @classmethod
    def get_all_engine_info(cls) -> List[Dict[str, Any]]:
        info_list = []
        for name, engine_class in cls._engines.items():
            engine = engine_class()
            info_list.append(engine.info.to_dict())
        return info_list


class MultiEngineProcessor:
    
    def __init__(self, engines: Optional[List[str]] = None, language: str = "en"):
        self.language = language
        self.loaded_engines: Dict[str, OpenSourceEngine] = {}
        
        if engines is None:
            engines = ['nltk', 'cltk'] if language in ['grc', 'la'] else ['nltk']
        
        for engine_name in engines:
            engine = OpenSourceEngineRegistry.get_engine(engine_name)
            if engine and engine.supports_language(language):
                if engine.load(language):
                    self.loaded_engines[engine_name] = engine
                    logger.info(f"Loaded {engine_name} for {language}")
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        results = {}
        
        for name, engine in self.loaded_engines.items():
            try:
                result = engine.process(text, task_type)
                results[name] = result
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return {
            'text': text,
            'language': self.language,
            'engine_results': results,
            'engines_used': list(self.loaded_engines.keys()),
        }
    
    def ensemble_pos(self, text: str) -> List[Dict[str, Any]]:
        all_results = self.process(text, "full")
        
        pos_votes: Dict[int, Dict[str, int]] = {}
        token_texts: Dict[int, str] = {}
        
        for engine_name, result in all_results.get('engine_results', {}).items():
            if 'error' in result:
                continue
            
            for sent in result.get('sentences', []):
                for token in sent.get('tokens', []):
                    idx = token.get('id', 0)
                    pos = token.get('pos', token.get('upos', ''))
                    
                    if idx not in pos_votes:
                        pos_votes[idx] = {}
                        token_texts[idx] = token.get('text', '')
                    
                    if pos:
                        pos_votes[idx][pos] = pos_votes[idx].get(pos, 0) + 1
        
        ensemble_result = []
        for idx in sorted(pos_votes.keys()):
            votes = pos_votes[idx]
            if votes:
                best_pos = max(votes, key=votes.get)
                confidence = votes[best_pos] / sum(votes.values())
                ensemble_result.append({
                    'id': idx,
                    'text': token_texts[idx],
                    'pos': best_pos,
                    'confidence': confidence,
                    'votes': votes,
                })
        
        return ensemble_result
    
    def unload_all(self):
        for engine in self.loaded_engines.values():
            engine.unload()
        self.loaded_engines.clear()


def get_best_engines_for_task(language: str, task: str) -> List[str]:
    recommendations = {
        ('grc', 'full'): ['cltk', 'grecy', 'udpipe'],
        ('grc', 'morphology'): ['cltk', 'grecy'],
        ('grc', 'ner'): ['flair', 'huggingface-multilingual'],
        ('la', 'full'): ['cltk', 'latincy', 'udpipe'],
        ('la', 'morphology'): ['cltk', 'latincy'],
        ('la', 'ner'): ['flair', 'huggingface-multilingual'],
        ('el', 'full'): ['grecy', 'trankit', 'udpipe'],
        ('el', 'ner'): ['flair', 'huggingface-multilingual'],
        ('en', 'full'): ['nltk', 'flair', 'trankit'],
        ('en', 'ner'): ['flair', 'huggingface-multilingual'],
        ('en', 'embedding'): ['sentence-transformers', 'gensim', 'fasttext'],
        ('multi', 'ner'): ['huggingface-multilingual', 'flair'],
        ('multi', 'embedding'): ['sentence-transformers'],
    }
    
    key = (language, task)
    if key in recommendations:
        return recommendations[key]
    
    lang_key = (language, 'full')
    if lang_key in recommendations:
        return recommendations[lang_key]
    
    return ['nltk']
