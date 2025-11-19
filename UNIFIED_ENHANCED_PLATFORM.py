#!/usr/bin/env python3
"""
UNIFIED ENHANCED CORPUS PLATFORM
Integrates all systems with multiple free open-source AI models
Prof. Nikolaos Lavidas - HFRI-NKUA
"""

import sys
import os
import sqlite3
import logging
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Set environment
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))
os.environ['TRANSFORMERS_CACHE'] = str(Path('Z:/models/transformers'))
os.environ['HF_HOME'] = str(Path('Z:/models/transformers'))

sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'unified_platform.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class UnifiedEnhancedPlatform:
    """
    Unified platform integrating:
    - Text collection
    - Multi-AI annotation
    - PROIEL treebank generation
    - Valency extraction
    - Diachronic analysis
    - Enhanced logging
    - Monitoring
    """
    
    def __init__(self):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.models_loaded = {}
        self.stats = {
            'texts_collected': 0,
            'texts_annotated': 0,
            'treebanks_generated': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        logger.info("="*80)
        logger.info("UNIFIED ENHANCED CORPUS PLATFORM - INITIALIZING")
        logger.info("="*80)
        
        self.setup_database()
        self.load_ai_models()
    
    def setup_database(self):
        """Setup unified database with all fields"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drop old table if exists
            cursor.execute("DROP TABLE IF EXISTS corpus_items")
            
            # Create comprehensive table
            cursor.execute("""
                CREATE TABLE corpus_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    language TEXT,
                    content TEXT,
                    word_count INTEGER,
                    date_added TEXT,
                    status TEXT,
                    
                    -- Metadata
                    metadata_quality REAL DEFAULT 0,
                    author TEXT,
                    period TEXT,
                    genre TEXT,
                    
                    -- Annotation metrics
                    annotation_score REAL DEFAULT 0,
                    tokens_count INTEGER DEFAULT 0,
                    lemmas_count INTEGER DEFAULT 0,
                    pos_tags_count INTEGER DEFAULT 0,
                    dependencies_count INTEGER DEFAULT 0,
                    
                    -- AI model results
                    stanza_annotation TEXT,
                    spacy_annotation TEXT,
                    transformers_annotation TEXT,
                    
                    -- PROIEL treebank
                    proiel_xml TEXT,
                    proiel_statistics TEXT,
                    
                    -- Valency
                    valency_patterns TEXT,
                    verb_count INTEGER DEFAULT 0,
                    
                    -- Quality scores
                    tokenization_score REAL DEFAULT 0,
                    pos_tagging_score REAL DEFAULT 0,
                    lemmatization_score REAL DEFAULT 0,
                    parsing_score REAL DEFAULT 0
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✓ Database initialized with comprehensive structure")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def load_ai_models(self):
        """Load multiple free open-source AI models"""
        logger.info("\n" + "="*80)
        logger.info("LOADING FREE OPEN-SOURCE AI MODELS")
        logger.info("="*80)
        
        # 1. Stanza (Stanford NLP)
        try:
            import stanza
            logger.info("\n>>> Loading Stanza (Stanford NLP)...")
            
            for lang in ['grc', 'la', 'en']:
                try:
                    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', 
                                         verbose=False, download_method=None)
                    self.models_loaded[f'stanza_{lang}'] = nlp
                    logger.info(f"  ✓ Stanza {lang} loaded")
                except:
                    logger.warning(f"  ⚠ Stanza {lang} not available (will download on first use)")
            
        except Exception as e:
            logger.warning(f"  ⚠ Stanza not available: {e}")
        
        # 2. spaCy
        try:
            import spacy
            logger.info("\n>>> Loading spaCy...")
            
            spacy_models = {
                'en': 'en_core_web_sm',
                'grc': 'grc_proiel_sm',
                'la': 'la_core_web_sm'
            }
            
            for lang, model_name in spacy_models.items():
                try:
                    nlp = spacy.load(model_name)
                    self.models_loaded[f'spacy_{lang}'] = nlp
                    logger.info(f"  ✓ spaCy {lang} loaded")
                except:
                    logger.warning(f"  ⚠ spaCy {lang} not available")
            
        except Exception as e:
            logger.warning(f"  ⚠ spaCy not available: {e}")
        
        # 3. Hugging Face Transformers
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            logger.info("\n>>> Loading Hugging Face Transformers...")
            
            # Ancient Greek BERT
            try:
                self.models_loaded['transformers_grc'] = pipeline(
                    "token-classification",
                    model="pranaydeeps/Ancient-Greek-BERT",
                    aggregation_strategy="simple"
                )
                logger.info("  ✓ Ancient Greek BERT loaded")
            except:
                logger.warning("  ⚠ Ancient Greek BERT not available")
            
            # Latin BERT
            try:
                self.models_loaded['transformers_lat'] = pipeline(
                    "token-classification",
                    model="bowphs/LaBerta",
                    aggregation_strategy="simple"
                )
                logger.info("  ✓ Latin BERT (LaBerta) loaded")
            except:
                logger.warning("  ⚠ Latin BERT not available")
            
        except Exception as e:
            logger.warning(f"  ⚠ Transformers not available: {e}")
        
        # 4. NLTK
        try:
            import nltk
            logger.info("\n>>> Loading NLTK...")
            
            nltk.data.path.append(str(Path('Z:/models/nltk_data')))
            
            self.models_loaded['nltk'] = nltk
            logger.info("  ✓ NLTK loaded")
            
        except Exception as e:
            logger.warning(f"  ⚠ NLTK not available: {e}")
        
        logger.info("\n" + "="*80)
        logger.info(f"MODELS LOADED: {len(self.models_loaded)}")
        logger.info("="*80 + "\n")
    
    def collect_text(self, url: str, title: str, language: str) -> Optional[Dict]:
        """Collect text from URL"""
        try:
            logger.info("\n" + "█"*80)
            logger.info("COLLECTING TEXT")
            logger.info("█"*80)
            logger.info(f"Title:    {title}")
            logger.info(f"URL:      {url}")
            logger.info(f"Language: {language}")
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                word_count = len(content.split())
                
                logger.info(f"✓ Downloaded: {word_count:,} words")
                
                return {
                    'url': url,
                    'title': title,
                    'language': language,
                    'content': content,
                    'word_count': word_count,
                    'date_added': datetime.now().isoformat(),
                    'status': 'collected'
                }
            else:
                logger.error(f"✗ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"✗ Collection failed: {e}")
            self.stats['errors'] += 1
            return None
    
    def annotate_with_multi_ai(self, text: str, language: str) -> Dict:
        """Annotate text using multiple AI models"""
        logger.info("\n>>> MULTI-AI ANNOTATION")
        logger.info("-"*80)
        
        results = {
            'stanza': None,
            'spacy': None,
            'transformers': None,
            'nltk': None,
            'ensemble': {}
        }
        
        # Stanza annotation
        stanza_key = f'stanza_{language}'
        if stanza_key in self.models_loaded:
            try:
                logger.info("  Processing with Stanza...")
                doc = self.models_loaded[stanza_key](text)
                
                tokens = []
                for sent in doc.sentences:
                    for word in sent.words:
                        tokens.append({
                            'text': word.text,
                            'lemma': word.lemma,
                            'pos': word.upos,
                            'feats': word.feats,
                            'head': word.head,
                            'deprel': word.deprel
                        })
                
                results['stanza'] = {
                    'tokens': tokens,
                    'sentences': len(doc.sentences),
                    'token_count': len(tokens)
                }
                
                logger.info(f"    ✓ Stanza: {len(tokens)} tokens, {len(doc.sentences)} sentences")
                
            except Exception as e:
                logger.warning(f"    ⚠ Stanza failed: {e}")
        
        # spaCy annotation
        spacy_key = f'spacy_{language}'
        if spacy_key in self.models_loaded:
            try:
                logger.info("  Processing with spaCy...")
                doc = self.models_loaded[spacy_key](text[:1000000])  # Limit for memory
                
                tokens = []
                for token in doc:
                    tokens.append({
                        'text': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'tag': token.tag_,
                        'dep': token.dep_
                    })
                
                results['spacy'] = {
                    'tokens': tokens,
                    'token_count': len(tokens)
                }
                
                logger.info(f"    ✓ spaCy: {len(tokens)} tokens")
                
            except Exception as e:
                logger.warning(f"    ⚠ spaCy failed: {e}")
        
        # Transformers annotation
        trans_key = f'transformers_{language}'
        if trans_key in self.models_loaded:
            try:
                logger.info("  Processing with Transformers...")
                # Limit text for transformers
                sample = text[:512]
                result = self.models_loaded[trans_key](sample)
                
                results['transformers'] = {
                    'entities': result,
                    'entity_count': len(result)
                }
                
                logger.info(f"    ✓ Transformers: {len(result)} entities")
                
            except Exception as e:
                logger.warning(f"    ⚠ Transformers failed: {e}")
        
        # Create ensemble results
        if results['stanza']:
            results['ensemble'] = results['stanza']
            logger.info("  ✓ Using Stanza as primary annotation")
        elif results['spacy']:
            results['ensemble'] = results['spacy']
            logger.info("  ✓ Using spaCy as primary annotation")
        
        logger.info("-"*80)
        return results
    
    def generate_proiel(self, text: str, language: str, annotations: Dict) -> Optional[Dict]:
        """Generate PROIEL treebank from annotations"""
        logger.info("\n>>> GENERATING PROIEL TREEBANK")
        logger.info("-"*80)
        
        try:
            # Use ensemble annotations
            if not annotations.get('ensemble'):
                logger.warning("  ⚠ No annotations available for PROIEL")
                return None
            
            ensemble = annotations['ensemble']
            tokens = ensemble.get('tokens', [])
            
            if not tokens:
                logger.warning("  ⚠ No tokens for PROIEL")
                return None
            
            # Generate PROIEL XML
            xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            xml_lines.append('<proiel>')
            xml_lines.append(f'  <source language="{language}">')
            xml_lines.append('    <sentence id="1">')
            
            for idx, token in enumerate(tokens, 1):
                xml_lines.append(
                    f'      <token id="{idx}" form="{token.get("text", "")}" '
                    f'lemma="{token.get("lemma", "")}" postag="{token.get("pos", "")}" '
                    f'head-id="{token.get("head", 0)}" relation="{token.get("deprel", "")}" />'
                )
            
            xml_lines.append('    </sentence>')
            xml_lines.append('  </source>')
            xml_lines.append('</proiel>')
            
            proiel_xml = '\n'.join(xml_lines)
            
            # Extract valency patterns
            valency_patterns = []
            for token in tokens:
                if token.get('pos') == 'VERB':
                    valency_patterns.append({
                        'verb': token.get('text'),
                        'lemma': token.get('lemma'),
                        'pattern': token.get('deprel', '')
                    })
            
            result = {
                'proiel_xml': proiel_xml,
                'statistics': {
                    'tokens': len(tokens),
                    'sentences': ensemble.get('sentences', 1),
                    'verbs': len(valency_patterns)
                },
                'valency_patterns': valency_patterns
            }
            
            logger.info(f"  ✓ PROIEL generated")
            logger.info(f"    Tokens: {len(tokens)}")
            logger.info(f"    Verbs: {len(valency_patterns)}")
            logger.info("-"*80)
            
            return result
            
        except Exception as e:
            logger.error(f"  ✗ PROIEL generation failed: {e}")
            return None
    
    def save_to_database(self, text_data: Dict, annotations: Dict, proiel: Dict):
        """Save everything to database"""
        logger.info("\n>>> SAVING TO DATABASE")
        logger.info("-"*80)
        
        try:
            import json
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate metrics
            metadata_quality = 0
            if text_data.get('title'): metadata_quality += 25
            if text_data.get('language'): metadata_quality += 25
            if text_data.get('word_count', 0) > 0: metadata_quality += 25
            if text_data.get('date_added'): metadata_quality += 25
            
            annotation_score = 0
            tokens_count = 0
            lemmas_count = 0
            pos_tags_count = 0
            dependencies_count = 0
            
            if annotations.get('ensemble'):
                tokens = annotations['ensemble'].get('tokens', [])
                tokens_count = len(tokens)
                lemmas_count = sum(1 for t in tokens if t.get('lemma'))
                pos_tags_count = sum(1 for t in tokens if t.get('pos'))
                dependencies_count = sum(1 for t in tokens if t.get('deprel'))
                
                if tokens_count > 0:
                    annotation_score = (
                        (lemmas_count / tokens_count * 100) +
                        (pos_tags_count / tokens_count * 100) +
                        (dependencies_count / tokens_count * 100)
                    ) / 3
            
            verb_count = 0
            if proiel:
                verb_count = len(proiel.get('valency_patterns', []))
            
            # Insert
            cursor.execute("""
                INSERT OR REPLACE INTO corpus_items
                (url, title, language, content, word_count, date_added, status,
                 metadata_quality, annotation_score, tokens_count, lemmas_count,
                 pos_tags_count, dependencies_count, stanza_annotation, spacy_annotation,
                 transformers_annotation, proiel_xml, proiel_statistics, valency_patterns,
                 verb_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_data['url'],
                text_data['title'],
                text_data['language'],
                text_data['content'],
                text_data['word_count'],
                text_data['date_added'],
                'completed',
                metadata_quality,
                annotation_score,
                tokens_count,
                lemmas_count,
                pos_tags_count,
                dependencies_count,
                json.dumps(annotations.get('stanza')) if annotations.get('stanza') else None,
                json.dumps(annotations.get('spacy')) if annotations.get('spacy') else None,
                json.dumps(annotations.get('transformers')) if annotations.get('transformers') else None,
                proiel['proiel_xml'] if proiel else None,
                json.dumps(proiel['statistics']) if proiel else None,
                json.dumps(proiel['valency_patterns']) if proiel else None,
                verb_count
            ))
            
            conn.commit()
            text_id = cursor.lastrowid
            conn.close()
            
            logger.info("="*80)
            logger.info(f"✓ TEXT SAVED - ID: {text_id}")
            logger.info("="*80)
            logger.info(f"Title:              {text_data['title']}")
            logger.info(f"Language:           {text_data['language']}")
            logger.info(f"Words:              {text_data['word_count']:,}")
            logger.info(f"Metadata Quality:   {metadata_quality:.0f}%")
            logger.info(f"Annotation Score:   {annotation_score:.1f}%")
            logger.info(f"Tokens:             {tokens_count:,}")
            logger.info(f"Lemmas:             {lemmas_count:,}")
            logger.info(f"POS Tags:           {pos_tags_count:,}")
            logger.info(f"Dependencies:       {dependencies_count:,}")
            logger.info(f"Verbs:              {verb_count}")
            logger.info("="*80)
            
            self.stats['texts_collected'] += 1
            if annotations: self.stats['texts_annotated'] += 1
            if proiel: self.stats['treebanks_generated'] += 1
            
        except Exception as e:
            logger.error(f"✗ Database save failed: {e}")
            self.stats['errors'] += 1
    
    def process_text(self, url: str, title: str, language: str):
        """Complete pipeline: collect, annotate, generate PROIEL, save"""
        logger.info("\n\n" + "🌟"*40)
        logger.info("="*80)
        logger.info(f"PROCESSING: {title}")
        logger.info("="*80)
        
        # Step 1: Collect
        text_data = self.collect_text(url, title, language)
        if not text_data:
            return
        
        # Step 2: Annotate with multiple AIs
        annotations = self.annotate_with_multi_ai(text_data['content'], language)
        
        # Step 3: Generate PROIEL
        proiel = self.generate_proiel(text_data['content'], language, annotations)
        
        # Step 4: Save
        self.save_to_database(text_data, annotations, proiel)
        
        # Step 5: Print stats
        self.print_stats()
        
        logger.info("🌟"*40 + "\n\n")
    
    def print_stats(self):
        """Print session statistics"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = elapsed / 3600
        
        logger.info("\n" + "="*80)
        logger.info("SESSION STATISTICS")
        logger.info("="*80)
        logger.info(f"Texts Collected:        {self.stats['texts_collected']}")
        logger.info(f"Texts Annotated:        {self.stats['texts_annotated']}")
        logger.info(f"Treebanks Generated:    {self.stats['treebanks_generated']}")
        logger.info(f"Errors:                 {self.stats['errors']}")
        logger.info(f"Elapsed Time:           {elapsed/60:.1f} minutes")
        if hours > 0:
            logger.info(f"Collection Rate:        {self.stats['texts_collected']/hours:.1f} texts/hour")
        logger.info("="*80 + "\n")
    
    def run_collection_batch(self, batch_size: int = 5):
        """Run a batch of text collection and processing"""
        # Curated list of high-quality texts
        texts = [
            ("http://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice - Jane Austen", "en"),
            ("http://www.gutenberg.org/files/84/84-0.txt", "Frankenstein - Mary Shelley", "en"),
            ("http://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland - Lewis Carroll", "en"),
            ("http://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities - Charles Dickens", "en"),
            ("http://www.gutenberg.org/files/174/174-0.txt", "The Picture of Dorian Gray - Oscar Wilde", "en"),
            ("http://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes - Arthur Conan Doyle", "en"),
            ("http://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick - Herman Melville", "en"),
            ("http://www.gutenberg.org/files/1952/1952-0.txt", "The Yellow Wallpaper - Charlotte Perkins Gilman", "en"),
            ("http://www.gutenberg.org/files/1080/1080-0.txt", "A Modest Proposal - Jonathan Swift", "en"),
            ("http://www.gutenberg.org/files/2554/2554-0.txt", "Crime and Punishment - Fyodor Dostoevsky", "en"),
        ]
        
        for i, (url, title, lang) in enumerate(texts[:batch_size], 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"TEXT {i}/{min(batch_size, len(texts))}")
            logger.info(f"{'='*80}")
            
            self.process_text(url, title, lang)
            
            # Respectful delay
            if i < batch_size:
                time.sleep(3)
        
        logger.info("\n" + "="*80)
        logger.info("BATCH COMPLETE")
        logger.info("="*80)
        self.print_stats()


def main():
    """Main entry point"""
    print("="*80)
    print("UNIFIED ENHANCED CORPUS PLATFORM")
    print("="*80)
    print("Integrating:")
    print("  - Text Collection")
    print("  - Multi-AI Annotation (Stanza, spaCy, Transformers, NLTK)")
    print("  - PROIEL Treebank Generation")
    print("  - Valency Extraction")
    print("  - Enhanced Logging")
    print("="*80)
    print()
    
    platform = UnifiedEnhancedPlatform()
    
    # Run initial batch
    platform.run_collection_batch(batch_size=5)
    
    print("\n" + "="*80)
    print("PLATFORM READY")
    print("="*80)
    print()
    print("To continue collecting:")
    print("  platform.run_collection_batch(batch_size=10)")
    print()
    print("To process specific text:")
    print("  platform.process_text(url, title, language)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
