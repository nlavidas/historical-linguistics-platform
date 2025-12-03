"""
Master Pipeline - Orchestrates all corpus processing
Text Acquisition → Preprocessing → Annotation → Storage → Visualization
"""

import os
import sys
import json
import sqlite3
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all modules
from text_acquisition import (
    MasterTextCollector, RawText, TextMetadata,
    PerseusCollector, First1KGreekCollector, PROIELCollector, NewTestamentCollector
)
from preprocessing import (
    PreprocessingPipeline, GreekNormalizer, GreekTokenizer,
    GreekSentenceSplitter, GreekMorphAnalyzer, PreprocessedDatabase
)
from proiel_annotation import (
    PROIELToken, PROIELSentence, PROIELSource,
    AnnotationDatabase, PROIELMorphology
)
from semantic_roles import (
    SRLPredictor, SRLAnnotation, SRLSentence, SRLDatabase,
    GREEK_VERB_FRAMES
)
from valency_lexicon import (
    ValencyDatabase, GREEK_VALENCY_LEXICON
)
from visualization import (
    DependencyTreeSVG, InterlinearGloss, MorphologyTable,
    DiachronicCharts, StatsDashboard
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/corpus_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_CONFIG = {
    "data_dir": "/root/corpus_platform/data",
    "db_path": "/root/corpus_platform/data/greek_corpus.db",
    "proiel_db_path": "/root/corpus_platform/data/proiel_annotations.db",
    "srl_db_path": "/root/corpus_platform/data/srl_annotations.db",
    "valency_db_path": "/root/corpus_platform/data/valency_lexicon.db",
    "preprocessed_db_path": "/root/corpus_platform/data/preprocessed.db",
    "output_dir": "/root/corpus_platform/data/output",
    "cache_dir": "/root/corpus_platform/data/cache",
    "max_workers": 4,
    "batch_size": 100,
    "enable_srl": True,
    "enable_visualization": True,
    "auto_annotate": True
}

# =============================================================================
# PIPELINE STATISTICS
# =============================================================================

@dataclass
class PipelineStats:
    """Statistics for pipeline run"""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0
    
    # Collection stats
    texts_collected: int = 0
    characters_collected: int = 0
    
    # Preprocessing stats
    sentences_processed: int = 0
    tokens_processed: int = 0
    
    # Annotation stats
    sentences_annotated: int = 0
    predicates_found: int = 0
    
    # Errors
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# MASTER PIPELINE
# =============================================================================

class MasterPipeline:
    """Master pipeline orchestrating all processing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or PIPELINE_CONFIG
        
        # Create directories
        for dir_key in ['data_dir', 'output_dir', 'cache_dir']:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = MasterTextCollector(self.config['data_dir'])
        self.preprocessor = PreprocessingPipeline(self.config['data_dir'])
        self.srl_predictor = SRLPredictor()
        
        # Initialize databases
        self.preprocessed_db = PreprocessedDatabase(self.config['preprocessed_db_path'])
        self.proiel_db = AnnotationDatabase(self.config['proiel_db_path'])
        self.srl_db = SRLDatabase(self.config['srl_db_path'])
        self.valency_db = ValencyDatabase(self.config['valency_db_path'])
        
        # Statistics
        self.stats = PipelineStats()
        
    def run_full_pipeline(self) -> PipelineStats:
        """Run the complete pipeline"""
        self.stats = PipelineStats()
        self.stats.start_time = datetime.now().isoformat()
        
        logger.info("=" * 70)
        logger.info("STARTING MASTER PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Collect texts
            logger.info("\n" + "=" * 50)
            logger.info("STEP 1: TEXT COLLECTION")
            logger.info("=" * 50)
            collected = self._collect_texts()
            
            # Step 2: Preprocess texts
            logger.info("\n" + "=" * 50)
            logger.info("STEP 2: PREPROCESSING")
            logger.info("=" * 50)
            preprocessed = self._preprocess_texts(collected)
            
            # Step 3: Annotate (PROIEL style)
            logger.info("\n" + "=" * 50)
            logger.info("STEP 3: PROIEL ANNOTATION")
            logger.info("=" * 50)
            annotated = self._annotate_proiel(preprocessed)
            
            # Step 4: Semantic Role Labeling
            if self.config.get('enable_srl', True):
                logger.info("\n" + "=" * 50)
                logger.info("STEP 4: SEMANTIC ROLE LABELING")
                logger.info("=" * 50)
                self._annotate_srl(annotated)
            
            # Step 5: Generate visualizations
            if self.config.get('enable_visualization', True):
                logger.info("\n" + "=" * 50)
                logger.info("STEP 5: VISUALIZATION")
                logger.info("=" * 50)
                self._generate_visualizations()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats.errors.append(str(e))
        
        self.stats.end_time = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.stats.start_time)
        end = datetime.fromisoformat(self.stats.end_time)
        self.stats.duration_seconds = (end - start).total_seconds()
        
        # Log summary
        self._log_summary()
        
        return self.stats
    
    def _collect_texts(self) -> Dict:
        """Collect texts from all sources"""
        results = self.collector.collect_all_sources()
        
        # Update stats
        for source, texts in results.items():
            if isinstance(texts, list):
                self.stats.texts_collected += len(texts)
                for text in texts:
                    if hasattr(text, 'content'):
                        self.stats.characters_collected += len(text.content)
        
        logger.info(f"Collected {self.stats.texts_collected} texts, "
                   f"{self.stats.characters_collected:,} characters")
        
        return results
    
    def _preprocess_texts(self, collected: Dict) -> List[Dict]:
        """Preprocess all collected texts"""
        all_processed = []
        
        # Process Perseus texts
        for text in collected.get('perseus', []):
            try:
                processed = self._preprocess_single_text(text)
                if processed:
                    all_processed.append(processed)
            except Exception as e:
                logger.warning(f"Failed to preprocess {text.metadata.id}: {e}")
                self.stats.errors.append(f"Preprocess error: {text.metadata.id}")
        
        # Process First1KGreek texts
        for text in collected.get('first1k', []):
            try:
                processed = self._preprocess_single_text(text)
                if processed:
                    all_processed.append(processed)
            except Exception as e:
                logger.warning(f"Failed to preprocess {text.metadata.id}: {e}")
        
        # Process NT texts
        for text in collected.get('nt', []):
            try:
                processed = self._preprocess_single_text(text)
                if processed:
                    all_processed.append(processed)
            except Exception as e:
                logger.warning(f"Failed to preprocess {text.metadata.id}: {e}")
        
        logger.info(f"Preprocessed {len(all_processed)} texts, "
                   f"{self.stats.sentences_processed:,} sentences, "
                   f"{self.stats.tokens_processed:,} tokens")
        
        return all_processed
    
    def _preprocess_single_text(self, text: RawText) -> Optional[Dict]:
        """Preprocess a single text"""
        if not text.content or len(text.content) < 50:
            return None
        
        # Process through pipeline
        sentences = self.preprocessor.process_text(text.content, text.metadata.id)
        
        if not sentences:
            return None
        
        # Update stats
        self.stats.sentences_processed += len(sentences)
        self.stats.tokens_processed += sum(len(s['tokens']) for s in sentences)
        
        # Store in database
        metadata = {
            'title': text.metadata.title,
            'author': text.metadata.author,
            'period': text.metadata.period,
            'century': text.metadata.century,
            'genre': text.metadata.genre,
            'source': text.metadata.source,
            'language': text.metadata.language
        }
        
        self.preprocessed_db.store_processed_text(text.metadata.id, metadata, sentences)
        
        return {
            'id': text.metadata.id,
            'metadata': metadata,
            'sentences': sentences
        }
    
    def _annotate_proiel(self, preprocessed: List[Dict]) -> List[PROIELSource]:
        """Convert to PROIEL annotation format"""
        sources = []
        
        for text_data in preprocessed:
            try:
                source = self._create_proiel_source(text_data)
                if source:
                    sources.append(source)
                    self.proiel_db.store_source(source)
                    self.stats.sentences_annotated += len(source.sentences)
            except Exception as e:
                logger.warning(f"Failed to annotate {text_data['id']}: {e}")
        
        logger.info(f"Created {len(sources)} PROIEL sources, "
                   f"{self.stats.sentences_annotated:,} sentences")
        
        return sources
    
    def _create_proiel_source(self, text_data: Dict) -> Optional[PROIELSource]:
        """Create PROIEL source from preprocessed data"""
        metadata = text_data['metadata']
        
        source = PROIELSource(
            id=text_data['id'],
            title=metadata.get('title', ''),
            author=metadata.get('author', ''),
            language=metadata.get('language', 'grc'),
            period=metadata.get('period', ''),
            genre=metadata.get('genre', '')
        )
        
        for sent_data in text_data['sentences']:
            sentence = PROIELSentence(
                id=sent_data['id'],
                status='auto-annotated'
            )
            
            for token_data in sent_data['tokens']:
                # Map to PROIEL POS
                pos = self._map_to_proiel_pos(token_data.get('pos', 'X'))
                
                # Create morphology tag
                morph = token_data.get('morph', '_')
                if morph == '_':
                    morph = '----------'
                
                token = PROIELToken(
                    id=token_data.get('id', 0),
                    form=token_data.get('form', ''),
                    lemma=token_data.get('lemma', ''),
                    pos=pos,
                    morphology=morph,
                    head_id=token_data.get('head', 0),
                    relation=self._map_to_proiel_relation(token_data.get('relation', ''))
                )
                sentence.tokens.append(token)
            
            source.sentences.append(sentence)
        
        return source
    
    def _map_to_proiel_pos(self, ud_pos: str) -> str:
        """Map UD POS to PROIEL POS"""
        mapping = {
            'NOUN': 'Nb',
            'PROPN': 'Ne',
            'VERB': 'V-',
            'AUX': 'V-',
            'ADJ': 'A-',
            'ADV': 'Df',
            'PRON': 'Pp',
            'DET': 'S-',
            'ADP': 'R-',
            'CCONJ': 'C-',
            'SCONJ': 'G-',
            'PART': 'N-',
            'INTJ': 'I-',
            'NUM': 'Ma',
            'PUNCT': 'X-',
            'X': 'F-'
        }
        return mapping.get(ud_pos, 'F-')
    
    def _map_to_proiel_relation(self, ud_rel: str) -> str:
        """Map UD relation to PROIEL relation"""
        mapping = {
            'root': 'pred',
            'nsubj': 'sub',
            'obj': 'obj',
            'iobj': 'obl',
            'obl': 'obl',
            'advmod': 'adv',
            'amod': 'atr',
            'nmod': 'atr',
            'det': 'atr',
            'case': 'aux',
            'mark': 'aux',
            'cc': 'aux',
            'conj': 'part',
            'punct': 'aux',
            'xcomp': 'xobj',
            'ccomp': 'comp',
            'advcl': 'xadv',
            'acl': 'atr',
            'appos': 'apos',
            'vocative': 'voc',
            'cop': 'pid',
            'aux': 'aux',
            'expl': 'expl'
        }
        return mapping.get(ud_rel, 'narg')
    
    def _annotate_srl(self, sources: List[PROIELSource]):
        """Add semantic role annotations"""
        total_predicates = 0
        
        for source in sources:
            for sentence in source.sentences:
                # Convert to dict format for SRL predictor
                tokens = [t.to_dict() for t in sentence.tokens]
                
                # Predict SRL
                annotations = self.srl_predictor.predict(tokens)
                
                # Store annotations
                for ann in annotations:
                    self.srl_db.store_annotation(sentence.id, ann)
                    total_predicates += 1
        
        self.stats.predicates_found = total_predicates
        logger.info(f"Found {total_predicates:,} predicates with SRL")
    
    def _generate_visualizations(self):
        """Generate sample visualizations"""
        output_dir = Path(self.config['output_dir'])
        
        # Get statistics
        prep_stats = self.preprocessed_db.get_statistics()
        proiel_stats = self.proiel_db.get_statistics()
        srl_stats = self.srl_db.get_statistics()
        valency_stats = self.valency_db.get_statistics()
        
        # Generate dashboard HTML
        dashboard_html = self._generate_dashboard_html(
            prep_stats, proiel_stats, srl_stats, valency_stats
        )
        
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(dashboard_html, encoding='utf-8')
        
        logger.info(f"Generated dashboard: {dashboard_path}")
        
        # Generate period distribution chart
        if 'by_period' in prep_stats:
            period_data = {k: v.get('tokens', 0) for k, v in prep_stats['by_period'].items()}
            svg = DiachronicCharts.generate_period_distribution_svg(period_data)
            
            chart_path = output_dir / "period_distribution.svg"
            chart_path.write_text(svg, encoding='utf-8')
            logger.info(f"Generated chart: {chart_path}")
    
    def _generate_dashboard_html(self, prep_stats: Dict, proiel_stats: Dict,
                                  srl_stats: Dict, valency_stats: Dict) -> str:
        """Generate HTML dashboard"""
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Greek Corpus Platform - Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            color: white; 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .metric-label {{
            color: #64748b;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            padding: 15px;
            background: #f8fafc;
            border-radius: 10px;
        }}
        .stat-item strong {{ color: #1e293b; }}
        .stat-item span {{ color: #64748b; font-size: 0.9em; }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }}
        .period-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .period-label {{ width: 120px; font-size: 0.9em; }}
        .period-bar-fill {{
            height: 24px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.5s;
        }}
        .period-count {{ margin-left: 10px; font-size: 0.9em; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Greek Diachronic Corpus Platform</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{prep_stats.get('text_count', 0):,}</div>
                <div class="metric-label">Texts</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{prep_stats.get('sentence_count', 0):,}</div>
                <div class="metric-label">Sentences</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{prep_stats.get('token_count', 0):,}</div>
                <div class="metric-label">Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{prep_stats.get('lemma_count', 0):,}</div>
                <div class="metric-label">Unique Lemmas</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{valency_stats.get('entry_count', 0)}</div>
                <div class="metric-label">Valency Entries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{srl_stats.get('annotation_count', 0):,}</div>
                <div class="metric-label">SRL Annotations</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Distribution by Period</h2>
            <div class="period-bars">
'''
        
        # Add period bars
        by_period = prep_stats.get('by_period', {})
        max_tokens = max((v.get('tokens', 0) for v in by_period.values()), default=1)
        
        period_order = ['archaic', 'classical', 'hellenistic', 'koine', 
                       'late_antique', 'byzantine', 'medieval', 'early_modern', 'modern']
        
        for period in period_order:
            if period in by_period:
                data = by_period[period]
                tokens = data.get('tokens', 0)
                pct = (tokens / max_tokens * 100) if max_tokens > 0 else 0
                
                html += f'''
                <div class="period-bar">
                    <div class="period-label">{period.replace('_', ' ').title()}</div>
                    <div class="period-bar-fill" style="width: {pct}%;"></div>
                    <div class="period-count">{tokens:,} tokens</div>
                </div>
'''
        
        html += '''
            </div>
        </div>
        
        <div class="section">
            <h2>🏷️ POS Distribution</h2>
            <div class="stats-grid">
'''
        
        # Add POS stats
        pos_dist = prep_stats.get('pos_distribution', {})
        for pos, count in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:12]:
            html += f'''
                <div class="stat-item">
                    <strong>{pos}</strong>
                    <span>{count:,}</span>
                </div>
'''
        
        html += f'''
            </div>
        </div>
        
        <div class="section">
            <h2>📚 Valency Lexicon</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <strong>Verb Entries</strong>
                    <span>{valency_stats.get('entry_count', 0)}</span>
                </div>
                <div class="stat-item">
                    <strong>Valency Frames</strong>
                    <span>{valency_stats.get('frame_count', 0)}</span>
                </div>
                <div class="stat-item">
                    <strong>Unique Patterns</strong>
                    <span>{valency_stats.get('pattern_count', 0)}</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Greek Diachronic Corpus Platform - University of Athens</p>
        </div>
    </div>
</body>
</html>
'''
        
        return html
    
    def _log_summary(self):
        """Log pipeline summary"""
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {self.stats.duration_seconds:.1f} seconds")
        logger.info(f"Texts collected: {self.stats.texts_collected:,}")
        logger.info(f"Characters: {self.stats.characters_collected:,}")
        logger.info(f"Sentences processed: {self.stats.sentences_processed:,}")
        logger.info(f"Tokens processed: {self.stats.tokens_processed:,}")
        logger.info(f"Sentences annotated: {self.stats.sentences_annotated:,}")
        logger.info(f"Predicates found: {self.stats.predicates_found:,}")
        
        if self.stats.errors:
            logger.warning(f"Errors: {len(self.stats.errors)}")
            for err in self.stats.errors[:5]:
                logger.warning(f"  - {err}")
        
        logger.info("=" * 70)


# =============================================================================
# INCREMENTAL PIPELINE
# =============================================================================

class IncrementalPipeline(MasterPipeline):
    """Pipeline that only processes new/changed texts"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.processed_hashes = self._load_processed_hashes()
    
    def _load_processed_hashes(self) -> Dict[str, str]:
        """Load hashes of already processed texts"""
        hash_file = Path(self.config['data_dir']) / "processed_hashes.json"
        if hash_file.exists():
            return json.loads(hash_file.read_text())
        return {}
    
    def _save_processed_hashes(self):
        """Save hashes of processed texts"""
        hash_file = Path(self.config['data_dir']) / "processed_hashes.json"
        hash_file.write_text(json.dumps(self.processed_hashes))
    
    def _should_process(self, text: RawText) -> bool:
        """Check if text needs processing"""
        content_hash = hashlib.md5(text.content.encode()).hexdigest()
        
        if text.metadata.id in self.processed_hashes:
            if self.processed_hashes[text.metadata.id] == content_hash:
                return False
        
        self.processed_hashes[text.metadata.id] = content_hash
        return True
    
    def run_incremental(self) -> PipelineStats:
        """Run incremental pipeline"""
        self.stats = PipelineStats()
        self.stats.start_time = datetime.now().isoformat()
        
        logger.info("=" * 70)
        logger.info("STARTING INCREMENTAL PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Collect texts
            collected = self._collect_texts()
            
            # Filter to only new/changed texts
            filtered = self._filter_new_texts(collected)
            
            if not any(filtered.values()):
                logger.info("No new texts to process")
                return self.stats
            
            # Process only new texts
            preprocessed = self._preprocess_texts(filtered)
            annotated = self._annotate_proiel(preprocessed)
            
            if self.config.get('enable_srl', True):
                self._annotate_srl(annotated)
            
            # Save hashes
            self._save_processed_hashes()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats.errors.append(str(e))
        
        self.stats.end_time = datetime.now().isoformat()
        self._log_summary()
        
        return self.stats
    
    def _filter_new_texts(self, collected: Dict) -> Dict:
        """Filter to only new/changed texts"""
        filtered = {}
        
        for source, texts in collected.items():
            if isinstance(texts, list):
                new_texts = [t for t in texts if self._should_process(t)]
                filtered[source] = new_texts
                logger.info(f"{source}: {len(new_texts)}/{len(texts)} new texts")
            else:
                filtered[source] = texts
        
        return filtered


# =============================================================================
# SCHEDULED PIPELINE
# =============================================================================

class ScheduledPipeline:
    """Pipeline that runs on a schedule"""
    
    def __init__(self, config: Dict = None, interval_hours: int = 1):
        self.pipeline = IncrementalPipeline(config)
        self.interval_hours = interval_hours
        self.running = False
        self.thread = None
    
    def start(self):
        """Start scheduled pipeline"""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Scheduled pipeline started (interval: {self.interval_hours}h)")
    
    def stop(self):
        """Stop scheduled pipeline"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Scheduled pipeline stopped")
    
    def _run_loop(self):
        """Main run loop"""
        while self.running:
            try:
                self.pipeline.run_incremental()
            except Exception as e:
                logger.error(f"Scheduled run failed: {e}")
            
            # Sleep until next run
            for _ in range(self.interval_hours * 3600):
                if not self.running:
                    break
                time.sleep(1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Greek Corpus Master Pipeline')
    parser.add_argument('--mode', choices=['full', 'incremental', 'scheduled'],
                       default='incremental', help='Pipeline mode')
    parser.add_argument('--interval', type=int, default=1,
                       help='Interval in hours for scheduled mode')
    parser.add_argument('--data-dir', type=str, default='/root/corpus_platform/data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # Update config
    config = PIPELINE_CONFIG.copy()
    config['data_dir'] = args.data_dir
    
    if args.mode == 'full':
        pipeline = MasterPipeline(config)
        stats = pipeline.run_full_pipeline()
        print(f"\nPipeline completed in {stats.duration_seconds:.1f}s")
        print(f"Processed {stats.tokens_processed:,} tokens")
        
    elif args.mode == 'incremental':
        pipeline = IncrementalPipeline(config)
        stats = pipeline.run_incremental()
        print(f"\nIncremental pipeline completed in {stats.duration_seconds:.1f}s")
        
    elif args.mode == 'scheduled':
        pipeline = ScheduledPipeline(config, args.interval)
        pipeline.start()
        
        print(f"Scheduled pipeline running (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pipeline.stop()
            print("\nPipeline stopped")
