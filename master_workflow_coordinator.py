#!/usr/bin/env python3
"""
MASTER HISTORICAL LINGUISTICS WORKFLOW COORDINATOR
===============================================
Complete cost-effective research pipeline using community-driven AIs
Optimized for low-cost servers and storage

WORKFLOW STAGES:
1. Text Collection → Free/public data sources
2. Preprocessing → Community-driven NLP tools
3. Parsing → Community-driven parsers
4. Annotation → Community-driven annotators
5. Valency Analysis → Verb argument structures
6. Diachronic Comparison → Historical evolution

All using FREE tools: NLTK, spaCy, Stanza, Trankit, UDPipe, Polyglot, TextBlob
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import time

# Import all pipeline components
try:
    from cost_effective_text_collection import CostEffectiveTextCollector
    from cost_effective_preprocessing import CostEffectivePreprocessor
    from cost_effective_parsing import CostEffectiveParser
    from cost_effective_annotation import CostEffectiveAnnotator
    from cost_effective_valency import CostEffectiveValencyAnalyzer
    from cost_effective_diachronic import DiachronicComparativeLexica
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all cost_effective_*.py files are in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_workflow.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MasterHistoricalLinguisticsWorkflow:
    """
    Master coordinator for the complete historical linguistics research pipeline
    """

    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.pipeline_components = {}

        # Initialize all pipeline components
        self._init_components()

    def _load_config(self, config_file: str = None) -> dict:
        """Load configuration from file or use defaults"""

        default_config = {
            'database_path': 'corpus_efficient.db',
            'batch_size': 50,
            'gpu_enabled': False,
            'max_collection_pages': 10,
            'processing_limit': None,
            'output_base_dir': 'research_output',
            'collection_sources': ['perseus', 'gutenberg', 'wikimedia'],
            'target_languages': ['grc', 'la', 'en', 'de', 'fr']
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")

        return default_config

    def _init_components(self):
        """Initialize all pipeline components"""

        logger.info("Initializing pipeline components...")

        try:
            self.pipeline_components['collector'] = CostEffectiveTextCollector(
                db_path=self.config['database_path']
            )
            logger.info("✓ Text collector initialized")
        except Exception as e:
            logger.error(f"✗ Text collector initialization failed: {e}")

        try:
            self.pipeline_components['preprocessor'] = CostEffectivePreprocessor(
                db_path=self.config['database_path'],
                batch_size=self.config['batch_size']
            )
            logger.info("✓ Preprocessor initialized")
        except Exception as e:
            logger.error(f"✗ Preprocessor initialization failed: {e}")

        try:
            self.pipeline_components['parser'] = CostEffectiveParser(
                db_path=self.config['database_path'],
                batch_size=self.config['batch_size']
            )
            logger.info("✓ Parser initialized")
        except Exception as e:
            logger.error(f"✗ Parser initialization failed: {e}")

        try:
            self.pipeline_components['annotator'] = CostEffectiveAnnotator(
                db_path=self.config['database_path'],
                batch_size=self.config['batch_size']
            )
            logger.info("✓ Annotator initialized")
        except Exception as e:
            logger.error(f"✗ Annotator initialization failed: {e}")

        try:
            self.pipeline_components['valency_analyzer'] = CostEffectiveValencyAnalyzer(
                db_path=self.config['database_path'],
                batch_size=self.config['batch_size']
            )
            logger.info("✓ Valency analyzer initialized")
        except Exception as e:
            logger.error(f"✗ Valency analyzer initialization failed: {e}")

        try:
            self.pipeline_components['diachronic_analyzer'] = DiachronicComparativeLexica(
                db_path=self.config['database_path'],
                batch_size=self.config['batch_size']
            )
            logger.info("✓ Diachronic analyzer initialized")
        except Exception as e:
            logger.error(f"✗ Diachronic analyzer initialization failed: {e}")

    def run_full_pipeline(self, stages: List[str] = None) -> dict:
        """
        Run the complete research pipeline
        """

        if stages is None:
            stages = ['collection', 'preprocessing', 'parsing', 'annotation', 'valency', 'diachronic']

        results = {
            'pipeline_start': datetime.now().isoformat(),
            'stages_completed': [],
            'stage_results': {},
            'errors': [],
            'total_runtime': 0
        }

        start_time = time.time()

        logger.info("Starting complete historical linguistics research pipeline")
        logger.info(f"Stages to run: {', '.join(stages)}")

        # Create output directory
        output_base = Path(self.config['output_base_dir'])
        output_base.mkdir(exist_ok=True)

        # Stage 1: Text Collection
        if 'collection' in stages:
            try:
                logger.info("STAGE 1: Text Collection")
                collector = self.pipeline_components['collector']

                # Add collection sources
                for source in self.config['collection_sources']:
                    if source == 'perseus':
                        collector.add_source("https://www.perseus.tufts.edu/hopper/", "Perseus Digital Library", "academic")
                    elif source == 'gutenberg':
                        collector.add_source("https://www.gutenberg.org/", "Project Gutenberg", "public_domain")
                    elif source == 'wikimedia':
                        collector.add_source("https://en.wikisource.org/", "Wikisource", "wiki")

                # Run collection
                collection_results = collector.run_collection_cycle(
                    sources=self.config['collection_sources']
                )

                results['stage_results']['collection'] = collection_results
                results['stages_completed'].append('collection')

                logger.info(f"✓ Collection completed: {collection_results['total_items']} texts collected")

            except Exception as e:
                error_msg = f"Collection stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Stage 2: Preprocessing
        if 'preprocessing' in stages:
            try:
                logger.info("STAGE 2: Text Preprocessing")
                preprocessor = self.pipeline_components['preprocessor']

                preprocessing_results = preprocessor.run_preprocessing_pipeline(
                    limit=self.config['processing_limit'],
                    output_dir=str(output_base / "preprocessed")
                )

                results['stage_results']['preprocessing'] = preprocessing_results
                results['stages_completed'].append('preprocessing')

                logger.info(f"✓ Preprocessing completed: {preprocessing_results} texts processed")

            except Exception as e:
                error_msg = f"Preprocessing stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Stage 3: Parsing
        if 'parsing' in stages:
            try:
                logger.info("STAGE 3: Linguistic Parsing")
                parser = self.pipeline_components['parser']

                parsing_results = parser.run_parsing_pipeline(
                    limit=self.config['processing_limit'],
                    output_dir=str(output_base / "parsed")
                )

                results['stage_results']['parsing'] = parsing_results
                results['stages_completed'].append('parsing')

                logger.info(f"✓ Parsing completed: {parsing_results} texts parsed")

            except Exception as e:
                error_msg = f"Parsing stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Stage 4: Annotation
        if 'annotation' in stages:
            try:
                logger.info("STAGE 4: Semantic Annotation")
                annotator = self.pipeline_components['annotator']

                annotation_results = annotator.run_annotation_pipeline(
                    limit=self.config['processing_limit'],
                    output_dir=str(output_base / "annotated")
                )

                results['stage_results']['annotation'] = annotation_results
                results['stages_completed'].append('annotation')

                logger.info(f"✓ Annotation completed: {annotation_results} texts annotated")

            except Exception as e:
                error_msg = f"Annotation stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Stage 5: Valency Analysis
        if 'valency' in stages:
            try:
                logger.info("STAGE 5: Valency Analysis")
                valency_analyzer = self.pipeline_components['valency_analyzer']

                valency_results = valency_analyzer.run_valency_analysis_pipeline(
                    output_dir=str(output_base / "valency_reports")
                )

                results['stage_results']['valency'] = valency_results
                results['stages_completed'].append('valency')

                logger.info(f"✓ Valency analysis completed: {valency_results['total_patterns']} patterns analyzed")

            except Exception as e:
                error_msg = f"Valency analysis stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Stage 6: Diachronic Analysis
        if 'diachronic' in stages:
            try:
                logger.info("STAGE 6: Diachronic Comparative Analysis")
                diachronic_analyzer = self.pipeline_components['diachronic_analyzer']

                diachronic_results = diachronic_analyzer.run_diachronic_analysis_pipeline(
                    output_dir=str(output_base / "diachronic_analysis")
                )

                if diachronic_results:
                    results['stage_results']['diachronic'] = diachronic_results
                    results['stages_completed'].append('diachronic')

                    logger.info(f"✓ Diachronic analysis completed: {diachronic_results['languages_analyzed']} languages analyzed")
                else:
                    logger.warning("Diachronic analysis skipped - no valency data available")

            except Exception as e:
                error_msg = f"Diachronic analysis stage failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Finalize results
        results['pipeline_end'] = datetime.now().isoformat()
        results['total_runtime'] = time.time() - start_time
        results['success_rate'] = len(results['stages_completed']) / len(stages)

        # Save results
        results_file = output_base / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Pipeline completed in {results['total_runtime']:.1f} seconds")
        logger.info(f"Success rate: {results['success_rate']:.1%}")
        logger.info(f"Results saved to: {results_file}")

        return results

    def generate_final_report(self, results: dict) -> str:
        """Generate comprehensive final report"""

        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("HISTORICAL LINGUISTICS RESEARCH PIPELINE - FINAL REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Pipeline overview
        report_lines.append("PIPELINE OVERVIEW")
        report_lines.append("-" * 18)
        report_lines.append(f"Total runtime: {results['total_runtime']:.1f} seconds")
        report_lines.append(f"Stages completed: {len(results['stages_completed'])}/{len(results.get('stage_results', {}))}")
        report_lines.append(f"Success rate: {results.get('success_rate', 0):.1%}")
        report_lines.append("")

        # Stage-by-stage results
        report_lines.append("STAGE RESULTS")
        report_lines.append("-" * 13)

        for stage, stage_results in results.get('stage_results', {}).items():
            report_lines.append(f"\n{stage.upper()} STAGE:")
            if isinstance(stage_results, dict):
                for key, value in stage_results.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  {key}: {value}")
                    elif isinstance(value, str) and len(value) < 100:
                        report_lines.append(f"  {key}: {value}")
            else:
                report_lines.append(f"  Result: {stage_results}")

        # Errors
        if results.get('errors'):
            report_lines.append("\n\nERRORS ENCOUNTERED")
            report_lines.append("-" * 18)
            for error in results['errors']:
                report_lines.append(f"• {error}")

        # Cost-effectiveness analysis
        report_lines.append("\n\nCOST-EFFECTIVENESS ANALYSIS")
        report_lines.append("-" * 26)
        report_lines.append("This pipeline uses exclusively FREE, community-driven tools:")
        report_lines.append("• NLTK - Natural Language Toolkit")
        report_lines.append("• spaCy - Industrial-strength NLP (free models)")
        report_lines.append("• Stanza - Stanford neural parsing")
        report_lines.append("• Trankit - Multilingual neural pipeline")
        report_lines.append("• UDPipe - Neural parsing")
        report_lines.append("• Polyglot - Multilingual semantic analysis")
        report_lines.append("• TextBlob - Simple but effective NLP")
        report_lines.append("")
        report_lines.append("STORAGE OPTIMIZATION:")
        report_lines.append("• Gzip compression for all text data")
        report_lines.append("• Efficient SQLite database schema")
        report_lines.append("• Batch processing to manage memory")
        report_lines.append("• Deduplication and quality filtering")

        # Research implications
        report_lines.append("\n\nRESEARCH IMPLICATIONS")
        report_lines.append("-" * 21)
        report_lines.append("This cost-effective pipeline enables:")
        report_lines.append("• Large-scale historical corpus analysis")
        report_lines.append("• Cross-linguistic valency comparisons")
        report_lines.append("• Diachronic evolution studies")
        report_lines.append("• Language contact research")
        report_lines.append("• Grammaticalization analysis")
        report_lines.append("")
        report_lines.append("All using commodity hardware and free software!")

        report_lines.append("\n" + "=" * 100)

        return "\n".join(report_lines)

    def get_system_status(self) -> dict:
        """Get comprehensive system status"""

        status = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_components': {},
            'available_tools': {},
            'system_resources': {},
            'configuration': self.config
        }

        # Component status
        for name, component in self.pipeline_components.items():
            try:
                if hasattr(component, 'get_corpus_stats'):
                    status['pipeline_components'][name] = component.get_corpus_stats()
                elif hasattr(component, 'get_preprocessing_stats'):
                    status['pipeline_components'][name] = component.get_preprocessing_stats()
                elif hasattr(component, 'get_parsing_stats'):
                    status['pipeline_components'][name] = component.get_parsing_stats()
                elif hasattr(component, 'get_annotation_stats'):
                    status['pipeline_components'][name] = component.get_annotation_stats()
                elif hasattr(component, 'get_valency_stats'):
                    status['pipeline_components'][name] = component.get_valency_stats()
                elif hasattr(component, 'get_diachronic_stats'):
                    status['pipeline_components'][name] = component.get_diachronic_stats()
                else:
                    status['pipeline_components'][name] = {'status': 'available'}
            except Exception as e:
                status['pipeline_components'][name] = {'status': 'error', 'error': str(e)}

        # Available tools summary
        for component_name, component_data in status['pipeline_components'].items():
            if 'available_tools' in component_data:
                status['available_tools'][component_name] = component_data['available_tools']

        return status

def main():
    """Main workflow coordinator"""

    parser = argparse.ArgumentParser(description="Master Historical Linguistics Workflow Coordinator")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--stages', '-s', nargs='+',
                       choices=['collection', 'preprocessing', 'parsing', 'annotation', 'valency', 'diachronic'],
                       default=['collection', 'preprocessing', 'parsing', 'annotation', 'valency', 'diachronic'],
                       help='Pipeline stages to run')
    parser.add_argument('--status', action='store_true', help='Show system status only')
    parser.add_argument('--limit', '-l', type=int, help='Processing limit for testing')

    args = parser.parse_args()

    print("Master Historical Linguistics Workflow Coordinator")
    print("=" * 55)
    print("Cost-effective research pipeline using community-driven AIs")
    print("NKUA Historical Linguistics Platform - HFRI Funded")
    print()

    # Initialize master workflow
    workflow = MasterHistoricalLinguisticsWorkflow(args.config)

    if args.limit:
        workflow.config['processing_limit'] = args.limit

    if args.status:
        # Show system status
        print("System Status:")
        print("-" * 13)
        status = workflow.get_system_status()

        print(f"Available pipeline components: {len(status['pipeline_components'])}")
        for component, data in status['pipeline_components'].items():
            status_indicator = "✓" if 'error' not in str(data).lower() else "✗"
            print(f"  {status_indicator} {component}")

        print(f"\nAvailable AI tools: {len(status['available_tools'])}")
        for component, tools in status['available_tools'].items():
            if isinstance(tools, dict):
                print(f"  {component}: {len(tools)} tools")

        return

    # Run full pipeline
    results = workflow.run_full_pipeline(args.stages)

    # Generate and display final report
    final_report = workflow.generate_final_report(results)
    print("\n" + final_report)

    # Save final report
    output_base = Path(workflow.config['output_base_dir'])
    output_base.mkdir(exist_ok=True)
    report_file = output_base / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(final_report)

    print(f"\nFinal report saved to: {report_file}")

if __name__ == "__main__":
    main()
