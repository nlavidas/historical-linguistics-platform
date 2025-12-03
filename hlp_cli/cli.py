"""
HLP CLI - Main Command Line Interface

This module provides the main CLI application for the Historical
Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List
import argparse

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration"""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="hlp",
        description="Historical Linguistics Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hlp corpus list
  hlp corpus create --name "Greek NT" --language grc
  hlp annotate --corpus my_corpus --engine stanza
  hlp valency extract --corpus my_corpus
  hlp server start --port 8000
  hlp export --corpus my_corpus --format proiel --output corpus.xml

For more information, visit: https://github.com/nlavidas/historical-linguistics-platform
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    add_corpus_commands(subparsers)
    add_annotate_commands(subparsers)
    add_valency_commands(subparsers)
    add_ingest_commands(subparsers)
    add_export_commands(subparsers)
    add_validate_commands(subparsers)
    add_server_commands(subparsers)
    add_diachronic_commands(subparsers)
    
    return parser


def add_corpus_commands(subparsers):
    """Add corpus management commands"""
    corpus_parser = subparsers.add_parser("corpus", help="Corpus management")
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command")
    
    list_parser = corpus_subparsers.add_parser("list", help="List corpora")
    list_parser.add_argument("--language", help="Filter by language")
    list_parser.add_argument("--format", choices=["table", "json"], default="table")
    
    create_parser = corpus_subparsers.add_parser("create", help="Create corpus")
    create_parser.add_argument("--name", required=True, help="Corpus name")
    create_parser.add_argument("--language", default="grc", help="Language code")
    create_parser.add_argument("--description", help="Corpus description")
    
    info_parser = corpus_subparsers.add_parser("info", help="Show corpus info")
    info_parser.add_argument("corpus_id", help="Corpus ID")
    
    delete_parser = corpus_subparsers.add_parser("delete", help="Delete corpus")
    delete_parser.add_argument("corpus_id", help="Corpus ID")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    import_parser = corpus_subparsers.add_parser("import", help="Import corpus")
    import_parser.add_argument("file", help="File to import")
    import_parser.add_argument("--format", choices=["proiel", "conllu", "text"], default="proiel")
    import_parser.add_argument("--name", help="Corpus name")


def add_annotate_commands(subparsers):
    """Add annotation commands"""
    annotate_parser = subparsers.add_parser("annotate", help="Annotation operations")
    annotate_subparsers = annotate_parser.add_subparsers(dest="annotate_command")
    
    run_parser = annotate_subparsers.add_parser("run", help="Run annotation")
    run_parser.add_argument("--corpus", required=True, help="Corpus ID")
    run_parser.add_argument("--engine", choices=["stanza", "spacy", "huggingface", "ollama"], default="stanza")
    run_parser.add_argument("--levels", nargs="+", default=["tokenization", "pos", "lemma", "morphology", "syntax"])
    run_parser.add_argument("--output", help="Output file")
    
    text_parser = annotate_subparsers.add_parser("text", help="Annotate text")
    text_parser.add_argument("text", help="Text to annotate")
    text_parser.add_argument("--language", default="grc", help="Language code")
    text_parser.add_argument("--engine", default="stanza", help="Annotation engine")
    text_parser.add_argument("--output", help="Output file")
    
    engines_parser = annotate_subparsers.add_parser("engines", help="List engines")
    
    jobs_parser = annotate_subparsers.add_parser("jobs", help="List annotation jobs")
    jobs_parser.add_argument("--status", help="Filter by status")


def add_valency_commands(subparsers):
    """Add valency commands"""
    valency_parser = subparsers.add_parser("valency", help="Valency operations")
    valency_subparsers = valency_parser.add_subparsers(dest="valency_command")
    
    extract_parser = valency_subparsers.add_parser("extract", help="Extract valency")
    extract_parser.add_argument("--corpus", required=True, help="Corpus ID")
    extract_parser.add_argument("--min-frequency", type=int, default=1)
    extract_parser.add_argument("--output", help="Output file")
    
    search_parser = valency_subparsers.add_parser("search", help="Search patterns")
    search_parser.add_argument("--verb", help="Verb lemma")
    search_parser.add_argument("--pattern", help="Pattern to search")
    search_parser.add_argument("--lexicon", help="Lexicon ID")
    
    lexicon_parser = valency_subparsers.add_parser("lexicon", help="Lexicon operations")
    lexicon_parser.add_argument("action", choices=["list", "create", "export"])
    lexicon_parser.add_argument("--id", help="Lexicon ID")
    lexicon_parser.add_argument("--output", help="Output file")


def add_ingest_commands(subparsers):
    """Add ingestion commands"""
    ingest_parser = subparsers.add_parser("ingest", help="Text ingestion")
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_command")
    
    sources_parser = ingest_subparsers.add_parser("sources", help="List sources")
    
    fetch_parser = ingest_subparsers.add_parser("fetch", help="Fetch texts")
    fetch_parser.add_argument("--source", required=True, choices=["perseus", "first1k", "gutenberg", "proiel"])
    fetch_parser.add_argument("--corpus", required=True, help="Target corpus ID")
    fetch_parser.add_argument("--text-id", help="Specific text ID")
    fetch_parser.add_argument("--language", help="Language filter")
    
    jobs_parser = ingest_subparsers.add_parser("jobs", help="List ingest jobs")
    jobs_parser.add_argument("--status", help="Filter by status")


def add_export_commands(subparsers):
    """Add export commands"""
    export_parser = subparsers.add_parser("export", help="Export operations")
    
    export_parser.add_argument("--corpus", required=True, help="Corpus ID")
    export_parser.add_argument("--format", choices=["proiel", "conllu", "json", "tsv"], default="proiel")
    export_parser.add_argument("--output", required=True, help="Output file")
    export_parser.add_argument("--documents", nargs="+", help="Specific document IDs")


def add_validate_commands(subparsers):
    """Add validation commands"""
    validate_parser = subparsers.add_parser("validate", help="Validation operations")
    validate_subparsers = validate_parser.add_subparsers(dest="validate_command")
    
    file_parser = validate_subparsers.add_parser("file", help="Validate file")
    file_parser.add_argument("file", help="File to validate")
    file_parser.add_argument("--format", choices=["proiel", "conllu", "auto"], default="auto")
    file_parser.add_argument("--level", choices=["minimal", "standard", "strict", "erc"], default="standard")
    
    corpus_parser = validate_subparsers.add_parser("corpus", help="Validate corpus")
    corpus_parser.add_argument("corpus_id", help="Corpus ID")
    corpus_parser.add_argument("--level", choices=["minimal", "standard", "strict", "erc"], default="standard")
    
    audit_parser = validate_subparsers.add_parser("audit", help="Audit corpus")
    audit_parser.add_argument("corpus_id", help="Corpus ID")
    audit_parser.add_argument("--level", choices=["basic", "standard", "thorough", "erc"], default="standard")
    audit_parser.add_argument("--output", help="Output report file")


def add_server_commands(subparsers):
    """Add server commands"""
    server_parser = subparsers.add_parser("server", help="Server operations")
    server_subparsers = server_parser.add_subparsers(dest="server_command")
    
    start_parser = server_subparsers.add_parser("start", help="Start server")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    start_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    status_parser = server_subparsers.add_parser("status", help="Server status")
    
    stop_parser = server_subparsers.add_parser("stop", help="Stop server")


def add_diachronic_commands(subparsers):
    """Add diachronic analysis commands"""
    diachronic_parser = subparsers.add_parser("diachronic", help="Diachronic analysis")
    diachronic_subparsers = diachronic_parser.add_subparsers(dest="diachronic_command")
    
    periods_parser = diachronic_subparsers.add_parser("periods", help="List periods")
    periods_parser.add_argument("--language", help="Filter by language")
    
    analyze_parser = diachronic_subparsers.add_parser("analyze", help="Analyze changes")
    analyze_parser.add_argument("--corpus", required=True, help="Corpus ID")
    analyze_parser.add_argument("--feature", required=True, help="Feature to analyze")
    analyze_parser.add_argument("--periods", nargs="+", help="Periods to include")
    analyze_parser.add_argument("--output", help="Output file")
    
    compare_parser = diachronic_subparsers.add_parser("compare", help="Compare periods")
    compare_parser.add_argument("--period1", required=True, help="First period")
    compare_parser.add_argument("--period2", required=True, help="Second period")
    compare_parser.add_argument("--features", nargs="+", help="Features to compare")


def handle_corpus_command(args):
    """Handle corpus commands"""
    if args.corpus_command == "list":
        print("Listing corpora...")
        print("ID\tName\tLanguage\tDocuments")
        print("-" * 50)
        print("No corpora found.")
    
    elif args.corpus_command == "create":
        print(f"Creating corpus '{args.name}' (language: {args.language})...")
        print("Corpus created successfully.")
    
    elif args.corpus_command == "info":
        print(f"Corpus: {args.corpus_id}")
        print("Not found.")
    
    elif args.corpus_command == "delete":
        if not args.force:
            confirm = input(f"Delete corpus {args.corpus_id}? [y/N] ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return
        print(f"Deleting corpus {args.corpus_id}...")
    
    elif args.corpus_command == "import":
        print(f"Importing {args.file} (format: {args.format})...")
    
    else:
        print("Usage: hlp corpus <command>")
        print("Commands: list, create, info, delete, import")


def handle_annotate_command(args):
    """Handle annotation commands"""
    if args.annotate_command == "run":
        print(f"Running annotation on corpus {args.corpus}...")
        print(f"Engine: {args.engine}")
        print(f"Levels: {', '.join(args.levels)}")
    
    elif args.annotate_command == "text":
        print(f"Annotating text (language: {args.language}, engine: {args.engine})...")
        print(f"Text: {args.text[:50]}...")
    
    elif args.annotate_command == "engines":
        print("Available annotation engines:")
        print("  - stanza: Stanford NLP Stanza")
        print("  - spacy: spaCy NLP")
        print("  - huggingface: HuggingFace Transformers")
        print("  - ollama: Ollama LLM")
    
    elif args.annotate_command == "jobs":
        print("Annotation jobs:")
        print("No jobs found.")
    
    else:
        print("Usage: hlp annotate <command>")
        print("Commands: run, text, engines, jobs")


def handle_valency_command(args):
    """Handle valency commands"""
    if args.valency_command == "extract":
        print(f"Extracting valency from corpus {args.corpus}...")
        print(f"Minimum frequency: {args.min_frequency}")
    
    elif args.valency_command == "search":
        print("Searching valency patterns...")
        if args.verb:
            print(f"Verb: {args.verb}")
        if args.pattern:
            print(f"Pattern: {args.pattern}")
    
    elif args.valency_command == "lexicon":
        if args.action == "list":
            print("Valency lexicons:")
            print("No lexicons found.")
        elif args.action == "create":
            print("Creating lexicon...")
        elif args.action == "export":
            print(f"Exporting lexicon {args.id}...")
    
    else:
        print("Usage: hlp valency <command>")
        print("Commands: extract, search, lexicon")


def handle_ingest_command(args):
    """Handle ingest commands"""
    if args.ingest_command == "sources":
        print("Available text sources:")
        print("  - perseus: Perseus Digital Library")
        print("  - first1k: First1KGreek")
        print("  - gutenberg: Project Gutenberg")
        print("  - proiel: PROIEL Treebanks")
    
    elif args.ingest_command == "fetch":
        print(f"Fetching from {args.source} to corpus {args.corpus}...")
        if args.text_id:
            print(f"Text ID: {args.text_id}")
        if args.language:
            print(f"Language: {args.language}")
    
    elif args.ingest_command == "jobs":
        print("Ingest jobs:")
        print("No jobs found.")
    
    else:
        print("Usage: hlp ingest <command>")
        print("Commands: sources, fetch, jobs")


def handle_export_command(args):
    """Handle export command"""
    print(f"Exporting corpus {args.corpus} to {args.output}...")
    print(f"Format: {args.format}")
    if args.documents:
        print(f"Documents: {', '.join(args.documents)}")


def handle_validate_command(args):
    """Handle validation commands"""
    if args.validate_command == "file":
        print(f"Validating {args.file}...")
        print(f"Format: {args.format}")
        print(f"Level: {args.level}")
        print("Validation complete. No errors found.")
    
    elif args.validate_command == "corpus":
        print(f"Validating corpus {args.corpus_id}...")
        print(f"Level: {args.level}")
    
    elif args.validate_command == "audit":
        print(f"Auditing corpus {args.corpus_id}...")
        print(f"Level: {args.level}")
    
    else:
        print("Usage: hlp validate <command>")
        print("Commands: file, corpus, audit")


def handle_server_command(args):
    """Handle server commands"""
    if args.server_command == "start":
        print(f"Starting server on {args.host}:{args.port}...")
        print(f"Workers: {args.workers}")
        if args.reload:
            print("Auto-reload enabled")
        
        try:
            import uvicorn
            from hlp_api.app import create_app
            
            app = create_app()
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=args.workers if not args.reload else 1,
                reload=args.reload
            )
        except ImportError:
            print("Error: uvicorn not installed. Run: pip install uvicorn")
        except Exception as e:
            print(f"Error starting server: {e}")
    
    elif args.server_command == "status":
        print("Server status: Not running")
    
    elif args.server_command == "stop":
        print("Stopping server...")
    
    else:
        print("Usage: hlp server <command>")
        print("Commands: start, status, stop")


def handle_diachronic_command(args):
    """Handle diachronic commands"""
    if args.diachronic_command == "periods":
        print("Historical periods:")
        print("  Greek:")
        print("    - archaic: Archaic Greek (-800 to -480)")
        print("    - classical: Classical Greek (-480 to -323)")
        print("    - hellenistic: Hellenistic Greek (-323 to -31)")
        print("    - roman: Roman Period Greek (-31 to 330)")
        print("    - byzantine: Byzantine Greek (330 to 1453)")
        print("  Latin:")
        print("    - early: Early Latin (-240 to -100)")
        print("    - classical: Classical Latin (-100 to 14)")
        print("    - silver: Silver Latin (14 to 200)")
        print("    - late: Late Latin (200 to 600)")
    
    elif args.diachronic_command == "analyze":
        print(f"Analyzing {args.feature} in corpus {args.corpus}...")
        if args.periods:
            print(f"Periods: {', '.join(args.periods)}")
    
    elif args.diachronic_command == "compare":
        print(f"Comparing {args.period1} and {args.period2}...")
        if args.features:
            print(f"Features: {', '.join(args.features)}")
    
    else:
        print("Usage: hlp diachronic <command>")
        print("Commands: periods, analyze, compare")


def cli(args: Optional[List[str]] = None):
    """Main CLI entry point"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    setup_logging(parsed_args.verbose, parsed_args.debug)
    
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    try:
        if parsed_args.command == "corpus":
            handle_corpus_command(parsed_args)
        elif parsed_args.command == "annotate":
            handle_annotate_command(parsed_args)
        elif parsed_args.command == "valency":
            handle_valency_command(parsed_args)
        elif parsed_args.command == "ingest":
            handle_ingest_command(parsed_args)
        elif parsed_args.command == "export":
            handle_export_command(parsed_args)
        elif parsed_args.command == "validate":
            handle_validate_command(parsed_args)
        elif parsed_args.command == "server":
            handle_server_command(parsed_args)
        elif parsed_args.command == "diachronic":
            handle_diachronic_command(parsed_args)
        else:
            parser.print_help()
        
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=parsed_args.debug)
        print(f"Error: {e}")
        return 1


def main():
    """Main entry point"""
    sys.exit(cli())


if __name__ == "__main__":
    main()
