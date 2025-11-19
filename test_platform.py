"""
Test script for Unified Corpus Platform
Validates all components and runs sample pipeline
"""

import asyncio
import sys
import sqlite3
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import aiohttp
        print("  ✓ aiohttp")
    except ImportError:
        print("  ✗ aiohttp - Install: pip install aiohttp")
        return False
    
    try:
        import stanza
        print("  ✓ stanza")
    except ImportError:
        print("  ✗ stanza - Install: pip install stanza")
        return False
    
    try:
        from unified_corpus_platform import UnifiedCorpusPlatform
        print("  ✓ unified_corpus_platform")
    except ImportError as e:
        print(f"  ✗ unified_corpus_platform - Error: {e}")
        return False
    
    return True

def test_database():
    """Test database creation"""
    print("\nTesting database...")
    try:
        from unified_corpus_platform import UnifiedCorpusDatabase
        db = UnifiedCorpusDatabase("test_corpus.db")
        
        # Check tables exist
        conn = sqlite3.connect("test_corpus.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['corpus_items', 'annotations', 'pipeline_status', 
                          'sources', 'statistics']
        
        for table in expected_tables:
            if table in tables:
                print(f"  ✓ Table: {table}")
            else:
                print(f"  ✗ Table missing: {table}")
                return False
        
        # Clean up
        Path("test_corpus.db").unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False

async def test_scraper():
    """Test scraper component"""
    print("\nTesting scraper...")
    try:
        from unified_corpus_platform import UnifiedCorpusDatabase, AutomaticScraper
        
        db = UnifiedCorpusDatabase("test_corpus.db")
        scraper = AutomaticScraper(db)
        
        # Add test URL (simple text file)
        test_url = "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133"
        db.add_item(test_url, "perseus", language="grc", priority=10)
        
        await scraper.init_session()
        print("  ✓ Scraper initialized")
        await scraper.close_session()
        
        # Clean up
        Path("test_corpus.db").unlink(missing_ok=True)
        import shutil
        shutil.rmtree("data", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Scraper error: {e}")
        return False

async def test_full_pipeline():
    """Test complete pipeline with sample URL"""
    print("\nTesting full pipeline...")
    try:
        from unified_corpus_platform import UnifiedCorpusPlatform
        
        # Create platform
        platform = UnifiedCorpusPlatform("test_corpus.db")
        
        # Add sample URL - simple text for testing
        test_urls = [
            "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/README.md"
        ]
        
        added = platform.add_source_urls(test_urls, source_type="github", 
                                        language="en", priority=10)
        print(f"  ✓ Added {added} test URL(s)")
        
        # Run 1 cycle
        print("  → Running pipeline cycle...")
        await platform.run_pipeline(cycles=1, cycle_delay=1)
        
        # Check results
        stats = platform.db.get_statistics()
        print(f"  ✓ Total items: {stats['total_items']}")
        print(f"  ✓ Status counts: {stats['status_counts']}")
        
        # Clean up
        Path("test_corpus.db").unlink(missing_ok=True)
        import shutil
        shutil.rmtree("data", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 70)
    print("UNIFIED CORPUS PLATFORM - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Database
    results['database'] = test_database()
    
    # Test 3: Scraper
    results['scraper'] = await test_scraper()
    
    # Test 4: Full pipeline (only if previous tests passed)
    if all(results.values()):
        results['pipeline'] = await test_full_pipeline()
    else:
        print("\nSkipping full pipeline test (previous tests failed)")
        results['pipeline'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.upper():15s}: {status}")
    
    print("=" * 70)
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED! Platform is ready to use.")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED. Please check errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
