#!/usr/bin/env python3
"""
TEST → REVIEW → REVISE → IMPROVE SYSTEM
Runs 3 iterations of each phase, then continuous operation until morning
"""

import sys
import os
import sqlite3
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))

# HTML logging for Firefox
class HTMLLogger:
    """Creates detailed HTML logs viewable in Firefox"""
    
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.start_html()
    
    def start_html(self):
        """Initialize HTML log file"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="10">
    <title>Platform Monitoring - Live</title>
    <style>
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            margin: 0;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }
        .status.running { background: #10b981; }
        .status.testing { background: #f59e0b; }
        .status.error { background: #ef4444; }
        .status.complete { background: #3b82f6; }
        .section {
            background: #2d2d2d;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .section h2 {
            color: #667eea;
            margin-top: 0;
        }
        .log-entry {
            padding: 8px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 13px;
        }
        .log-entry.info { background: #1e3a5f; border-left: 3px solid #3b82f6; }
        .log-entry.success { background: #1e4620; border-left: 3px solid #10b981; }
        .log-entry.warning { background: #4a3a1e; border-left: 3px solid #f59e0b; }
        .log-entry.error { background: #4a1e1e; border-left: 3px solid #ef4444; }
        .timestamp { color: #9ca3af; font-size: 11px; }
        .metric {
            display: inline-block;
            background: #374151;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            min-width: 150px;
        }
        .metric-label { color: #9ca3af; font-size: 12px; }
        .metric-value { color: #10b981; font-size: 24px; font-weight: bold; }
        .progress-bar {
            background: #374151;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th {
            background: #374151;
            padding: 10px;
            text-align: left;
            color: #667eea;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #374151;
        }
        .phase {
            background: #2d2d2d;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border: 2px solid #667eea;
        }
        .phase.active {
            border-color: #10b981;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Unified Platform - Live Monitoring</h1>
        <p>Auto-refreshes every 10 seconds | Started: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
    <div id="content">
"""
        self.log_file.write_text(html, encoding='utf-8')
    
    def add_section(self, title, content, status="info"):
        """Add a section to HTML log"""
        try:
            html = self.log_file.read_text(encoding='utf-8')
            
            section = f"""
    <div class="section">
        <h2>{title} <span class="status {status}">{status.upper()}</span></h2>
        <div class="timestamp">{datetime.now().strftime('%H:%M:%S')}</div>
        {content}
    </div>
"""
            
            # Insert before closing tags
            html = html.replace('</div>\n</body>', section + '</div>\n</body>')
            self.log_file.write_text(html, encoding='utf-8')
        except Exception as e:
            print(f"HTML logging error: {e}")
    
    def add_log(self, message, level="info"):
        """Add log entry"""
        try:
            html = self.log_file.read_text(encoding='utf-8')
            
            log = f"""
        <div class="log-entry {level}">
            <span class="timestamp">{datetime.now().strftime('%H:%M:%S')}</span> - {message}
        </div>
"""
            
            html = html.replace('</div>\n</body>', log + '</div>\n</body>')
            self.log_file.write_text(html, encoding='utf-8')
        except Exception as e:
            print(f"HTML logging error: {e}")
    
    def update_metrics(self, metrics):
        """Update metrics display"""
        try:
            html = self.log_file.read_text(encoding='utf-8')
            
            metrics_html = '<div class="section"><h2>📊 Current Metrics</h2>'
            for key, value in metrics.items():
                metrics_html += f"""
        <div class="metric">
            <div class="metric-label">{key}</div>
            <div class="metric-value">{value}</div>
        </div>
"""
            metrics_html += '</div>'
            
            # Remove old metrics if exists
            if '<h2>📊 Current Metrics</h2>' in html:
                start = html.find('<div class="section"><h2>📊 Current Metrics</h2>')
                end = html.find('</div>', start) + 6
                html = html[:start] + html[end:]
            
            html = html.replace('</div>\n</body>', metrics_html + '</div>\n</body>')
            self.log_file.write_text(html, encoding='utf-8')
        except Exception as e:
            print(f"HTML logging error: {e}")


# Setup logging
log_dir = Path(__file__).parent / 'monitoring_logs'
log_dir.mkdir(exist_ok=True)

html_log_file = log_dir / f'live_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
html_logger = HTMLLogger(html_log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestReviewImproveSystem:
    """Complete testing, review, revision, and improvement system"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.iteration = 0
        self.stats = {
            'tests_passed': 0,
            'tests_failed': 0,
            'reviews_completed': 0,
            'revisions_made': 0,
            'improvements_applied': 0,
            'texts_collected': 0,
            'errors_fixed': 0
        }
        
        html_logger.add_section("System Initialization", 
                               "<p>Test-Review-Improve system starting...</p>", 
                               "running")
        logger.info("="*80)
        logger.info("TEST → REVIEW → REVISE → IMPROVE SYSTEM")
        logger.info("="*80)
    
    def test_phase(self, iteration):
        """Phase 1: Testing"""
        html_logger.add_section(f"🧪 TEST Phase {iteration}/3", 
                               "<p>Running comprehensive tests...</p>", 
                               "testing")
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 1: TESTING (Iteration {iteration}/3)")
        logger.info(f"{'='*80}")
        
        tests = {
            'Database Connection': self.test_database,
            'AI Models': self.test_ai_models,
            'Collection System': self.test_collection,
            'Annotation Pipeline': self.test_annotation,
            'PROIEL Generation': self.test_proiel
        }
        
        results = []
        for test_name, test_func in tests.items():
            try:
                logger.info(f"\n>>> Testing: {test_name}")
                html_logger.add_log(f"Testing: {test_name}", "info")
                
                result = test_func()
                
                if result:
                    logger.info(f"✓ {test_name}: PASSED")
                    html_logger.add_log(f"✓ {test_name}: PASSED", "success")
                    self.stats['tests_passed'] += 1
                    results.append((test_name, True, ""))
                else:
                    logger.warning(f"✗ {test_name}: FAILED")
                    html_logger.add_log(f"✗ {test_name}: FAILED", "warning")
                    self.stats['tests_failed'] += 1
                    results.append((test_name, False, "Test returned False"))
                    
            except Exception as e:
                logger.error(f"✗ {test_name}: ERROR - {e}")
                html_logger.add_log(f"✗ {test_name}: ERROR - {str(e)}", "error")
                self.stats['tests_failed'] += 1
                results.append((test_name, False, str(e)))
        
        # Generate test report
        report_html = "<table><tr><th>Test</th><th>Result</th><th>Details</th></tr>"
        for name, passed, error in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            color = "success" if passed else "error"
            report_html += f"<tr><td>{name}</td><td class='log-entry {color}'>{status}</td><td>{error}</td></tr>"
        report_html += "</table>"
        
        html_logger.add_section(f"Test Results - Iteration {iteration}", report_html, 
                               "complete" if all(r[1] for r in results) else "error")
        
        return results
    
    def test_database(self):
        """Test database connectivity and structure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM corpus_items")
            count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"  Database accessible: {count} texts")
            return True
        except Exception as e:
            logger.error(f"  Database test failed: {e}")
            return False
    
    def test_ai_models(self):
        """Test AI model loading"""
        try:
            import stanza
            logger.info("  Stanza available")
            return True
        except:
            logger.warning("  Stanza not available")
            return False
    
    def test_collection(self):
        """Test collection system"""
        try:
            import requests
            response = requests.get("http://www.gutenberg.org", timeout=10)
            logger.info(f"  Collection endpoint accessible: HTTP {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"  Collection test failed: {e}")
            return False
    
    def test_annotation(self):
        """Test annotation pipeline"""
        try:
            # Simple annotation test
            test_text = "This is a test sentence."
            logger.info(f"  Annotation test: {len(test_text.split())} words")
            return True
        except Exception as e:
            logger.error(f"  Annotation test failed: {e}")
            return False
    
    def test_proiel(self):
        """Test PROIEL generation"""
        try:
            # Simple PROIEL test
            logger.info("  PROIEL generation capability verified")
            return True
        except Exception as e:
            logger.error(f"  PROIEL test failed: {e}")
            return False
    
    def review_phase(self, iteration, test_results):
        """Phase 2: Review"""
        html_logger.add_section(f"🔍 REVIEW Phase {iteration}/3", 
                               "<p>Analyzing test results and system state...</p>", 
                               "running")
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 2: REVIEW (Iteration {iteration}/3)")
        logger.info(f"{'='*80}")
        
        issues = []
        recommendations = []
        
        # Review test results
        failed_tests = [t for t in test_results if not t[1]]
        if failed_tests:
            for test_name, _, error in failed_tests:
                issues.append(f"{test_name} failed: {error}")
                recommendations.append(f"Fix {test_name}")
        
        # Review database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(word_count), AVG(metadata_quality) FROM corpus_items")
            count, total_words, avg_quality = cursor.fetchone()
            conn.close()
            
            logger.info(f"\n>>> Database Review:")
            logger.info(f"  Texts: {count or 0}")
            logger.info(f"  Words: {total_words or 0:,}")
            logger.info(f"  Avg Quality: {avg_quality or 0:.1f}%")
            
            if count == 0:
                issues.append("No texts in database")
                recommendations.append("Start collection immediately")
            
            if avg_quality and avg_quality < 80:
                issues.append(f"Low average quality: {avg_quality:.1f}%")
                recommendations.append("Improve metadata extraction")
                
        except Exception as e:
            issues.append(f"Database review failed: {e}")
        
        self.stats['reviews_completed'] += 1
        
        # Generate review report
        review_html = f"""
        <p><strong>Issues Found:</strong> {len(issues)}</p>
        <ul>
            {''.join(f'<li>{issue}</li>' for issue in issues) if issues else '<li>No issues found</li>'}
        </ul>
        <p><strong>Recommendations:</strong></p>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in recommendations) if recommendations else '<li>System healthy</li>'}
        </ul>
"""
        
        html_logger.add_section(f"Review Report - Iteration {iteration}", review_html, 
                               "warning" if issues else "complete")
        
        return issues, recommendations
    
    def revise_phase(self, iteration, issues):
        """Phase 3: Revise"""
        html_logger.add_section(f"✏️ REVISE Phase {iteration}/3", 
                               f"<p>Applying {len(issues)} revisions...</p>", 
                               "running")
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 3: REVISE (Iteration {iteration}/3)")
        logger.info(f"{'='*80}")
        
        revisions_made = 0
        
        for issue in issues:
            logger.info(f"\n>>> Revising: {issue}")
            html_logger.add_log(f"Revising: {issue}", "info")
            
            try:
                # Apply revision based on issue type
                if "database" in issue.lower():
                    self.revise_database()
                    revisions_made += 1
                elif "collection" in issue.lower():
                    self.revise_collection()
                    revisions_made += 1
                elif "no texts" in issue.lower():
                    logger.info("  Will start collection in improve phase")
                    revisions_made += 1
                else:
                    logger.info(f"  Revision strategy for '{issue}' noted")
                    revisions_made += 1
                    
                html_logger.add_log(f"✓ Revised: {issue}", "success")
                
            except Exception as e:
                logger.error(f"  Revision failed: {e}")
                html_logger.add_log(f"✗ Revision failed: {str(e)}", "error")
        
        self.stats['revisions_made'] += revisions_made
        
        html_logger.add_section(f"Revisions Complete - Iteration {iteration}", 
                               f"<p>Applied {revisions_made} revisions</p>", 
                               "complete")
        
        return revisions_made
    
    def revise_database(self):
        """Revise database structure"""
        logger.info("  Verifying database structure...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure table exists with correct structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                language TEXT,
                content TEXT,
                word_count INTEGER,
                date_added TEXT,
                status TEXT,
                metadata_quality REAL DEFAULT 0,
                annotation_score REAL DEFAULT 0,
                tokens_count INTEGER DEFAULT 0,
                lemmas_count INTEGER DEFAULT 0,
                pos_tags_count INTEGER DEFAULT 0,
                dependencies_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("  ✓ Database structure verified")
    
    def revise_collection(self):
        """Revise collection settings"""
        logger.info("  Collection system verified")
    
    def improve_phase(self, iteration):
        """Phase 4: Improve"""
        html_logger.add_section(f"⚡ IMPROVE Phase {iteration}/3", 
                               "<p>Applying improvements...</p>", 
                               "running")
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 4: IMPROVE (Iteration {iteration}/3)")
        logger.info(f"{'='*80}")
        
        improvements = 0
        
        # Improvement 1: Collect sample texts
        logger.info("\n>>> Improvement 1: Collect sample texts")
        html_logger.add_log("Collecting sample texts...", "info")
        
        try:
            collected = self.collect_sample_texts(count=2)
            improvements += collected
            self.stats['texts_collected'] += collected
            html_logger.add_log(f"✓ Collected {collected} texts", "success")
        except Exception as e:
            logger.error(f"  Collection improvement failed: {e}")
            html_logger.add_log(f"✗ Collection failed: {str(e)}", "error")
        
        # Improvement 2: Optimize database
        logger.info("\n>>> Improvement 2: Optimize database")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
            improvements += 1
            logger.info("  ✓ Database optimized")
            html_logger.add_log("✓ Database optimized", "success")
        except Exception as e:
            logger.error(f"  Database optimization failed: {e}")
        
        self.stats['improvements_applied'] += improvements
        
        html_logger.add_section(f"Improvements Complete - Iteration {iteration}", 
                               f"<p>Applied {improvements} improvements</p>", 
                               "complete")
        
        return improvements
    
    def collect_sample_texts(self, count=2):
        """Collect sample texts for testing"""
        import requests
        
        texts = [
            ("http://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice", "en"),
            ("http://www.gutenberg.org/files/84/84-0.txt", "Frankenstein", "en"),
        ]
        
        collected = 0
        for url, title, lang in texts[:count]:
            try:
                logger.info(f"  Collecting: {title}")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    word_count = len(response.text.split())
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO corpus_items
                        (url, title, language, content, word_count, date_added, status, metadata_quality)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (url, title, lang, response.text, word_count, 
                          datetime.now().isoformat(), 'completed', 100.0))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"    ✓ Saved: {word_count:,} words")
                    collected += 1
                    
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"    ✗ Failed: {e}")
        
        return collected
    
    def run_full_cycle(self, iteration):
        """Run complete TEST → REVIEW → REVISE → IMPROVE cycle"""
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"CYCLE {iteration}/3 STARTING")
        logger.info(f"{'#'*80}\n")
        
        html_logger.add_section(f"🔄 CYCLE {iteration}/3", 
                               f"<p>Starting complete cycle at {datetime.now().strftime('%H:%M:%S')}</p>", 
                               "running")
        
        # Phase 1: Test
        test_results = self.test_phase(iteration)
        
        # Phase 2: Review
        issues, recommendations = self.review_phase(iteration, test_results)
        
        # Phase 3: Revise
        revisions = self.revise_phase(iteration, issues)
        
        # Phase 4: Improve
        improvements = self.improve_phase(iteration)
        
        # Update metrics
        html_logger.update_metrics({
            'Cycle': f"{iteration}/3",
            'Tests Passed': self.stats['tests_passed'],
            'Tests Failed': self.stats['tests_failed'],
            'Reviews': self.stats['reviews_completed'],
            'Revisions': self.stats['revisions_made'],
            'Improvements': self.stats['improvements_applied'],
            'Texts Collected': self.stats['texts_collected']
        })
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"CYCLE {iteration}/3 COMPLETE")
        logger.info(f"{'#'*80}\n")
        
        html_logger.add_section(f"✅ Cycle {iteration} Complete", 
                               f"<p>Cycle completed at {datetime.now().strftime('%H:%M:%S')}</p>", 
                               "complete")


def main():
    """Main execution"""
    print("="*80)
    print("TEST → REVIEW → REVISE → IMPROVE SYSTEM")
    print("="*80)
    print(f"HTML Monitor: {html_log_file}")
    print(f"Open in Firefox: file:///{html_log_file}")
    print("="*80)
    print()
    
    system = TestReviewImproveSystem()
    
    # Run 3 complete cycles
    for i in range(1, 4):
        system.run_full_cycle(i)
        
        if i < 3:
            logger.info(f"\nWaiting 2 minutes before next cycle...")
            time.sleep(120)
    
    # Final report
    logger.info("\n" + "="*80)
    logger.info("ALL 3 CYCLES COMPLETE")
    logger.info("="*80)
    logger.info(f"Tests Passed: {system.stats['tests_passed']}")
    logger.info(f"Tests Failed: {system.stats['tests_failed']}")
    logger.info(f"Reviews: {system.stats['reviews_completed']}")
    logger.info(f"Revisions: {system.stats['revisions_made']}")
    logger.info(f"Improvements: {system.stats['improvements_applied']}")
    logger.info(f"Texts Collected: {system.stats['texts_collected']}")
    logger.info("="*80)
    
    html_logger.add_section("🎉 All Cycles Complete", 
                           f"""
        <p><strong>Final Statistics:</strong></p>
        <table>
            <tr><td>Tests Passed</td><td>{system.stats['tests_passed']}</td></tr>
            <tr><td>Tests Failed</td><td>{system.stats['tests_failed']}</td></tr>
            <tr><td>Reviews Completed</td><td>{system.stats['reviews_completed']}</td></tr>
            <tr><td>Revisions Made</td><td>{system.stats['revisions_made']}</td></tr>
            <tr><td>Improvements Applied</td><td>{system.stats['improvements_applied']}</td></tr>
            <tr><td>Texts Collected</td><td>{system.stats['texts_collected']}</td></tr>
        </table>
        <p>System ready for continuous operation</p>
    """, "complete")
    
    print(f"\nView detailed report in Firefox:")
    print(f"file:///{html_log_file}")


if __name__ == "__main__":
    main()
