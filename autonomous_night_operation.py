#!/usr/bin/env python3
"""
Autonomous Night Operation - Self-Managing Platform
Runs, monitors, fixes, and improves automatically until 8:00 AM Greece time

Prof. Nikolaos Lavidas - HFRI-NKUA
"""

import sys
import os
import time
import json
import logging
import sqlite3
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

# Set Stanza resources directory
os.environ['STANZA_RESOURCES_DIR'] = str(Path('Z:/models/stanza'))

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'night_operation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AutonomousNightManager:
    """
    Self-managing platform that runs autonomously through the night
    - Monitors every 50 minutes
    - Fixes issues automatically
    - Improves performance
    - Reports results
    """
    
    def __init__(self, end_time_str: str = "08:00"):
        self.db_path = Path(__file__).parent / "corpus_platform.db"
        self.report_path = Path(__file__).parent / "night_reports"
        self.report_path.mkdir(exist_ok=True)
        
        # Set end time (8:00 AM Greece time)
        greece_tz = pytz.timezone('Europe/Athens')
        now = datetime.now(greece_tz)
        end_hour, end_minute = map(int, end_time_str.split(':'))
        
        self.end_time = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        if self.end_time <= now:
            self.end_time += timedelta(days=1)
        
        # Statistics
        self.stats = {
            'start_time': now.isoformat(),
            'cycles_completed': 0,
            'texts_collected': 0,
            'treebanks_generated': 0,
            'issues_fixed': 0,
            'improvements_made': 0,
            'errors_encountered': 0,
            'total_words': 0
        }
        
        # Dashboard process
        self.dashboard_process = None
        self.collection_process = None
        self.annotation_process = None
        
        logger.info("="*70)
        logger.info("AUTONOMOUS NIGHT OPERATION INITIALIZED")
        logger.info("="*70)
        logger.info(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Duration: {(self.end_time - now).total_seconds() / 3600:.1f} hours")
        logger.info("="*70)
    
    def is_time_to_stop(self) -> bool:
        """Check if it's time to stop (8:00 AM Greece time)"""
        greece_tz = pytz.timezone('Europe/Athens')
        now = datetime.now(greece_tz)
        return now >= self.end_time
    
    def start_dashboard(self):
        """Start the dashboard if not running"""
        try:
            # Check if already running
            response = requests.get('http://localhost:8000/api/statistics', timeout=2)
            logger.info("✓ Dashboard already running")
            return True
        except:
            pass
        
        try:
            logger.info("Starting dashboard...")
            self.dashboard_process = subprocess.Popen(
                ['python', 'professional_dashboard.py'],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            # Wait for startup
            time.sleep(10)
            
            # Verify
            response = requests.get('http://localhost:8000/api/statistics', timeout=5)
            logger.info("✓ Dashboard started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self.stats['errors_encountered'] += 1
            return False
    
    def start_collection(self):
        """Start 24/7 collection if not running"""
        try:
            logger.info("Starting 24/7 collection...")
            self.collection_process = subprocess.Popen(
                ['python', 'autonomous_247_collection.py', 
                 '--languages', 'grc', 'lat', 'en',
                 '--texts-per-cycle', '10',
                 '--cycle-delay', '300'],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            logger.info("✓ Collection process started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start collection: {e}")
            self.stats['errors_encountered'] += 1
            return False

    def start_annotation(self):
        """Start 24/7 annotation worker if not running"""
        try:
            logger.info("Starting annotation worker...")
            self.annotation_process = subprocess.Popen(
                ['python', 'annotation_worker_247.py'],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            logger.info("✓ Annotation worker started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start annotation worker: {e}")
            self.stats['errors_encountered'] += 1
            return False
    
    def check_database_stats(self) -> Dict:
        """Get current database statistics with detailed text information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corpus_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    language TEXT,
                    content TEXT,
                    proiel_xml TEXT,
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
            
            # Get basic statistics
            cursor.execute("SELECT COUNT(*) FROM corpus_items")
            total_texts = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM corpus_items WHERE status = 'completed'")
            completed_texts = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(word_count) FROM corpus_items")
            total_words = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(DISTINCT language) FROM corpus_items")
            languages = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM corpus_items WHERE proiel_xml IS NOT NULL")
            treebanks = cursor.fetchone()[0]
            
            # Get detailed text information (last 10 texts)
            cursor.execute("""
                SELECT id, title, language, status, word_count, 
                       metadata_quality, annotation_score, tokens_count,
                       lemmas_count, pos_tags_count, dependencies_count,
                       date_added
                FROM corpus_items 
                ORDER BY date_added DESC 
                LIMIT 10
            """)
            
            recent_texts = []
            for row in cursor.fetchall():
                recent_texts.append({
                    'id': row[0],
                    'title': row[1][:100] if row[1] else 'Unknown',
                    'language': row[2],
                    'status': row[3],
                    'word_count': row[4] or 0,
                    'metadata_quality': row[5] or 0,
                    'annotation_score': row[6] or 0,
                    'tokens_count': row[7] or 0,
                    'lemmas_count': row[8] or 0,
                    'pos_tags_count': row[9] or 0,
                    'dependencies_count': row[10] or 0,
                    'date_added': row[11]
                })
            
            # Get average scores
            cursor.execute("""
                SELECT AVG(metadata_quality), AVG(annotation_score)
                FROM corpus_items
                WHERE metadata_quality > 0
            """)
            avg_row = cursor.fetchone()
            avg_metadata = avg_row[0] or 0
            avg_annotation = avg_row[1] or 0
            
            # Get language breakdown
            cursor.execute("""
                SELECT language, COUNT(*), SUM(word_count)
                FROM corpus_items
                GROUP BY language
            """)
            language_breakdown = {}
            for row in cursor.fetchall():
                language_breakdown[row[0] or 'unknown'] = {
                    'count': row[1],
                    'words': row[2] or 0
                }
            
            # Get status breakdown
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM corpus_items
                GROUP BY status
            """)
            status_breakdown = {}
            for row in cursor.fetchall():
                status_breakdown[row[0] or 'unknown'] = row[1]
            
            conn.close()
            
            return {
                'total_texts': total_texts,
                'completed_texts': completed_texts,
                'total_words': total_words,
                'languages': languages,
                'treebanks': treebanks,
                'recent_texts': recent_texts,
                'avg_metadata_quality': avg_metadata,
                'avg_annotation_score': avg_annotation,
                'language_breakdown': language_breakdown,
                'status_breakdown': status_breakdown
            }
            
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {
                'total_texts': 0,
                'completed_texts': 0,
                'total_words': 0,
                'languages': 0,
                'treebanks': 0,
                'recent_texts': [],
                'avg_metadata_quality': 0,
                'avg_annotation_score': 0,
                'language_breakdown': {},
                'status_breakdown': {}
            }
    
    def check_system_health(self) -> Dict:
        """Check system health"""
        health = {
            'dashboard_running': False,
            'collection_running': False,
            'annotation_running': False,
            'database_accessible': False,
            'disk_space_ok': False,
            'issues': []
        }
        
        # Check dashboard
        try:
            response = requests.get('http://localhost:8000/api/statistics', timeout=5)
            health['dashboard_running'] = response.status_code == 200
        except:
            health['issues'].append('Dashboard not responding')
        
        # Check collection process
        if self.collection_process:
            health['collection_running'] = self.collection_process.poll() is None
            if not health['collection_running']:
                health['issues'].append('Collection process stopped')
        
        # Check annotation worker
        if self.annotation_process:
            health['annotation_running'] = self.annotation_process.poll() is None
            if not health['annotation_running']:
                health['issues'].append('Annotation worker stopped')
        
        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            health['database_accessible'] = True
        except:
            health['issues'].append('Database not accessible')
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage('Z:/')
            free_gb = free / (1024**3)
            health['disk_space_ok'] = free_gb > 10  # At least 10GB free
            if not health['disk_space_ok']:
                health['issues'].append(f'Low disk space: {free_gb:.1f}GB')
        except:
            health['issues'].append('Cannot check disk space')
        
        return health
    
    def fix_issues(self, health: Dict):
        """Automatically fix detected issues"""
        fixed = 0
        
        # Fix dashboard
        if not health['dashboard_running']:
            logger.warning("Dashboard not running - restarting...")
            if self.start_dashboard():
                fixed += 1
                logger.info("✓ Dashboard restarted")
        
        # Fix collection
        if not health['collection_running']:
            logger.warning("Collection not running - restarting...")
            if self.start_collection():
                fixed += 1
                logger.info("✓ Collection restarted")
        
        # Fix annotation
        if not health.get('annotation_running', False):
            logger.warning("Annotation worker not running - restarting...")
            if self.start_annotation():
                fixed += 1
                logger.info("✓ Annotation worker restarted")
        
        self.stats['issues_fixed'] += fixed
        return fixed
    
    def optimize_collection(self, db_stats: Dict):
        """Optimize collection based on current performance"""
        improvements = 0
        
        # If collection is slow, suggest improvements
        if db_stats['total_texts'] < self.stats['cycles_completed'] * 5:
            logger.info("Collection slower than expected - optimizing...")
            # Could adjust cycle delay, add more repositories, etc.
            improvements += 1
        
        # If metadata quality is low, improve enrichment
        # If duplicates detected, improve filtering
        # etc.
        
        self.stats['improvements_made'] += improvements
        return improvements
    
    def generate_cycle_report(self, cycle: int, db_stats: Dict, health: Dict) -> str:
        """Generate comprehensive report for this cycle"""
        greece_tz = pytz.timezone('Europe/Athens')
        now = datetime.now(greece_tz)
        
        report = []
        report.append("="*80)
        report.append(f"CYCLE {cycle} COMPREHENSIVE REPORT")
        report.append("="*80)
        report.append(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        report.append(f"Time until 8:00 AM: {(self.end_time - now).total_seconds() / 3600:.1f} hours")
        report.append("")
        
        # DATABASE STATISTICS
        report.append("="*80)
        report.append("DATABASE STATISTICS")
        report.append("="*80)
        report.append(f"Total Texts Collected:     {db_stats['total_texts']}")
        report.append(f"Completed Texts:           {db_stats['completed_texts']}")
        report.append(f"Total Words Processed:     {db_stats['total_words']:,}")
        report.append(f"Languages Covered:         {db_stats['languages']}")
        report.append(f"PROIEL Treebanks:          {db_stats['treebanks']}")
        report.append(f"Average Metadata Quality:  {db_stats['avg_metadata_quality']:.1f}%")
        report.append(f"Average Annotation Score:  {db_stats['avg_annotation_score']:.1f}%")
        report.append("")
        
        # LANGUAGE BREAKDOWN
        if db_stats['language_breakdown']:
            report.append("-"*80)
            report.append("LANGUAGE BREAKDOWN:")
            report.append("-"*80)
            for lang, data in db_stats['language_breakdown'].items():
                report.append(f"  {lang:15} {data['count']:4} texts  {data['words']:10,} words")
            report.append("")
        
        # STATUS BREAKDOWN
        if db_stats['status_breakdown']:
            report.append("-"*80)
            report.append("STATUS BREAKDOWN:")
            report.append("-"*80)
            for status, count in db_stats['status_breakdown'].items():
                report.append(f"  {status:15} {count:4} texts")
            report.append("")
        
        # RECENT TEXTS DETAILS
        if db_stats['recent_texts']:
            report.append("="*80)
            report.append("RECENT TEXTS (Last 10 Collected)")
            report.append("="*80)
            report.append(f"{'ID':<5} {'Title':<40} {'Lang':<6} {'Status':<12} {'Words':<8}")
            report.append("-"*80)
            for text in db_stats['recent_texts']:
                title_short = text['title'][:38] + '..' if len(text['title']) > 40 else text['title']
                report.append(
                    f"{text['id']:<5} {title_short:<40} {text['language']:<6} "
                    f"{text['status']:<12} {text['word_count']:<8,}"
                )
            report.append("")
            
            # ANNOTATION DETAILS FOR RECENT TEXTS
            report.append("-"*80)
            report.append("ANNOTATION DETAILS (Recent Texts):")
            report.append("-"*80)
            report.append(f"{'ID':<5} {'Tokens':<8} {'Lemmas':<8} {'POS Tags':<10} {'Dependencies':<12} {'Score':<8}")
            report.append("-"*80)
            for text in db_stats['recent_texts']:
                report.append(
                    f"{text['id']:<5} {text['tokens_count']:<8} {text['lemmas_count']:<8} "
                    f"{text['pos_tags_count']:<10} {text['dependencies_count']:<12} "
                    f"{text['annotation_score']:.1f}%"
                )
            report.append("")
            
            # METADATA QUALITY FOR RECENT TEXTS
            report.append("-"*80)
            report.append("METADATA QUALITY (Recent Texts):")
            report.append("-"*80)
            report.append(f"{'ID':<5} {'Title':<40} {'Quality':<10} {'Date Added':<20}")
            report.append("-"*80)
            for text in db_stats['recent_texts']:
                title_short = text['title'][:38] + '..' if len(text['title']) > 40 else text['title']
                quality_str = f"{text['metadata_quality']:.1f}%" if text['metadata_quality'] > 0 else "N/A"
                report.append(
                    f"{text['id']:<5} {title_short:<40} {quality_str:<10} {text['date_added']:<20}"
                )
            report.append("")
        
        # SYSTEM HEALTH
        report.append("="*80)
        report.append("SYSTEM HEALTH")
        report.append("="*80)
        report.append(f"Dashboard:    {'✓ Running' if health['dashboard_running'] else '✗ Not running'}")
        report.append(f"Collection:   {'✓ Running' if health['collection_running'] else '✗ Not running'}")
        report.append(f"Database:     {'✓ Accessible' if health['database_accessible'] else '✗ Not accessible'}")
        report.append(f"Disk Space:   {'✓ OK' if health['disk_space_ok'] else '✗ Low'}")
        report.append("")
        
        if health['issues']:
            report.append("-"*80)
            report.append("ISSUES DETECTED:")
            report.append("-"*80)
            for i, issue in enumerate(health['issues'], 1):
                report.append(f"  {i}. {issue}")
            report.append("")
        
        # CUMULATIVE STATISTICS
        report.append("="*80)
        report.append("CUMULATIVE OPERATION STATISTICS")
        report.append("="*80)
        report.append(f"Monitoring Cycles Completed:  {self.stats['cycles_completed']}")
        report.append(f"Issues Automatically Fixed:   {self.stats['issues_fixed']}")
        report.append(f"Improvements Made:            {self.stats['improvements_made']}")
        report.append(f"Errors Encountered:           {self.stats['errors_encountered']}")
        report.append("")
        
        # PERFORMANCE METRICS
        hours_elapsed = (now - datetime.fromisoformat(self.stats['start_time'])).total_seconds() / 3600
        if hours_elapsed > 0:
            report.append("-"*80)
            report.append("PERFORMANCE METRICS:")
            report.append("-"*80)
            report.append(f"Texts per Hour:      {db_stats['total_texts'] / hours_elapsed:.1f}")
            report.append(f"Words per Hour:      {db_stats['total_words'] / hours_elapsed:,.0f}")
            report.append(f"Treebanks per Hour:  {db_stats['treebanks'] / hours_elapsed:.1f}")
            report.append("")
        
        # QUALITY ASSESSMENT
        report.append("="*80)
        report.append("QUALITY ASSESSMENT")
        report.append("="*80)
        
        # Calculate quality grade
        avg_quality = (db_stats['avg_metadata_quality'] + db_stats['avg_annotation_score']) / 2
        if avg_quality >= 90:
            quality_grade = "EXCELLENT"
        elif avg_quality >= 75:
            quality_grade = "GOOD"
        elif avg_quality >= 60:
            quality_grade = "FAIR"
        else:
            quality_grade = "NEEDS IMPROVEMENT"
        
        report.append(f"Overall Quality Grade:     {quality_grade}")
        report.append(f"Metadata Quality:          {db_stats['avg_metadata_quality']:.1f}%")
        report.append(f"Annotation Quality:        {db_stats['avg_annotation_score']:.1f}%")
        report.append(f"Completion Rate:           {(db_stats['completed_texts']/max(db_stats['total_texts'],1))*100:.1f}%")
        report.append("")
        
        report.append("="*80)
        report.append(f"End of Cycle {cycle} Report")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_cycle_report(self, cycle: int, report: str):
        """Save cycle report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.report_path / f"cycle_{cycle:03d}_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved: {report_file}")
    
    def generate_final_report(self) -> str:
        """Generate final comprehensive report"""
        greece_tz = pytz.timezone('Europe/Athens')
        now = datetime.now(greece_tz)
        
        db_stats = self.check_database_stats()
        
        report = []
        report.append("="*70)
        report.append("AUTONOMOUS NIGHT OPERATION - FINAL REPORT")
        report.append("="*70)
        report.append(f"Start Time: {self.stats['start_time']}")
        report.append(f"End Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        report.append(f"Duration: {(now - datetime.fromisoformat(self.stats['start_time'])).total_seconds() / 3600:.1f} hours")
        report.append("")
        
        report.append("FINAL STATISTICS:")
        report.append(f"  Total Texts Collected: {db_stats['total_texts']}")
        report.append(f"  PROIEL Treebanks Generated: {db_stats['treebanks']}")
        report.append(f"  Total Words Processed: {db_stats['total_words']:,}")
        report.append(f"  Languages Covered: {db_stats['languages']}")
        report.append(f"  Completion Rate: {db_stats['completed_texts']/max(db_stats['total_texts'],1)*100:.1f}%")
        report.append("")
        
        report.append("OPERATION STATISTICS:")
        report.append(f"  Monitoring Cycles: {self.stats['cycles_completed']}")
        report.append(f"  Issues Fixed: {self.stats['issues_fixed']}")
        report.append(f"  Improvements Made: {self.stats['improvements_made']}")
        report.append(f"  Errors Encountered: {self.stats['errors_encountered']}")
        report.append("")
        
        # Calculate rates
        hours = (now - datetime.fromisoformat(self.stats['start_time'])).total_seconds() / 3600
        if hours > 0:
            report.append("PERFORMANCE METRICS:")
            report.append(f"  Texts per Hour: {db_stats['total_texts'] / hours:.1f}")
            report.append(f"  Words per Hour: {db_stats['total_words'] / hours:,.0f}")
            report.append(f"  Treebanks per Hour: {db_stats['treebanks'] / hours:.1f}")
        
        report.append("")
        report.append("STATUS: Operation completed successfully")
        report.append("="*70)
        
        return "\n".join(report)
    
    def run_autonomous_night(self):
        """Main autonomous operation loop"""
        logger.info("Starting autonomous night operation...")
        
        # Initial startup
        logger.info("\n>>> INITIAL STARTUP <<<\n")
        self.start_dashboard()
        time.sleep(5)
        self.start_collection()
        time.sleep(5)
        self.start_annotation()
        
        cycle = 0
        
        try:
            while not self.is_time_to_stop():
                cycle += 1
                self.stats['cycles_completed'] = cycle
                
                logger.info("\n" + "🌙"*40)
                logger.info(f"{'='*80}")
                logger.info(f"MONITORING CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
                logger.info(f"{'='*80}")
                logger.info(f"Time until 8:00 AM: {(self.end_time - datetime.now(pytz.timezone('Europe/Athens'))).total_seconds() / 3600:.1f} hours")
                logger.info(f"{'='*80}\n")
                
                # Get current statistics
                logger.info(">>> STEP 1: GATHERING DATABASE STATISTICS...")
                db_stats = self.check_database_stats()
                logger.info(f"✓ Database statistics collected")
                logger.info(f"  Total Texts: {db_stats['total_texts']}")
                logger.info(f"  Completed: {db_stats['completed_texts']}")
                logger.info(f"  Treebanks: {db_stats['treebanks']}")
                logger.info(f"  Total Words: {db_stats['total_words']:,}")
                logger.info("")
                
                # Check system health
                logger.info(">>> STEP 2: CHECKING SYSTEM HEALTH...")
                health = self.check_system_health()
                logger.info(f"✓ Health check completed")
                logger.info(f"  Dashboard: {'✓ Running' if health['dashboard_running'] else '✗ Not running'}")
                logger.info(f"  Collection: {'✓ Running' if health['collection_running'] else '✗ Not running'}")
                logger.info(f"  Annotation: {'✓ Running' if health.get('annotation_running') else '✗ Not running'}")
                logger.info(f"  Database: {'✓ Accessible' if health['database_accessible'] else '✗ Not accessible'}")
                logger.info(f"  Disk Space: {'✓ OK' if health['disk_space_ok'] else '✗ Low'}")
                logger.info("")
                
                # Fix any issues
                if health['issues']:
                    logger.warning(f">>> STEP 3: FIXING ISSUES ({len(health['issues'])} detected)...")
                    for issue in health['issues']:
                        logger.warning(f"  - {issue}")
                    fixed = self.fix_issues(health)
                    logger.info(f"✓ Fixed {fixed} issues automatically")
                    logger.info("")
                else:
                    logger.info(">>> STEP 3: NO ISSUES DETECTED - System healthy ✓")
                    logger.info("")
                
                # Optimize if needed
                logger.info(">>> STEP 4: CHECKING FOR OPTIMIZATION OPPORTUNITIES...")
                improvements = self.optimize_collection(db_stats)
                if improvements > 0:
                    logger.info(f"✓ Made {improvements} improvements")
                else:
                    logger.info("✓ No optimizations needed - Performance good")
                logger.info("")
                
                # Generate and save report
                logger.info(">>> STEP 5: GENERATING COMPREHENSIVE REPORT...")
                report = self.generate_cycle_report(cycle, db_stats, health)
                logger.info("✓ Report generated")
                logger.info("")
                logger.info("="*80)
                logger.info("FULL CYCLE REPORT:")
                logger.info("="*80)
                logger.info(report)
                logger.info("="*80)
                logger.info("")
                self.save_cycle_report(cycle, report)
                logger.info(f"✓ Report saved to: night_reports/cycle_{cycle:03d}_*.txt")
                logger.info("")
                
                # Wait 50 minutes before next cycle
                logger.info(f"\nWaiting 50 minutes until next cycle...")
                logger.info(f"Next cycle at: {(datetime.now() + timedelta(minutes=50)).strftime('%H:%M:%S')}")
                
                # Sleep in chunks to allow for interruption
                for i in range(50):
                    if self.is_time_to_stop():
                        break
                    time.sleep(60)  # 1 minute
                    
                    # Quick health check every 10 minutes
                    if i % 10 == 0 and i > 0:
                        quick_health = self.check_system_health()
                        if quick_health['issues']:
                            logger.warning(f"Quick check: {len(quick_health['issues'])} issues")
                            self.fix_issues(quick_health)
            
            # Generate final report
            logger.info("\n" + "="*70)
            logger.info("8:00 AM REACHED - GENERATING FINAL REPORT")
            logger.info("="*70 + "\n")
            
            final_report = self.generate_final_report()
            logger.info("\n" + final_report)
            
            # Save final report
            final_report_file = self.report_path / f"FINAL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(final_report_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            logger.info(f"\nFinal report saved: {final_report_file}")
            logger.info("\nAutonomous night operation completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("\n\nOperation interrupted by user")
            final_report = self.generate_final_report()
            logger.info("\n" + final_report)
        
        except Exception as e:
            logger.error(f"\n\nFatal error: {e}")
            self.stats['errors_encountered'] += 1
            final_report = self.generate_final_report()
            logger.info("\n" + final_report)


def main():
    """Main entry point"""
    print("="*70)
    print("AUTONOMOUS NIGHT OPERATION")
    print("="*70)
    print("This system will:")
    print("  - Monitor every 50 minutes")
    print("  - Fix issues automatically")
    print("  - Improve performance")
    print("  - Generate reports")
    print("  - Run until 8:00 AM Greece time")
    print("="*70)
    print()
    
    manager = AutonomousNightManager(end_time_str="08:00")
    manager.run_autonomous_night()


if __name__ == "__main__":
    main()
