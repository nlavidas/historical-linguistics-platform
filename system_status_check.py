#!/usr/bin/env python3
"""
QUICK SYSTEM STATUS CHECK
Shows current state of monitoring, pipeline, and research progress
"""

import os
import json
from pathlib import Path
from datetime import datetime
import sqlite3

def check_system_status():
    """Comprehensive system status check"""

    print("🔍 HISTORICAL LINGUISTICS RESEARCH PLATFORM - STATUS CHECK")
    print("=" * 60)

    # Check databases
    print("\n📊 DATABASE STATUS:")
    databases = ['corpus_platform.db', 'monitoring.db', 'corpus_efficient.db']
    for db in databases:
        if os.path.exists(db):
            size_mb = os.path.getsize(db) / 1024 / 1024
            print(".1f"        else:
            print(f"❌ {db}: Not found")

    # Check monitoring status
    print("\n📈 MONITORING SYSTEM:")
    if os.path.exists('monitoring.db'):
        try:
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()

            # Performance metrics count
            cursor.execute('SELECT COUNT(*) FROM performance_metrics')
            metrics_count = cursor.fetchone()[0]

            # Active alerts
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE resolved = 0')
            active_alerts = cursor.fetchone()[0]

            # Improvement suggestions
            cursor.execute('SELECT COUNT(*) FROM improvement_suggestions WHERE implementation_status = "pending"')
            pending_suggestions = cursor.fetchone()[0]

            print(f"✅ Monitoring active: {metrics_count} metrics collected")
            print(f"⚠️  Active alerts: {active_alerts}")
            print(f"💡 Pending improvements: {pending_suggestions}")

            conn.close()
        except Exception as e:
            print(f"❌ Monitoring database error: {e}")
    else:
        print("❌ Monitoring system not initialized")

    # Check corpus status
    print("\n📚 CORPUS STATUS:")
    corpus_dbs = ['corpus_platform.db', 'corpus_efficient.db']
    for db in corpus_dbs:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                cursor = conn.cursor()

                # Get total texts
                cursor.execute('SELECT COUNT(*) FROM corpus_items')
                total_texts = cursor.fetchone()[0]

                # Get word count
                cursor.execute('SELECT SUM(word_count) FROM corpus_items')
                total_words = cursor.fetchone()[0] or 0

                # Get language breakdown
                cursor.execute('SELECT language, COUNT(*) FROM corpus_items GROUP BY language')
                languages = dict(cursor.fetchall())

                print(f"📖 {db}: {total_texts:,} texts, {total_words:,} words")
                if languages:
                    lang_str = ", ".join([f"{lang}: {count}" for lang, count in languages.items()])
                    print(f"   Languages: {lang_str}")

                conn.close()
            except Exception as e:
                print(f"❌ {db} error: {e}")

    # Check pipeline components
    print("\n🔧 PIPELINE COMPONENTS:")
    components = [
        'cost_effective_text_collection.py',
        'cost_effective_preprocessing.py',
        'cost_effective_parsing.py',
        'cost_effective_annotation.py',
        'cost_effective_valency.py',
        'cost_effective_diachronic.py',
        'continuous_monitoring.py',
        'progress_dashboard.py',
        'master_workflow_coordinator.py'
    ]

    for component in components:
        if os.path.exists(component):
            print(f"✅ {component}")
        else:
            print(f"❌ {component} - Missing")

    # Check research outputs
    print("\n📋 RESEARCH OUTPUTS:")
    output_dirs = ['research_output', 'valency_reports', 'diachronic_analysis']
    for dir_name in output_dirs:
        if os.path.exists(dir_name):
            files = list(Path(dir_name).glob("*"))
            print(f"📁 {dir_name}: {len(files)} files")
        else:
            print(f"❌ {dir_name}: Directory not found")

    # Check git status
    print("\n🔄 GIT STATUS:")
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            changes = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            print(f"✅ Git repository: {changes} uncommitted changes")
        else:
            print("❌ Git repository issue")
    except:
        print("❌ Git not available or not initialized")

    # System recommendations
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Run continuous monitoring: python continuous_monitoring.py")
    print("2. Start dashboard: python progress_dashboard.py")
    print("3. Execute pipeline: python master_workflow_coordinator.py")
    print("4. Push to GitHub: git push origin master")
    print("5. Monitor progress: Check http://localhost:5000")

    print("\n" + "=" * 60)
    print(f"Status check completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_system_status()
