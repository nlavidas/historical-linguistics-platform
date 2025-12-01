#!/usr/bin/env python3
"""
CODEBASE CONSOLIDATION SCRIPT
Analyzes and consolidates the corpus platform codebase
Removes redundant files and creates a clean structure
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

def get_file_hash(filepath):
    """Get MD5 hash of file content"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def analyze_codebase():
    """Analyze the codebase for duplicates and redundancy"""
    print("🔍 Analyzing codebase...")

    # Get all Python files
    py_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and not root.startswith('./__pycache__'):
                py_files.append(os.path.join(root, file))

    print(f"Found {len(py_files)} Python files")

    # Group by hash to find duplicates
    hash_groups = defaultdict(list)
    for filepath in py_files:
        file_hash = get_file_hash(filepath)
        if file_hash:
            hash_groups[file_hash].append(filepath)

    # Find duplicates
    duplicates = []
    for file_hash, files in hash_groups.items():
        if len(files) > 1:
            duplicates.append((file_hash, files))

    print(f"Found {len(duplicates)} duplicate groups")

    # Analyze web panel files specifically
    web_panels = [f for f in py_files if 'web' in f.lower() and 'panel' in f.lower()]
    print(f"Found {len(web_panels)} web panel files:")
    for panel in sorted(web_panels):
        print(f"  {panel}")

    return {
        'total_files': len(py_files),
        'duplicates': duplicates,
        'web_panels': web_panels
    }

def create_cleanup_recommendations(analysis):
    """Create recommendations for codebase cleanup"""
    print("\n🧹 CLEANUP RECOMMENDATIONS:")

    recommendations = []

    # Web panel consolidation
    if len(analysis['web_panels']) > 1:
        print(f"\n⚠️  MULTIPLE WEB PANELS DETECTED ({len(analysis['web_panels'])} files):")
        for panel in sorted(analysis['web_panels']):
            print(f"  ❌ {panel}")
        print("  ✅ RECOMMENDATION: Keep only 'unified_web_panel.py'")
        recommendations.append("Consolidate web panels into single unified_web_panel.py")

    # Duplicate files
    if analysis['duplicates']:
        print(f"\n⚠️  DUPLICATE FILES FOUND ({len(analysis['duplicates'])} groups):")
        for i, (file_hash, files) in enumerate(analysis['duplicates'][:5]):  # Show first 5
            print(f"  Group {i+1}: {len(files)} identical files")
            for file in files[:3]:  # Show first 3 files per group
                print(f"    {file}")
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more")
        recommendations.append("Remove duplicate files")

    # Large files that might be logs
    large_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > 50 * 1024 * 1024:  # 50MB
                    large_files.append((filepath, size))
            except:
                pass

    if large_files:
        print(f"\n⚠️  LARGE FILES DETECTED ({len(large_files)} files > 50MB):")
        for filepath, size in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {filepath}: {size / (1024*1024):.1f} MB")
        recommendations.append("Archive or clean large log files")

    return recommendations

def create_clean_structure():
    """Create a clean directory structure"""
    print("\n📁 PROPOSED CLEAN STRUCTURE:")

    clean_structure = {
        "core/": [
            "unified_web_panel.py",  # Single web interface
            "corpus_platform.db",   # Main database
            "requirements.txt"      # Dependencies
        ],
        "collection/": [
            "autonomous_247_collection.py",
            "diachronic_multilingual_collector.py"
        ],
        "analysis/": [
            "annotation_worker_247.py",
            "world_class_proiel_processor.py"
        ],
        "monitoring/": [
            "system_health_checker.py",
            "secure_monitoring_service.py"
        ],
        "deployment/": [
            "COMPLETE_AUTONOMOUS_DEPLOY.ps1",
            "autonomous_vm_setup.sh"
        ],
        "docs/": [
            "README.md",
            "PLATFORM_README.md",
            "SECURITY_DOCUMENTATION.md"
        ]
    }

    for directory, files in clean_structure.items():
        print(f"\n{directory}")
        for file in files:
            status = "✅" if os.path.exists(file) else "❌"
            print(f"  {status} {file}")

def main():
    """Main consolidation analysis"""
    print("🧹 CORPUS PLATFORM CODEBASE CONSOLIDATION ANALYSIS")
    print("=" * 60)

    analysis = analyze_codebase()
    recommendations = create_cleanup_recommendations(analysis)
    create_clean_structure()

    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"Total Python files: {analysis['total_files']}")
    print(f"Duplicate groups: {len(analysis['duplicates'])}")
    print(f"Web panel files: {len(analysis['web_panels'])}")
    print(f"Cleanup recommendations: {len(recommendations)}")

    print("\n🎯 ACTION ITEMS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print("\n💡 NEXT STEPS:")
    print("1. Review and approve cleanup recommendations")
    print("2. Backup important files before cleanup")
    print("3. Run consolidation script")
    print("4. Test unified web panel")
    print("5. Update deployment scripts")

if __name__ == "__main__":
    main()
