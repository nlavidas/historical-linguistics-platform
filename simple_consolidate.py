#!/usr/bin/env python3
"""
CODEBASE CONSOLIDATION SCRIPT
Analyzes and consolidates the corpus platform codebase
"""

import os
import hashlib
from collections import defaultdict

def analyze_codebase():
    """Analyze the codebase for duplicates and redundancy"""
    print("Analyzing codebase...")

    # Get all Python files
    py_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and not root.startswith('./__pycache__'):
                py_files.append(os.path.join(root, file))

    print(f"Found {len(py_files)} Python files")

    # Analyze web panel files specifically
    web_panels = [f for f in py_files if 'web' in f.lower() and 'panel' in f.lower()]
    print(f"Found {len(web_panels)} web panel files:")
    for panel in sorted(web_panels):
        print(f"  {panel}")

    return {
        'total_files': len(py_files),
        'web_panels': web_panels
    }

def main():
    """Main consolidation analysis"""
    print("CORPUS PLATFORM CODEBASE CONSOLIDATION ANALYSIS")
    print("=" * 50)

    analysis = analyze_codebase()

    print("\nCLEANUP RECOMMENDATIONS:")
    print(f"- Multiple web panels detected: {len(analysis['web_panels'])} files")
    print("- Recommendation: Keep only 'unified_web_panel.py'")

    print("\nSUMMARY:")
    print(f"Total Python files: {analysis['total_files']}")
    print(f"Web panel files: {len(analysis['web_panels'])}")

if __name__ == "__main__":
    main()
