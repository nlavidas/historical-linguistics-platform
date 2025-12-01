#!/usr/bin/env python3
"""
COMPREHENSIVE PLATFORM TEST SUITE
Tests all components of the historical linguistics platform
"""

import os
import sys
import sqlite3
import subprocess
import requests
from pathlib import Path

def test_web_panel():
    """Test web panel accessibility"""
    print("=== WEB PANEL TEST ===")
    try:
        response = requests.get("http://135.125.216.3/login", timeout=10)
        if "Secure Corpus Platform" in response.text:
            print("✅ Web panel accessible")
            return True
        else:
            print("❌ Web panel not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Web panel error: {e}")
        return False

def test_database():
    """Test database integrity"""
    print("\n=== DATABASE TEST ===")
    db_path = "corpus_platform.db"
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Found tables: {tables}")

        if 'corpus_items' in tables:
            cursor.execute("SELECT COUNT(*) FROM corpus_items")
            count = cursor.fetchone()[0]
            print(f"✅ Corpus items: {count}")

            cursor.execute("SELECT status, COUNT(*) FROM corpus_items GROUP BY status")
            for status, cnt in cursor.fetchall():
                print(f"  {status}: {cnt}")

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_packages():
    """Test Python package imports"""
    print("\n=== PYTHON PACKAGES TEST ===")

    packages = [
        ('sqlite3', 'Database'),
        ('flask', 'Web Framework'),
        ('requests', 'HTTP Client'),
    ]

    optional_packages = [
        ('stanza', 'NLP Engine'),
        ('cltk', 'Classical Languages'),
        ('crewai', 'AI Agents'),
    ]

    all_ok = True

    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {description}: OK")
        except ImportError:
            print(f"❌ {description}: MISSING")
            all_ok = False

    print("\nOptional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {description}: OK")
        except ImportError:
            print(f"⚠️  {description}: Not installed")

    return all_ok

def test_files():
    """Test important files exist"""
    print("\n=== FILE SYSTEM TEST ===")

    important_files = [
        "README.md",
        "requirements.txt",
        "corpus_platform.db",
        "secure_web_panel.py",
        "autonomous_platform_builder.py",
    ]

    all_exist = True
    for file in important_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file}: {size} bytes")
        else:
            print(f"❌ {file}: MISSING")
            all_exist = False

    return all_exist

def test_scripts():
    """Test script executability"""
    print("\n=== SCRIPT TEST ===")

    scripts = [
        "run_professional_cycle.py",
        "test_platform.py",
    ]

    for script in scripts:
        if os.path.exists(script):
            try:
                # Just check syntax
                result = subprocess.run([sys.executable, "-m", "py_compile", script],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ {script}: Syntax OK")
                else:
                    print(f"❌ {script}: Syntax error")
            except Exception as e:
                print(f"❌ {script}: Error - {e}")
        else:
            print(f"⚠️  {script}: Not found")

def main():
    """Run all tests"""
    print("🔬 COMPREHENSIVE PLATFORM TEST SUITE")
    print("=" * 50)

    tests = [
        ("Web Panel", test_web_panel),
        ("Database", test_database),
        ("Python Packages", test_packages),
        ("File System", test_files),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            results[name] = False

    test_scripts()  # Run separately

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CORE TESTS PASSED!")
        print("Your platform is in good health.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Please review the errors above.")

    return all_passed

if __name__ == "__main__":
    main()
