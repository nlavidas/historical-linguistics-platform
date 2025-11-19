"""
Validation and Fix Script for HFRI-NKUA Corpus Platform
Checks for common issues and provides fixes
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected - need Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_imports():
    """Check if required packages can be imported"""
    packages = {
        'aiohttp': 'pip install aiohttp',
        'stanza': 'pip install stanza',
        'fastapi': 'pip install fastapi',
        'uvicorn': 'pip install uvicorn',
    }
    
    optional_packages = {
        'spacy': 'pip install spacy',
        'transformers': 'pip install transformers',
        'nltk': 'pip install nltk',
        'textblob': 'pip install textblob',
        'trankit': 'pip install trankit',
        'ollama': 'pip install ollama',
    }
    
    print("\n📦 Checking required packages:")
    all_ok = True
    for package, install_cmd in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Install: {install_cmd}")
            all_ok = False
    
    print("\n📦 Checking optional packages (for multi-AI):")
    optional_available = 0
    for package, install_cmd in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}")
            optional_available += 1
        except ImportError:
            print(f"  ⚠️  {package} - Install: {install_cmd}")
    
    print(f"\nOptional models available: {optional_available}/{len(optional_packages)}")
    return all_ok

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking files:")
    required_files = [
        'unified_corpus_platform.py',
        'multi_ai_annotator.py',
        'integrated_platform.py',
        'web_dashboard.py',
        'test_platform.py',
        'requirements.txt',
        'requirements_full.txt',
    ]
    
    all_ok = True
    for filename in required_files:
        path = Path(filename)
        if path.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - Missing!")
            all_ok = False
    
    return all_ok

def check_syntax():
    """Check Python syntax of main files"""
    print("\n🔍 Checking Python syntax:")
    files_to_check = [
        'unified_corpus_platform.py',
        'multi_ai_annotator.py',
        'integrated_platform.py',
        'web_dashboard.py',
    ]
    
    all_ok = True
    for filename in files_to_check:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                compile(f.read(), filename, 'exec')
            print(f"  ✅ {filename}")
        except SyntaxError as e:
            print(f"  ❌ {filename} - Syntax error: {e}")
            all_ok = False
        except FileNotFoundError:
            print(f"  ⚠️  {filename} - File not found")
    
    return all_ok

def check_imports_in_code():
    """Check if imports work in the actual code"""
    print("\n🔌 Testing module imports:")
    
    tests = [
        ('unified_corpus_platform', 'UnifiedCorpusPlatform'),
        ('multi_ai_annotator', 'MultiAIAnnotator'),
        ('integrated_platform', 'IntegratedPlatform'),
    ]
    
    all_ok = True
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name} - Error: {e}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check and create necessary directories"""
    print("\n📂 Checking directories:")
    directories = [
        'data',
        'data/raw',
        'data/parsed',
        'data/annotated',
    ]
    
    for dirname in directories:
        path = Path(dirname)
        if path.exists():
            print(f"  ✅ {dirname}")
        else:
            print(f"  ⚠️  {dirname} - Creating...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dirname} - Created")
    
    return True

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test:")
    
    try:
        from multi_ai_annotator import MultiAIAnnotator
        annotator = MultiAIAnnotator()
        available = len(annotator.available_models)
        print(f"  ✅ Multi-AI Annotator initialized")
        print(f"  ✅ Available AI models: {available}")
        
        if available == 0:
            print(f"  ⚠️  No AI models available - install at least Stanza")
            print(f"     Run: pip install stanza")
            print(f"     Then: python -c \"import stanza; stanza.download('en')\"")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def print_fix_recommendations():
    """Print recommendations for fixing issues"""
    print("\n" + "=" * 70)
    print("🔧 FIX RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1️⃣  Install all required packages:")
    print("   pip install -r requirements_full.txt")
    
    print("\n2️⃣  Download AI models (at minimum):")
    print("   python -c \"import stanza; stanza.download('en'); stanza.download('grc')\"")
    
    print("\n3️⃣  Test installation:")
    print("   python test_platform.py")
    
    print("\n4️⃣  Start the platform:")
    print("   python web_dashboard.py")
    print("   OR")
    print("   python integrated_platform.py --status")
    
    print("\n5️⃣  For 24/7 operation:")
    print("   START_24_7_NOW.bat")

def main():
    """Main validation routine"""
    print("=" * 70)
    print("HFRI-NKUA AI CORPUS PLATFORM - VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Files", check_files()))
    results.append(("Syntax", check_syntax()))
    results.append(("Directories", check_directories()))
    results.append(("Packages", check_imports()))
    results.append(("Module Imports", check_imports_in_code()))
    results.append(("Functionality", run_quick_test()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    print(f"\nScore: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED! Platform is ready to use.")
        print("\nStart with:")
        print("  python web_dashboard.py")
        print("  OR")
        print("  START_24_7_NOW.bat")
        return 0
    else:
        print(f"\n⚠️  {total - passed} checks failed. See recommendations below.")
        print_fix_recommendations()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print("\n" + "=" * 70)
    input("Press Enter to exit...")
    sys.exit(exit_code)
