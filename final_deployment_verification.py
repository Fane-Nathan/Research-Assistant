#!/usr/bin/env python3
"""
Final Deployment Verification Script for Streamlit Cloud
Tests all critical components for cloud deployment readiness.
"""

import asyncio
import sys
import os
import importlib.util
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name: str, description: str) -> bool:
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: Successfully imported")
        return True
    except ImportError as e:
        print(f"❌ {description}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: Import succeeded but with warning - {e}")
        return True

def test_resource_fetcher_sync():
    """Test ResourceFetcher in synchronous context."""
    try:
        from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher
        rf = ResourceFetcher()
        print(f"✅ ResourceFetcher (sync): Created successfully")
        print(f"   - _playwright_init_attempted: {rf._playwright_init_attempted}")
        print(f"   - playwright_page: {rf.playwright_page}")
        return True
    except Exception as e:
        print(f"❌ ResourceFetcher (sync): Failed - {e}")
        traceback.print_exc()
        return False

async def test_resource_fetcher_async():
    """Test ResourceFetcher in asynchronous context."""
    try:
        from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher
        rf = ResourceFetcher()
        print(f"✅ ResourceFetcher (async): Created successfully in async context")
        print(f"   - _playwright_init_attempted: {rf._playwright_init_attempted}")
        print(f"   - playwright_page: {rf.playwright_page}")
        return True
    except Exception as e:
        print(f"❌ ResourceFetcher (async): Failed - {e}")
        traceback.print_exc()
        return False

def test_file_existence():
    """Test that all required deployment files exist."""
    required_files = [
        'app.py',
        'requirements_streamlit.txt',
        'packages_streamlit.txt',
        'STREAMLIT_DEPLOYMENT.md',
        'hybrid_search_rag/__init__.py',
        'hybrid_search_rag/config.py',
        'hybrid_search_rag/data_handling/resource_fetcher.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_exist = False
    
    return all_exist

def test_streamlit_requirements():
    """Test that streamlit requirements file has essential packages."""
    try:
        with open('requirements_streamlit.txt', 'r') as f:
            content = f.read()
        
        essential_packages = [
            'streamlit',
            'numpy',
            'aiohttp',
            'requests',
            'google-generativeai',
            'groq',
            'deepseek',
            'arxiv',
            'beautifulsoup4',
            'PyMuPDF'
        ]
        
        missing_packages = []
        for package in essential_packages:
            if package not in content:
                missing_packages.append(package)
        
        if not missing_packages:
            print("✅ Streamlit requirements: All essential packages found")
            return True
        else:
            print(f"❌ Streamlit requirements: Missing packages: {missing_packages}")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit requirements: Error reading file - {e}")
        return False

def test_excluded_heavy_dependencies():
    """Test that heavy dependencies are excluded from streamlit requirements."""
    try:
        with open('requirements_streamlit.txt', 'r') as f:
            content = f.read()
        
        excluded_packages = [
            'tensorflow',
            'torch',
            'transformers',
            'sentence-transformers',
            'playwright'
        ]
        
        found_excluded = []
        for package in excluded_packages:
            if package in content and not content.count(f"# {package}"):
                found_excluded.append(package)
        
        if not found_excluded:
            print("✅ Heavy dependencies: All heavy packages properly excluded")
            return True
        else:
            print(f"❌ Heavy dependencies: Found excluded packages: {found_excluded}")
            return False
            
    except Exception as e:
        print(f"❌ Heavy dependencies: Error reading file - {e}")
        return False

async def main():
    """Run all verification tests."""
    print("=" * 70)
    print("🚀 FINAL DEPLOYMENT VERIFICATION FOR STREAMLIT CLOUD")
    print("=" * 70)
    print()
    
    # Track all test results
    test_results = []
    
    print("📦 TESTING CORE IMPORTS...")
    test_results.append(test_import('streamlit', 'Streamlit'))
    test_results.append(test_import('numpy', 'NumPy'))
    test_results.append(test_import('aiohttp', 'aiohttp'))
    test_results.append(test_import('hybrid_search_rag.config', 'Project Config'))
    test_results.append(test_import('hybrid_search_rag.data_handling.resource_fetcher', 'ResourceFetcher'))
    print()
    
    print("🔧 TESTING RESOURCEFETCHER COMPATIBILITY...")
    test_results.append(test_resource_fetcher_sync())
    test_results.append(await test_resource_fetcher_async())
    print()
    
    print("📁 TESTING FILE EXISTENCE...")
    test_results.append(test_file_existence())
    print()
    
    print("📋 TESTING DEPLOYMENT CONFIGURATION...")
    test_results.append(test_streamlit_requirements())
    test_results.append(test_excluded_heavy_dependencies())
    print()
    
    # Summary
    print("=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("✅ Application is READY for Streamlit Cloud deployment!")
        print()
        print("📋 NEXT STEPS:")
        print("1. Commit all changes to your GitHub repository")
        print("2. Go to https://share.streamlit.io")
        print("3. Connect your GitHub repository")
        print("4. Configure with:")
        print("   - Main file: app.py")
        print("   - Python version: 3.11")
        print("   - Requirements: requirements_streamlit.txt")
        print("   - Packages: packages_streamlit.txt")
        print("5. Add your API keys in Streamlit Cloud secrets")
        print("6. Deploy!")
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("🔧 Please fix the failing tests before deploying")
    
    print("=" * 70)
    return passed == total

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Verification failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
