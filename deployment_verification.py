#!/usr/bin/env python3
"""
Final deployment verification script for Streamlit Cloud readiness.
Tests all critical components for async compatibility and deployment readiness.
"""

import asyncio
import sys
import os
import logging

# Add current directory to path
sys.path.append(os.getcwd())

# Test imports
def test_imports():
    print("🔍 Testing core imports...")
    try:
        from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher
        from hybrid_search_rag.config import Config
        import streamlit
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

# Test ResourceFetcher in sync context
def test_sync_resource_fetcher():
    print("\n🔍 Testing ResourceFetcher in sync context...")
    try:
        from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher
        rf = ResourceFetcher()
        print("✅ ResourceFetcher created successfully in sync context")
        print(f"   _playwright_init_attempted: {rf._playwright_init_attempted}")
        print(f"   playwright_page: {rf.playwright_page}")
        return True
    except Exception as e:
        print(f"❌ Sync ResourceFetcher error: {e}")
        return False

# Test ResourceFetcher in async context
async def test_async_resource_fetcher():
    print("\n🔍 Testing ResourceFetcher in async context...")
    try:
        from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher
        rf = ResourceFetcher()
        print("✅ ResourceFetcher created successfully in async context")
        print(f"   _playwright_init_attempted: {rf._playwright_init_attempted}")
        print(f"   playwright_page: {rf.playwright_page}")
        return True
    except Exception as e:
        print(f"❌ Async ResourceFetcher error: {e}")
        return False

# Test configuration files
def test_deployment_files():
    print("\n🔍 Testing deployment configuration files...")
    files_to_check = [
        'requirements_streamlit.txt',
        'packages_streamlit.txt', 
        'STREAMLIT_DEPLOYMENT.md',
        'app.py'
    ]
    
    all_good = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    return all_good

def main():
    print("=" * 70)
    print("🚀 FINAL DEPLOYMENT VERIFICATION")
    print("=" * 70)
    
    results = []
    
    # Test all components
    results.append(test_imports())
    results.append(test_sync_resource_fetcher())
    results.append(asyncio.run(test_async_resource_fetcher()))
    results.append(test_deployment_files())
    
    print("\n" + "=" * 70)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 70)
    
    if all(results):
        print("🎯 ALL TESTS PASSED - READY FOR STREAMLIT CLOUD DEPLOYMENT!")
        print("\n📋 Next Steps:")
        print("1. Push code to GitHub repository")
        print("2. Go to share.streamlit.io")
        print("3. Configure with:")
        print("   - Main file: app.py")
        print("   - Requirements: requirements_streamlit.txt")
        print("   - Packages: packages_streamlit.txt")
        print("4. Add API keys in Streamlit Cloud secrets")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ISSUES BEFORE DEPLOYMENT")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
