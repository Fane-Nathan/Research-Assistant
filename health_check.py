#!/usr/bin/env python3
"""
StudyAssistant Project Health Check
This script verifies the project's health by checking dependencies, 
file structure, and common issues.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Python Version Check")
    print(f"   Current version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version compatible")
        return True

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n📦 Dependency Check")
    
    # Core dependencies with their actual import names
    required_packages = {
        'streamlit': 'streamlit',
        'langchain': 'langchain',
        'openai': 'openai', 
        'anthropic': 'anthropic',
        'chromadb': 'chromadb',
        'sentence-transformers': 'sentence_transformers',
        'PyPDF2': 'PyPDF2',
        'python-docx': 'docx',  # python-docx imports as 'docx'
        'openpyxl': 'openpyxl',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # scikit-learn imports as 'sklearn'
        'faiss-cpu': 'faiss'  # faiss-cpu imports as 'faiss'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (missing)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("   ✅ All dependencies installed")
        return True

def check_file_structure():
    """Check if essential files and directories exist."""
    print("\n📁 File Structure Check")
    
    essential_files = [
        'app.py',
        'requirements.txt',
        '.gitignore',
        'hybrid_search_rag/__init__.py',
        'docs/PROJECT_CLEANUP.md'
    ]
    
    essential_dirs = [
        'hybrid_search_rag',
        'docs',
        'logs'
    ]
    
    all_good = True
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_good = False
    
    for dir_path in essential_dirs:
        if os.path.isdir(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (missing)")
            all_good = False
    
    return all_good

def check_configuration():
    """Check configuration files for common issues."""
    print("\n⚙️  Configuration Check")
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if len(content.strip()) > 0:
                print("   ✅ requirements.txt exists and has content")
            else:
                print("   ❌ requirements.txt is empty")
                return False
    else:
        print("   ❌ requirements.txt missing")
        return False
    
    # Check for environment variables or config files
    if os.path.exists('.env') or os.path.exists('config.py'):
        print("   ✅ Configuration files present")
    else:
        print("   ⚠️  No .env or config.py found (may need API keys)")
    
    return True

def check_code_quality():
    """Run basic code quality checks."""
    print("\n🔍 Code Quality Check")
    
    # Check if main app.py can be imported
    try:
        sys.path.insert(0, '.')
        import app
        print("   ✅ app.py imports successfully")
        return True
    except Exception as e:
        print(f"   ❌ app.py import failed: {str(e)}")
        return False

def cleanup_project():
    """Clean up temporary files and cache."""
    print("\n🧹 Cleanup Check")
    
    cleanup_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        'temp_files/test_*',
        'temp_files/*verification*'
    ]
    
    cleaned = False
    for pattern in cleanup_patterns:
        if pattern == '__pycache__':
            # Remove __pycache__ directories
            for root, dirs, files in os.walk('.'):
                if '__pycache__' in dirs:
                    cache_dir = os.path.join(root, '__pycache__')
                    try:
                        import shutil
                        shutil.rmtree(cache_dir)
                        print(f"   🗑️  Removed {cache_dir}")
                        cleaned = True
                    except Exception:
                        pass
    
    if not cleaned:
        print("   ✅ No cleanup needed")
    
    return True

def main():
    """Run all health checks."""
    print("🏥 StudyAssistant Health Check")
    print("=" * 40)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_file_structure(),
        check_configuration(),
        check_code_quality(),
        cleanup_project()
    ]
    
    print("\n" + "=" * 40)
    
    if all(checks):
        print("🎉 Project health: EXCELLENT")
        print("   All checks passed! Your project is ready to run.")
    else:
        print("⚠️  Project health: NEEDS ATTENTION")
        print("   Some issues found. Please address the items marked with ❌")
    
    print("\nTo run the application:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
