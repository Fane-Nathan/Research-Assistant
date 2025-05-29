@echo off
REM StudyAssistant Project Management Script
REM Usage: manage.bat [command]
REM Commands: cleanup, health, install, run, help

if "%1"=="" (
    echo StudyAssistant Project Management
    echo.
    echo Available commands:
    echo   cleanup  - Remove cache files, logs, and temporary files
    echo   health   - Check project health and dependencies
    echo   install  - Install/update dependencies
    echo   run      - Start the application
    echo   help     - Show this help message
    echo.
    echo Usage: manage.bat [command]
    exit /b 0
)

if "%1"=="help" (
    echo StudyAssistant Project Management
    echo.
    echo Available commands:
    echo   cleanup  - Remove cache files, logs, and temporary files
    echo   health   - Check project health and dependencies
    echo   install  - Install/update dependencies
    echo   run      - Start the application
    echo   help     - Show this help message
    exit /b 0
)

if "%1"=="cleanup" (
    echo Cleaning up project files...
    echo Removing Python cache files...
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    del /q /s *.pyc >nul 2>&1
    
    echo Removing log files...
    if exist app.log del app.log
    if exist logs\*.log del logs\*.log
    
    echo Removing temporary files...
    if exist temp_files\test_* del temp_files\test_*
    if exist temp_files\*verification* del temp_files\*verification*
    
    echo Cleanup completed!
    exit /b 0
)

if "%1"=="health" (
    echo Checking project health...
    python -c "
import sys
import importlib
import pkg_resources

print('Python version:', sys.version)
print()

# Check required packages
required_packages = [
    'streamlit', 'langchain', 'openai', 'anthropic', 'chromadb',
    'sentence-transformers', 'PyPDF2', 'python-docx', 'openpyxl',
    'plotly', 'pandas', 'numpy', 'scikit-learn', 'faiss-cpu'
]

missing_packages = []
for package in required_packages:
    try:
        pkg_resources.get_distribution(package)
        print(f'✓ {package}')
    except pkg_resources.DistributionNotFound:
        print(f'✗ {package} (missing)')
        missing_packages.append(package)

if missing_packages:
    print()
    print('Missing packages:', ', '.join(missing_packages))
    print('Run: manage.bat install')
else:
    print()
    print('All dependencies are installed!')
"
    exit /b 0
)

if "%1"=="install" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Dependencies installed!
    exit /b 0
)

if "%1"=="run" (
    echo Starting StudyAssistant...
    streamlit run app.py
    exit /b 0
)

echo Unknown command: %1
echo Use 'manage.bat help' for available commands.
exit /b 1
