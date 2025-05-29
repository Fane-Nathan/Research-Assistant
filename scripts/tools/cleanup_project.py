#!/usr/bin/env python3
"""
Project Cleanup Script for StudyAssistant

This script provides automated cleanup functionality for the StudyAssistant project,
removing temporary files, old logs, and Python cache files.

Usage:
    python scripts/tools/cleanup_project.py [--dry-run] [--days DAYS]

Args:
    --dry-run: Show what would be cleaned without actually removing files
    --days: Number of days to keep log files (default: 30)
"""

import argparse
import os
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set


def get_project_root() -> Path:
    """Get the project root directory."""
    current_dir = Path(__file__).parent
    # Navigate up to find the project root (where app.py is located)
    while current_dir.parent != current_dir:
        if (current_dir / "app.py").exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Could not find project root directory")


def find_files_to_clean(project_root: Path, days_to_keep: int) -> tuple[List[Path], List[Path], List[Path]]:
    """
    Find files that should be cleaned up.
    
    Returns:
        tuple: (cache_files, old_logs, temp_files)
    """
    cache_files = []
    old_logs = []
    temp_files = []
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # Find Python cache files
    for cache_dir in project_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            cache_files.append(cache_dir)
    
    for pyc_file in project_root.rglob("*.pyc"):
        cache_files.append(pyc_file)
    
    # Find old log files
    log_patterns = ["*.log", "*.log.txt", "logs-*.txt"]
    for pattern in log_patterns:
        for log_file in project_root.rglob(pattern):
            if log_file.is_file():
                try:
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        old_logs.append(log_file)
                except OSError:
                    # Skip files we can't stat
                    pass
    
    # Find temporary test/verification files
    temp_patterns = [
        "temp_files/test_*.py",
        "temp_files/final_*.py", 
        "temp_files/*verification*.py",
        "temp_files/*diagnose*.py",
        "temp_files/logs-*.txt",
        "test_*.py",  # Root level temp test files
        "final_*.py",  # Root level temp final files
    ]
    
    for pattern in temp_patterns:
        for temp_file in project_root.glob(pattern):
            if temp_file.is_file():
                temp_files.append(temp_file)
    
    return cache_files, old_logs, temp_files


def clean_files(files_to_remove: List[Path], dry_run: bool = False) -> int:
    """
    Remove the specified files and directories.
    
    Args:
        files_to_remove: List of Path objects to remove
        dry_run: If True, only show what would be removed
        
    Returns:
        Number of items successfully removed (or would be removed in dry-run)
    """
    removed_count = 0
    
    for file_path in files_to_remove:
        try:
            if dry_run:
                if file_path.is_dir():
                    print(f"Would remove directory: {file_path}")
                else:
                    print(f"Would remove file: {file_path}")
                removed_count += 1
            else:
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
                else:
                    file_path.unlink()
                    print(f"Removed file: {file_path}")
                removed_count += 1
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not remove {file_path}: {e}")
    
    return removed_count


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up temporary files, logs, and cache from StudyAssistant project"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be cleaned without actually removing files"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Number of days to keep log files (default: 30)"
    )
    
    args = parser.parse_args()
    
    try:
        project_root = get_project_root()
        print(f"Project root: {project_root}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No files will actually be removed")
        
        print(f"\n🧹 Finding files to clean (keeping logs newer than {args.days} days)...")
        
        cache_files, old_logs, temp_files = find_files_to_clean(project_root, args.days)
        
        total_files = len(cache_files) + len(old_logs) + len(temp_files)
        
        if total_files == 0:
            print("✅ No files found to clean. Project is already tidy!")
            return
        
        print(f"\n📊 Found {total_files} items to clean:")
        print(f"  - {len(cache_files)} Python cache files/directories")
        print(f"  - {len(old_logs)} old log files")
        print(f"  - {len(temp_files)} temporary files")
        
        if not args.dry_run:
            response = input("\nProceed with cleanup? (y/N): ")
            if response.lower() != 'y':
                print("Cleanup cancelled.")
                return
        
        print(f"\n🗑️  Cleaning cache files...")
        cache_removed = clean_files(cache_files, args.dry_run)
        
        print(f"\n🗑️  Cleaning old logs...")
        logs_removed = clean_files(old_logs, args.dry_run)
        
        print(f"\n🗑️  Cleaning temporary files...")
        temp_removed = clean_files(temp_files, args.dry_run)
        
        total_removed = cache_removed + logs_removed + temp_removed
        
        action = "Would clean" if args.dry_run else "Successfully cleaned"
        print(f"\n✅ {action} {total_removed} items:")
        print(f"  - {cache_removed} cache items")
        print(f"  - {logs_removed} old logs") 
        print(f"  - {temp_removed} temporary files")
        
        if not args.dry_run:
            print(f"\n💡 To prevent future clutter:")
            print(f"  - Run this script regularly: python scripts/tools/cleanup_project.py")
            print(f"  - Use 'git clean -fd' to remove untracked files")
            print(f"  - Keep temp files in temp_files/ directory only")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
