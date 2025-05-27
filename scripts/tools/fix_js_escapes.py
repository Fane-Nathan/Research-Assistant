#!/usr/bin/env python3
"""
Simple script to fix the JavaScript escape sequences in document_web_interface.py
"""

import os

# Read the file
file_path = '/workspaces/StudyAssistant/scripts/document_web_interface.py'
with open(file_path, 'r') as f:
    content = f.read()

# Replace problematic escape sequence
content = content.replace(r"return id.replace(/\//g, '_').replace(/:/g, '_');", 
                          r"return id.replace(/\\//g, '_').replace(/:/g, '_');")

# Write back to the file
with open(file_path, 'w') as f:
    f.write(content)

print("Fixed JavaScript escape sequences in", file_path)
