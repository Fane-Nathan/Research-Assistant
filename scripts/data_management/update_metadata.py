#!/usr/bin/env python3
# Update the combined_metadata.json file to include 'entry_id' field 
# based on the existing 'arxiv_entry_id' field

import json
import os

METADATA_FILE = '/workspaces/StudyAssistant/data_hybrid/combined_metadata.json'

def update_metadata():
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    needs_update = False
    for item in metadata:
        if 'entry_id' not in item and 'arxiv_entry_id' in item:
            needs_update = True
            break
    
    if not needs_update:
        print("No updates needed. All metadata items already have 'entry_id'.")
        return
    
    updated_count = 0
    for item in metadata:
        if 'entry_id' not in item:
            if 'arxiv_entry_id' in item:
                item['entry_id'] = item['arxiv_entry_id']
                updated_count += 1
            elif 'chunk_id' in item:
                item['entry_id'] = item['chunk_id']
                updated_count += 1
            else:
                item['entry_id'] = f"unknown_id_{metadata.index(item)}"
                updated_count += 1
    
    backup_file = f"{METADATA_FILE}.bak"
    os.rename(METADATA_FILE, backup_file)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Update complete. Added 'entry_id' field to {updated_count} items.")
    print(f"Backup of original file saved to {backup_file}")

if __name__ == "__main__":
    update_metadata()
