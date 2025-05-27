#!/usr/bin/env python3
# Update the combined_metadata.json file to include 'entry_id' field 
# based on the existing 'arxiv_entry_id' field

import json
import os

# Path to the metadata file
METADATA_FILE = '/workspaces/StudyAssistant/data_hybrid/combined_metadata.json'

def update_metadata():
    # Load the existing metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Check if any items are missing the 'entry_id' field
    needs_update = False
    for item in metadata:
        if 'entry_id' not in item and 'arxiv_entry_id' in item:
            needs_update = True
            break
    
    if not needs_update:
        print("No updates needed. All metadata items already have 'entry_id'.")
        return
    
    # Add 'entry_id' field based on 'arxiv_entry_id' to each item
    updated_count = 0
    for item in metadata:
        if 'entry_id' not in item:
            # If arxiv_entry_id exists, use it as entry_id
            if 'arxiv_entry_id' in item:
                item['entry_id'] = item['arxiv_entry_id']
                updated_count += 1
            # If chunk_id exists, use it as entry_id (fallback)
            elif 'chunk_id' in item:
                item['entry_id'] = item['chunk_id']
                updated_count += 1
            # Otherwise assign a placeholder ID
            else:
                item['entry_id'] = f"unknown_id_{metadata.index(item)}"
                updated_count += 1
    
    # Create a backup of the original file
    backup_file = f"{METADATA_FILE}.bak"
    os.rename(METADATA_FILE, backup_file)
    
    # Save the updated metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Update complete. Added 'entry_id' field to {updated_count} items.")
    print(f"Backup of original file saved to {backup_file}")

if __name__ == "__main__":
    update_metadata()
