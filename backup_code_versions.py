#!/usr/bin/env python
"""
This script creates the necessary backups for comparison:
1. Backup of original NeuOpt code
2. Backup of memory-augmented NeuOpt code
"""
import os
import sys
import shutil
from toggle_memory import FILES_TO_SWAP

def main():
    # 1. Make sure we have the memory-augmented version in the working directory
    print("Creating backup of memory-augmented version...")
    for file_path in FILES_TO_SWAP:
        backup_path = f"{file_path}.memory"
        if os.path.exists(file_path):
            print(f"Backing up memory-augmented file: {file_path} -> {backup_path}")
            shutil.copy2(file_path, backup_path)
        else:
            print(f"Warning: File not found: {file_path}")
            return 1
    
    # 2. Check if we have original versions or need to recreate them
    missing_originals = False
    for file_path in FILES_TO_SWAP:
        backup_path = f"{file_path}.original"
        if not os.path.exists(backup_path):
            missing_originals = True
            break
    
    if missing_originals:
        print("\nSome original file backups are missing!")
        print("To proceed, you need original versions of these files:")
        for file_path in FILES_TO_SWAP:
            print(f"  - {file_path}")
        
        response = input("\nDo you want to try to recreate original versions by reverting your changes? (y/n): ")
        if response.lower() == 'y':
            # Try to recreate originals by reverting memory-related changes
            print("\nThis feature is not yet implemented.")
            print("Please manually restore original files and run this script again.")
            return 1
        else:
            print("\nPlease manually create the original file backups and try again.")
            return 1
    else:
        print("\nAll necessary backups exist.")
        print("Ready to run comparison!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 