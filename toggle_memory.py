import os
import shutil
import argparse

# Files that need to be swapped
FILES_TO_SWAP = [
    'nets/graph_layers.py',
    'nets/actor_network.py',
    'agent/ppo.py'
]

def backup_original_files():
    """Backup original NeuOpt files if backups don't exist yet"""
    for file_path in FILES_TO_SWAP:
        backup_path = f"{file_path}.original"
        if not os.path.exists(backup_path):
            if os.path.exists(file_path):
                print(f"Backing up original file: {file_path} -> {backup_path}")
                shutil.copy2(file_path, backup_path)
            else:
                print(f"Warning: Original file {file_path} not found!")

def backup_memory_files():
    """Backup memory-augmented files if backups don't exist yet"""
    for file_path in FILES_TO_SWAP:
        backup_path = f"{file_path}.memory"
        if not os.path.exists(backup_path):
            if os.path.exists(file_path):
                print(f"Backing up memory-augmented file: {file_path} -> {backup_path}")
                shutil.copy2(file_path, backup_path)
            else:
                print(f"Warning: Memory-augmented file {file_path} not found!")

def enable_memory_mode():
    """Switch to memory-augmented NeuOpt version"""
    for file_path in FILES_TO_SWAP:
        memory_path = f"{file_path}.memory"
        if os.path.exists(memory_path):
            print(f"Enabling memory mode: {memory_path} -> {file_path}")
            shutil.copy2(memory_path, file_path)
        else:
            print(f"Warning: Memory-augmented file backup {memory_path} not found!")

def disable_memory_mode():
    """Switch to original NeuOpt version"""
    for file_path in FILES_TO_SWAP:
        original_path = f"{file_path}.original"
        if os.path.exists(original_path):
            print(f"Disabling memory mode: {original_path} -> {file_path}")
            shutil.copy2(original_path, file_path)
        else:
            print(f"Warning: Original file backup {original_path} not found!")

def toggle_memory_mode(enable=True):
    """Toggle between original and memory-augmented versions"""
    if enable:
        enable_memory_mode()
    else:
        disable_memory_mode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toggle between original and memory-augmented NeuOpt versions')
    parser.add_argument('--mode', choices=['backup', 'memory', 'original'], required=True,
                        help='backup: create both backups if needed; memory: enable memory mode; original: disable memory mode')
    
    args = parser.parse_args()
    
    if args.mode == 'backup':
        backup_original_files()
        backup_memory_files()
    elif args.mode == 'memory':
        toggle_memory_mode(enable=True)
    elif args.mode == 'original':
        toggle_memory_mode(enable=False) 