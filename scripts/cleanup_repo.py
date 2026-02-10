#!/usr/bin/env python3
"""Utility to clean local repository artifacts (safe, optional).

Usage:
  python scripts/cleanup_repo.py [--remove-venv]

This script will remove `__pycache__` directories and optionally remove common
virtualenv folders. It will NOT delete tracked git objects. Run locally before
committing to reduce noise in the repo.
"""
import os
import shutil
import argparse


def remove_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")


def find_and_remove_pycache(root):
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "__pycache__" in dirnames:
            target = os.path.join(dirpath, "__pycache__")
            remove_path(target)
            removed += 1
    print(f"Removed {removed} __pycache__ directories under {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-venv", action="store_true", help="Remove common venv folders (venv, venv310)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    find_and_remove_pycache(repo_root)

    if args.remove_venv:
        for name in ("venv", "venv310", ".venv"):
            p = os.path.join(repo_root, name)
            if os.path.exists(p):
                remove_path(p)

    print("Cleanup complete. Review changes, then run `git status` to verify before committing.")


if __name__ == '__main__':
    main()
