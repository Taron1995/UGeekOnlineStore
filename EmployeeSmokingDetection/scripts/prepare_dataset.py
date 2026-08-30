"""Standalone entry point for splitting data/raw into data/split/{train,val,test}.

Usage:
    python scripts/prepare_dataset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import split_dataset

if __name__ == "__main__":
    split_dataset()
